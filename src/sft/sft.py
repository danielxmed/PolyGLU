"""Supervised fine-tuning script for PolychromaticLM.

Loads a pre-trained checkpoint, freezes tau at 0.1, and fine-tunes on
Nemotron-Math-v2 with loss masking on assistant tokens only.

Usage:
    deepspeed --num_gpus=1 -m src.sft.sft --config configs/sft_config.yaml
"""

import argparse
import json
import math
import os
import time

import deepspeed
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from src.model.config import ModelConfig, SFTConfig
from src.model.model import create_model, get_routing_entropy
from src.data.sft_dataset import SFTDataset


def create_optimizer(model: nn.Module, config: SFTConfig) -> torch.optim.AdamW:
    """AdamW with param group separation (matching pre-training pattern)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.peak_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sft_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss masked to assistant tokens only.

    Args:
        logits: (bs, seq_len, vocab_size)
        targets: (bs, seq_len)
        loss_mask: (bs, seq_len) with 1.0 for assistant tokens, 0.0 otherwise
    """
    vocab_size = logits.shape[-1]
    # Per-token loss
    per_token_loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)

    # Apply mask
    masked_loss = per_token_loss * loss_mask
    # Normalize by number of assistant tokens
    n_tokens = loss_mask.sum()
    if n_tokens > 0:
        return masked_loss.sum() / n_tokens
    return masked_loss.sum()


def train_sft(model_config: ModelConfig, sft_config: SFTConfig, local_rank: int = 0):
    """Main SFT training loop."""
    device = torch.device(f"cuda:{local_rank}")

    # --- Model from pre-trained checkpoint ---
    print(f"Loading pre-trained checkpoint: {sft_config.pretrained_checkpoint}")
    model = create_model(model_config)

    checkpoint = torch.load(sft_config.pretrained_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from step {checkpoint.get('step', '?')}")

    # Freeze tau at 0.1 (end of annealing)
    for block in model.model_core:
        block.polyglu.tau = sft_config.tau

    # --- Data ---
    print("Loading SFT dataset...")
    dataset = SFTDataset(
        dataset_id=sft_config.dataset_id,
        splits=sft_config.dataset_splits,
        max_seq_length=sft_config.max_seq_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=sft_config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Compute total steps
    steps_per_epoch = len(dataloader) // sft_config.grad_accum_steps
    total_steps = steps_per_epoch * sft_config.epochs
    print(f"Dataset size: {len(dataset)}, steps/epoch: {steps_per_epoch}, total_steps: {total_steps}")

    # --- Optimizer & Scheduler ---
    optimizer = create_optimizer(model, sft_config)
    scheduler = create_scheduler(optimizer, sft_config.warmup_steps, total_steps)

    # --- DeepSpeed ---
    with open(sft_config.ds_config) as f:
        ds_config = json.load(f)

    # Override batch config for SFT
    ds_config["gradient_accumulation_steps"] = sft_config.grad_accum_steps
    ds_config["train_micro_batch_size_per_gpu"] = sft_config.micro_batch_size

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )

    # --- WandB ---
    if local_rank == 0:
        wandb.init(
            project=sft_config.wandb_project,
            name=sft_config.wandb_run_name,
            config={
                "model": model_config.__dict__,
                "sft": sft_config.__dict__,
                "total_steps": total_steps,
            },
        )

    os.makedirs(sft_config.checkpoint_dir, exist_ok=True)

    # --- Training ---
    print(f"Starting SFT for {sft_config.epochs} epoch(s), {total_steps} steps")
    model_engine.train()

    global_step = 0
    accum_loss = 0.0

    for epoch in range(sft_config.epochs):
        for batch in dataloader:
            if global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # Forward (no document masking for SFT — each example is one conversation)
            logits = model_engine(input_ids)

            loss = sft_loss(logits, targets, loss_mask)
            loss = loss / sft_config.grad_accum_steps

            model_engine.backward(loss)
            accum_loss += loss.item()

            if model_engine.is_gradient_accumulation_boundary():
                model_engine.step()
                global_step += 1

                # Logging
                if global_step % sft_config.log_every == 0 and local_rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    entropy_dict = get_routing_entropy(model_engine.module)
                    wandb.log(
                        {
                            "sft/loss": accum_loss,
                            "sft/lr": lr,
                            "sft/step": global_step,
                            "sft/epoch": epoch,
                            **entropy_dict,
                        },
                        step=global_step,
                    )
                    print(
                        f"Epoch {epoch} Step {global_step}/{total_steps} | "
                        f"Loss: {accum_loss:.4f} | LR: {lr:.6f}"
                    )

                accum_loss = 0.0

                # Checkpointing
                if global_step % sft_config.checkpoint_every == 0:
                    ckpt_path = os.path.join(
                        sft_config.checkpoint_dir, f"sft_step{global_step}.pt"
                    )
                    torch.save(
                        {
                            "model": model_engine.module.state_dict(),
                            "step": global_step,
                            "tau": sft_config.tau,
                            "epoch": epoch,
                        },
                        ckpt_path,
                    )
                    print(f"Saved SFT checkpoint at step {global_step}")

    # --- Final checkpoint ---
    if local_rank == 0:
        final_path = os.path.join(sft_config.checkpoint_dir, "sft_final.pt")
        torch.save(
            {
                "model": model_engine.module.state_dict(),
                "step": global_step,
                "tau": sft_config.tau,
                "epoch": sft_config.epochs,
            },
            final_path,
        )
        print(f"SFT complete. Final checkpoint: {final_path}")
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="PolychromaticLM SFT")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    model_config = ModelConfig.from_yaml(args.config)
    sft_config = SFTConfig.from_yaml(args.config)

    train_sft(model_config, sft_config, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
