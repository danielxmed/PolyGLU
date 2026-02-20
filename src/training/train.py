"""Production pre-training loop for PolychromaticLM.

Replicates the frozen Trainer logic (full_model.py lines 254-348) with:
- DeepSpeed ZeRO Stage 2 integration
- Weights & Biases logging
- Document masking via Flash Attention
- Data mix annealing
- Dual checkpointing (DeepSpeed + portable .pt)
- Resume from checkpoint

Usage:
    deepspeed --num_gpus=1 -m src.training.train --config configs/train_config.yaml
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

from src.model.config import ModelConfig, TrainConfig
from src.model.model import create_model, get_routing_entropy
from src.data.dataloader import PretrainingDataLoader


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """Create AdamW optimizer with param group separation.

    Replicates frozen Trainer lines 260-273:
    - 2D+ params (weights): weight_decay = 0.1
    - 1D params (biases, norms, alpha, beta): weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.peak_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    return optimizer


def get_lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Cosine LR schedule with linear warmup.

    Replicates frozen Trainer lines 275-280.
    """
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


def create_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create LR scheduler matching frozen Trainer lines 282-286."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(step, warmup_steps, total_steps),
    )


def save_portable_checkpoint(
    model_engine, optimizer, scheduler, step: int, tau: float, path: str
):
    """Save portable checkpoint in frozen Trainer format (lines 296-303).

    This format is used for eval, SFT, and sharing. It does NOT include
    DeepSpeed state — use DeepSpeed's own checkpointing for resume.
    """
    # Extract the underlying model from DeepSpeed wrapper
    model = model_engine.module

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "tau": tau,
        },
        path,
    )


def train(model_config: ModelConfig, train_config: TrainConfig, local_rank: int = 0):
    """Main training loop."""
    device = torch.device(f"cuda:{local_rank}")

    # --- Model ---
    print(f"Creating model (vocab={model_config.vocab_size}, layers={model_config.n_layers})...")
    model = create_model(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config.warmup_steps, train_config.total_steps)

    # --- DeepSpeed ---
    with open(train_config.ds_config) as f:
        ds_config = json.load(f)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )

    # --- Resume ---
    start_step = 0
    if train_config.resume_from:
        # Try DeepSpeed checkpoint first
        ds_ckpt_dir = os.path.join(train_config.resume_from, "ds_checkpoint")
        if os.path.exists(ds_ckpt_dir):
            _, client_state = model_engine.load_checkpoint(ds_ckpt_dir)
            start_step = client_state.get("step", 0)
            tau = client_state.get("tau", 1.0)
            model_engine.module.update_tau(start_step, train_config.total_steps)
            print(f"Resumed from DeepSpeed checkpoint at step {start_step}")
        else:
            # Fall back to portable checkpoint
            checkpoint = torch.load(train_config.resume_from, map_location=device, weights_only=False)
            model_engine.module.load_state_dict(checkpoint["model"])
            start_step = checkpoint.get("step", 0)
            tau = checkpoint.get("tau", 1.0)
            model_engine.module.update_tau(start_step, train_config.total_steps)
            # Reload optimizer and scheduler state
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"Resumed from portable checkpoint at step {start_step}")

    # --- Data ---
    print("Creating dataloader...")
    dataloader = PretrainingDataLoader(
        data_dir=train_config.data_dir,
        math_subdir=train_config.math_subdir,
        stem_subdir=train_config.stem_subdir,
        code_subdir=train_config.code_subdir,
        seq_length=model_config.seq_length,
        micro_batch_size=train_config.micro_batch_size,
        total_steps=train_config.total_steps,
        grad_accum_steps=train_config.grad_accum_steps,
        mix_math=train_config.mix_math,
        mix_stem=train_config.mix_stem,
        mix_code=train_config.mix_code,
        anneal_start_frac=train_config.anneal_start_frac,
        anneal_math=train_config.anneal_math,
        anneal_stem=train_config.anneal_stem,
        anneal_code=train_config.anneal_code,
        start_step=start_step,
    )

    # --- WandB ---
    if local_rank == 0:
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
            config={
                "model": model_config.__dict__,
                "training": train_config.__dict__,
                "total_params": total_params,
            },
            resume="allow" if train_config.resume_from else None,
        )

    # --- Checkpoint dir ---
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    print(f"Starting training from step {start_step} to {train_config.total_steps}")
    model_engine.train()

    global_step = start_step
    accum_loss = 0.0
    micro_step = 0
    tokens_processed = 0
    step_start_time = time.time()

    for batch in dataloader:
        if global_step >= train_config.total_steps:
            break

        # Update mix ratios based on current step
        dataloader.set_step(global_step)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        cu_seqlens = batch["cu_seqlens"].to(device)
        max_seqlen = batch["max_seqlen"]

        # Forward pass with document masking
        logits = model_engine(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Loss — matching frozen Trainer lines 288-294
        vocab_size = logits.shape[-1]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1)
        )
        loss = loss / train_config.grad_accum_steps

        # Backward
        model_engine.backward(loss)
        accum_loss += loss.item()
        micro_step += 1
        tokens_processed += input_ids.numel()

        # Step — matching frozen Trainer lines 332-348
        if model_engine.is_gradient_accumulation_boundary():
            model_engine.step()

            # Tau annealing — matching frozen Trainer line 337
            model_engine.module.update_tau(global_step, train_config.total_steps)

            global_step += 1

            # Logging
            if global_step % train_config.log_every == 0 and local_rank == 0:
                elapsed = time.time() - step_start_time
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                tau = model_engine.module.model_core[0].polyglu.tau
                lr = scheduler.get_last_lr()[0]

                log_dict = {
                    "train/loss": accum_loss,
                    "train/lr": lr,
                    "train/tau": tau,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": global_step,
                }

                # Routing entropy (every log_every steps)
                entropy_dict = get_routing_entropy(model_engine.module)
                log_dict.update(entropy_dict)

                wandb.log(log_dict, step=global_step)

                print(
                    f"Step {global_step}/{train_config.total_steps} | "
                    f"Loss: {accum_loss:.4f} | LR: {lr:.6f} | "
                    f"τ: {tau:.3f} | tok/s: {tokens_per_sec:.0f} | "
                    f"entropy: {entropy_dict['routing_entropy/mean']:.3f}"
                )

                tokens_processed = 0
                step_start_time = time.time()

            accum_loss = 0.0

            # Checkpointing
            if global_step % train_config.checkpoint_every == 0:
                # DeepSpeed checkpoint (for resume)
                ds_ckpt_path = os.path.join(
                    train_config.checkpoint_dir, "ds_checkpoint"
                )
                tau = model_engine.module.model_core[0].polyglu.tau
                model_engine.save_checkpoint(
                    ds_ckpt_path,
                    client_state={"step": global_step, "tau": tau},
                )
                print(f"Saved DeepSpeed checkpoint at step {global_step}")

            if global_step % train_config.portable_checkpoint_every == 0:
                # Portable checkpoint (for eval/SFT)
                tau = model_engine.module.model_core[0].polyglu.tau
                portable_path = os.path.join(
                    train_config.checkpoint_dir, f"portable_step{global_step}.pt"
                )
                save_portable_checkpoint(
                    model_engine, optimizer, scheduler, global_step, tau, portable_path
                )
                print(f"Saved portable checkpoint at step {global_step}")

    # --- Final checkpoints ---
    if local_rank == 0:
        tau = model_engine.module.model_core[0].polyglu.tau

        # DeepSpeed
        ds_ckpt_path = os.path.join(train_config.checkpoint_dir, "ds_checkpoint")
        model_engine.save_checkpoint(
            ds_ckpt_path, client_state={"step": global_step, "tau": tau}
        )

        # Portable
        portable_path = os.path.join(train_config.checkpoint_dir, "portable_final.pt")
        save_portable_checkpoint(
            model_engine, optimizer, scheduler, global_step, tau, portable_path
        )

        print(f"Training complete. Final step: {global_step}")
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="PolychromaticLM pre-training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--local_rank", type=int, default=0, help="Set by DeepSpeed launcher")
    args = parser.parse_args()

    model_config = ModelConfig.from_yaml(args.config)
    train_config = TrainConfig.from_yaml(args.config)

    train(model_config, train_config, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
