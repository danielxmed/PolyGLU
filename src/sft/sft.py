"""Supervised fine-tuning script for PolychromaticLM.

Loads a pre-trained checkpoint, freezes tau at 0.1, and fine-tunes on
Nemotron-Math-v2 with loss masking on assistant tokens only.

Fixes vs skeleton:
- Alpha params excluded from weight decay (matching pre-training bug fix)
- DeepSpeed gas=1 with manual micro-step tracking (no double loss scaling)
- Chunked masked cross-entropy (avoids materializing full logit tensor)
- TF32 enabled, micro_batch=16 for GPU utilization

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
from src.data.sft_dataset import SFTDataset, sft_collate_fn

# Enable TF32 for any float32 operations (matching pre-training)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_optimizer(model: nn.Module, config: SFTConfig) -> torch.optim.AdamW:
    """AdamW with param group separation (matching pre-training pattern).

    - 2D+ weight matrices: weight_decay = 0.1
    - 1D params (biases, norms, beta): weight_decay = 0.0
    - Routing params (alpha): weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.ndim == 1:
            no_decay_params.append(param)
        elif 'alpha' in name:
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


def chunked_masked_cross_entropy(
    output_head: nn.Linear,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Memory-efficient masked cross-entropy by chunking the output head projection.

    Adapted from pre-training's chunked_cross_entropy for SFT loss masking.
    Avoids materializing the full [B, T, 151669] logit tensor.

    Args:
        output_head: The nn.Linear(d_model, vocab_size) output projection.
        hidden: Hidden states [B, T, D] from the model (before output head).
        targets: Target token IDs [B, T].
        loss_mask: Mask [B, T] with 1.0 for assistant tokens, 0.0 otherwise.
        chunk_size: Number of sequence positions per chunk.

    Returns:
        Scalar loss (mean cross-entropy over assistant tokens).
    """
    B, T, D = hidden.shape
    total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
    total_tokens = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)

    def _chunk_loss(chunk_hidden, chunk_targets, chunk_mask):
        logits = output_head(chunk_hidden)
        per_token = nn.functional.cross_entropy(
            logits, chunk_targets, reduction="none"
        )
        masked = per_token * chunk_mask
        return masked.sum(), chunk_mask.sum()

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_h = hidden[:, start:end, :].reshape(-1, D)
        chunk_t = targets[:, start:end].reshape(-1)
        chunk_m = loss_mask[:, start:end].reshape(-1)

        chunk_loss, chunk_count = torch.utils.checkpoint.checkpoint(
            _chunk_loss, chunk_h, chunk_t, chunk_m, use_reentrant=False
        )
        total_loss = total_loss + chunk_loss
        total_tokens = total_tokens + chunk_count

    if total_tokens > 0:
        return total_loss / total_tokens
    return total_loss


def train_sft(model_config: ModelConfig, sft_config: SFTConfig, local_rank: int = 0):
    """Main SFT training loop."""
    device = torch.device(f"cuda:{local_rank}")

    # --- Model from pre-trained checkpoint ---
    print(f"Loading pre-trained checkpoint: {sft_config.pretrained_checkpoint}")
    model = create_model(model_config)

    checkpoint = torch.load(sft_config.pretrained_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    pretrain_step = checkpoint.get('step', '?')
    pretrain_tau = checkpoint.get('tau', '?')
    print(f"Loaded model from step {pretrain_step} (tau={pretrain_tau})")

    # Freeze tau at 0.1 (end of annealing)
    for block in model.model_core:
        block.polyglu.tau = sft_config.tau

    total_params = sum(p.numel() for p in model.parameters())
    routing_params = sum(p.numel() for n, p in model.named_parameters() if 'alpha' in n or 'beta' in n or 'gate_net' in n)
    print(f"Total parameters: {total_params:,} (routing: {routing_params:,}, {routing_params/total_params*100:.2f}%)")

    # --- Data ---
    print("Loading SFT dataset...")
    dataset = SFTDataset(
        dataset_id=sft_config.dataset_id,
        splits=sft_config.dataset_splits,
        max_seq_length=sft_config.max_seq_length,
        max_examples=sft_config.max_examples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=sft_config.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=sft_collate_fn,
    )

    # Compute total steps (manual gas tracking)
    micro_batches_per_epoch = len(dataloader)
    steps_per_epoch = micro_batches_per_epoch // sft_config.grad_accum_steps
    total_steps = steps_per_epoch * sft_config.epochs
    if sft_config.max_steps > 0:
        total_steps = min(total_steps, sft_config.max_steps)

    print(f"\n{'='*60}")
    print(f"SFT Training Summary")
    print(f"{'='*60}")
    print(f"  Dataset size:       {len(dataset):,} examples")
    print(f"  Micro batch size:   {sft_config.micro_batch_size}")
    print(f"  Grad accum steps:   {sft_config.grad_accum_steps}")
    print(f"  Effective batch:    {sft_config.micro_batch_size * sft_config.grad_accum_steps}")
    print(f"  Micro batches/epoch:{micro_batches_per_epoch:,}")
    print(f"  Steps/epoch:        {steps_per_epoch:,}")
    print(f"  Epochs:             {sft_config.epochs}")
    print(f"  Total steps:        {total_steps:,}")
    print(f"  Warmup steps:       {sft_config.warmup_steps}")
    print(f"  Peak LR:            {sft_config.peak_lr}")
    print(f"  Tau (frozen):       {sft_config.tau}")
    print(f"{'='*60}\n")

    # --- Optimizer & Scheduler ---
    optimizer = create_optimizer(model, sft_config)
    scheduler = create_scheduler(optimizer, sft_config.warmup_steps, total_steps)

    # --- DeepSpeed (gas=1, we handle accumulation manually) ---
    with open(sft_config.ds_config) as f:
        ds_config = json.load(f)

    # CRITICAL: gas=1 in DeepSpeed, manual micro-step tracking
    # This avoids the double-scaling bug (DeepSpeed auto-scales when gas>1)
    ds_config["gradient_accumulation_steps"] = 1
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
    micro_step = 0
    accum_loss = 0.0
    accum_assistant_tokens = 0
    tokens_processed = 0
    step_start_time = time.time()

    for epoch in range(sft_config.epochs):
        for batch in dataloader:
            if global_step >= total_steps:
                break

            t_micro = time.time()

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # Debug: first micro-step diagnostics
            if micro_step == 0:
                print(f"  [debug] training mode: {model_engine.module.training}", flush=True)
                print(f"  [debug] input shape: {input_ids.shape}, dtype: {input_ids.dtype}", flush=True)
                print(f"  [debug] mem before forward: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

            # Forward — get hidden states for chunked CE
            hidden = model_engine(input_ids, return_hidden=True)

            if micro_step == 0:
                print(f"  [debug] hidden shape: {hidden.shape}, dtype: {hidden.dtype}", flush=True)
                print(f"  [debug] mem after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

            # Chunked masked cross-entropy
            loss = chunked_masked_cross_entropy(
                model_engine.module.output_head, hidden, targets, loss_mask
            )

            if micro_step == 0:
                print(f"  [debug] loss: {loss.item():.4f}", flush=True)
                print(f"  [debug] mem after loss: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

            # Scale loss for manual gradient accumulation (ds gas=1)
            scaled_loss = loss / sft_config.grad_accum_steps

            # Backward
            model_engine.backward(scaled_loss)

            accum_loss += loss.item() / sft_config.grad_accum_steps
            n_assistant = int(loss_mask.sum().item())
            accum_assistant_tokens += n_assistant
            tokens_processed += input_ids.numel()
            micro_step += 1

            # Diagnostic: log first few micro-steps
            if micro_step <= 5 or (micro_step <= 64 and micro_step % 32 == 0):
                dt = time.time() - t_micro
                seq_len = input_ids.shape[1]
                print(
                    f"  [micro {micro_step}] loss={loss.item():.4f} "
                    f"assistant_tokens={n_assistant} seq_len={seq_len} dt={dt:.3f}s",
                    flush=True,
                )

            # Optimizer step at accumulation boundary
            if micro_step % sft_config.grad_accum_steps == 0:
                model_engine.step()
                global_step += 1

                # Brief per-step log (every step for first 10, then every 10)
                if local_rank == 0 and (global_step <= 10 or global_step % 10 == 0):
                    elapsed = time.time() - step_start_time
                    tps = tokens_processed / elapsed if elapsed > 0 else 0
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    print(
                        f"  [step {global_step}] loss={accum_loss:.4f} "
                        f"tok/s={tps:.0f} asst_tok={accum_assistant_tokens} "
                        f"mem={mem_gb:.1f}GB",
                        flush=True,
                    )

                # Detailed logging
                if global_step % sft_config.log_every == 0 and local_rank == 0:
                    elapsed = time.time() - step_start_time
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    lr = scheduler.get_last_lr()[0]
                    mem_gb = torch.cuda.memory_allocated() / 1e9

                    # Routing entropy
                    entropy_dict = get_routing_entropy(model_engine.module)

                    log_dict = {
                        "sft/loss": accum_loss,
                        "sft/lr": lr,
                        "sft/tokens_per_sec": tokens_per_sec,
                        "sft/assistant_tokens": accum_assistant_tokens,
                        "sft/gpu_memory_gb": mem_gb,
                        "sft/step": global_step,
                        "sft/epoch": epoch,
                        **entropy_dict,
                    }
                    wandb.log(log_dict, step=global_step)

                    print(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {accum_loss:.4f} | LR: {lr:.6f} | "
                        f"tok/s: {tokens_per_sec:.0f} | mem: {mem_gb:.1f}GB | "
                        f"entropy: {entropy_dict['routing_entropy/mean']:.3f}",
                        flush=True,
                    )

                    tokens_processed = 0
                    step_start_time = time.time()

                accum_loss = 0.0
                accum_assistant_tokens = 0

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

        if global_step >= total_steps:
            break

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
        print(f"\nSFT complete. Final step: {global_step}. Checkpoint: {final_path}")
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
