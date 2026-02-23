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
import sys
import time

import deepspeed
import torch
import torch.nn as nn
import wandb

from src.model.config import ModelConfig, TrainConfig
from src.model.model import create_model, get_routing_entropy
from src.data.dataloader import PretrainingDataLoader

# Enable TF32 for any float32 operations (e.g., loss accumulation)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def chunked_cross_entropy(
    output_head: nn.Linear,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Memory-efficient cross-entropy by chunking the output head projection.

    With vocab=151K, full logits for micro_batch=16 = ~20GB. This function
    processes the output head + CE loss in chunks along the sequence dimension.
    With chunk_size=2048, only 2 chunks are needed (seq_len=4096), each ~10GB.

    Args:
        output_head: The nn.Linear(d_model, vocab_size) output projection.
        hidden: Hidden states [B, T, D] from the model (before output head).
        targets: Target token IDs [B, T].
        chunk_size: Number of sequence positions per chunk.

    Returns:
        Scalar loss (mean cross-entropy over all tokens).
    """
    B, T, D = hidden.shape
    total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)

    def _chunk_loss(chunk_hidden, chunk_targets):
        logits = output_head(chunk_hidden)
        return nn.functional.cross_entropy(logits, chunk_targets, reduction="sum")

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_h = hidden[:, start:end, :].reshape(-1, D)
        chunk_t = targets[:, start:end].reshape(-1)
        # Checkpoint each chunk so logits are recomputed (not stored) during backward
        chunk_loss = torch.utils.checkpoint.checkpoint(
            _chunk_loss, chunk_h, chunk_t, use_reentrant=False
        )
        total_loss = total_loss + chunk_loss

    return total_loss / (B * T)


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
    print(f"Creating model (vocab={model_config.vocab_size}, layers={model_config.n_layers})...", flush=True)
    model = create_model(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}", flush=True)

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
    print("Creating dataloader...", flush=True)
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
    print(f"Starting training from step {start_step} to {train_config.total_steps}", flush=True)
    model_engine.train()

    global_step = start_step
    accum_loss = 0.0
    micro_step = 0
    tokens_processed = 0
    step_start_time = time.time()
    micro_start_time = time.time()

    for batch in dataloader:
        if global_step >= train_config.total_steps:
            break

        # Update mix ratios based on current step
        dataloader.set_step(global_step)

        t_micro = time.time()

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        cu_seqlens = batch["cu_seqlens"].to(device)
        max_seqlen = batch["max_seqlen"]

        # Debug: check memory and training mode on first micro-step
        if micro_step == 0:
            print(f"  [debug] training mode: {model_engine.module.training}", flush=True)
            print(f"  [debug] mem before forward: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        # Forward pass — get hidden states (before output head) for chunked CE
        hidden = model_engine(
            input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_hidden=True
        )

        if micro_step == 0:
            print(f"  [debug] hidden shape: {hidden.shape}, dtype: {hidden.dtype}", flush=True)
            print(f"  [debug] mem after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        # Chunked cross-entropy — avoids materializing full [B*T, vocab] logits
        # (saves ~15-20GB GPU memory vs naive approach with vocab=151K)
        loss = chunked_cross_entropy(
            model_engine.module.output_head, hidden, targets
        )

        if micro_step == 0:
            print(f"  [debug] loss: {loss.item():.4f}", flush=True)
            print(f"  [debug] mem after loss: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        # Scale loss for gradient accumulation (ds_config gas=1, we handle manually)
        scaled_loss = loss / train_config.grad_accum_steps

        # Backward (gradients accumulate in .grad tensors across micro-steps)
        model_engine.backward(scaled_loss)

        accum_loss += loss.item() / train_config.grad_accum_steps
        micro_step += 1
        tokens_processed += input_ids.numel()

        # Diagnostic: log first few micro-steps to measure speed
        if micro_step <= 5 or (micro_step <= 64 and micro_step % 32 == 0):
            dt = time.time() - t_micro
            print(f"  [micro {micro_step}] loss={loss.item():.4f} dt={dt:.3f}s", flush=True)

        # Check if we just completed a gradient accumulation cycle
        if micro_step % train_config.grad_accum_steps == 0:
            # Optimizer step (ds_config gas=1, so step() always does the full update)
            model_engine.step()

            # Tau annealing — matching frozen Trainer line 337
            model_engine.module.update_tau(global_step, train_config.total_steps)

            global_step += 1

            # Brief per-step log (every step for first 10, then every 10)
            if local_rank == 0 and (global_step <= 10 or global_step % 10 == 0):
                elapsed_micro = time.time() - micro_start_time
                micro_tps = tokens_processed / elapsed_micro if elapsed_micro > 0 else 0
                print(
                    f"  [step {global_step}] loss={accum_loss:.4f} "
                    f"tok/s={micro_tps:.0f} "
                    f"elapsed={elapsed_micro:.1f}s",
                    flush=True,
                )

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
                    f"entropy: {entropy_dict['routing_entropy/mean']:.3f}",
                    flush=True,
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
