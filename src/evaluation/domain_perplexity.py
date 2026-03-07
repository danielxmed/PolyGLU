"""Compute per-domain perplexity on held-out training data.

Usage:
    python -m src.evaluation.domain_perplexity \
        --checkpoint checkpoints/portable_final.pt \
        --data-dir data/tokenized \
        --output results/base_eval/domain_perplexity.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.model.architecture import PolychromaticLM
from src.model.config import ModelConfig


DOMAINS = {
    "math": "math",
    "stem": "stem",
    "code": "code",
}


def load_model(checkpoint_path: str, device: str = "cuda") -> PolychromaticLM:
    """Load model for evaluation (no Flash Attention)."""
    config = ModelConfig()
    model = PolychromaticLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        eps=config.eps,
        head_dim=config.head_dim,
        seq_length=config.seq_length,
        n_activations=config.n_activations,
        n_q_heads=config.n_q_heads,
        n_kv_heads=config.n_kv_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    tau = checkpoint.get("tau", 0.1)
    for block in model.model_core:
        block.polyglu.tau = tau

    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    step = checkpoint.get("step", "?")
    print(f"Loaded model from {checkpoint_path} (step {step}, tau={tau:.3f})")
    return model


def load_held_out_chunk(data_dir: str, domain_subdir: str, min_tokens: int = 100_000) -> np.ndarray:
    """Memory-map a held-out chunk from a domain directory.

    Uses the last chunk that has at least min_tokens. Falls back to
    second-to-last if the last chunk is a small remainder.
    """
    from pathlib import Path
    chunk_dir = Path(data_dir) / domain_subdir
    chunk_paths = sorted(chunk_dir.glob("chunk_*.bin"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files in {chunk_dir}")

    # Try from the end, pick first chunk with enough tokens
    for chunk_path in reversed(chunk_paths):
        data = np.memmap(str(chunk_path), dtype=np.uint32, mode="r")
        if len(data) >= min_tokens:
            print(f"  {domain_subdir}: loaded {chunk_path.name} ({len(data):,} tokens)")
            return data
        print(f"  {domain_subdir}: skipped {chunk_path.name} (only {len(data):,} tokens)")

    # Fallback: use whatever is available
    data = np.memmap(str(chunk_paths[-1]), dtype=np.uint32, mode="r")
    print(f"  {domain_subdir}: loaded {chunk_paths[-1].name} ({len(data):,} tokens, small)")
    return data


def compute_perplexity(
    model: PolychromaticLM,
    tokens: np.ndarray,
    seq_length: int = 4096,
    num_sequences: int = 244,
    device: str = "cuda",
) -> dict:
    """Compute perplexity on a sample of sequences from a token array.

    Returns dict with loss, perplexity, bits_per_byte, tokens_evaluated.
    """
    total_tokens = len(tokens)
    # Sample non-overlapping sequences
    max_start = total_tokens - (seq_length + 1)
    if max_start <= 0:
        raise ValueError(f"Chunk too small: {total_tokens} tokens for seq_length={seq_length}")

    # Deterministic sampling: evenly spaced starts
    num_sequences = min(num_sequences, max_start // (seq_length + 1))
    stride = max_start // num_sequences
    starts = [i * stride for i in range(num_sequences)]

    total_loss = 0.0
    total_tokens_evaluated = 0

    for idx, start in enumerate(starts):
        seq = tokens[start:start + seq_length + 1].astype(np.int64)
        input_ids = torch.tensor(seq[:seq_length], dtype=torch.long, device=device).unsqueeze(0)
        targets = torch.tensor(seq[1:seq_length + 1], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_ids)  # (1, seq_length, vocab)
            # Compute cross-entropy without materializing full vocab logits in float32
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )

        total_loss += loss.item()
        total_tokens_evaluated += seq_length

        if (idx + 1) % 50 == 0:
            running_ppl = math.exp(total_loss / total_tokens_evaluated)
            print(f"    {idx+1}/{num_sequences} sequences, running ppl: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens_evaluated
    perplexity = math.exp(avg_loss)
    # bits per byte: loss_nats / ln(2) / bytes_per_token
    # Approximate bytes_per_token from Qwen3 tokenizer (~3.5 bytes/token typical)
    bits_per_byte = avg_loss / math.log(2)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "bits_per_token": avg_loss / math.log(2),
        "tokens_evaluated": total_tokens_evaluated,
        "num_sequences": num_sequences,
    }


def main():
    parser = argparse.ArgumentParser(description="Domain perplexity evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/tokenized")
    parser.add_argument("--num-sequences", type=int, default=244,
                        help="Sequences per domain (~1M tokens at 244*4096)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/base_eval/domain_perplexity.json")
    args = parser.parse_args()

    print("=" * 60)
    print("DOMAIN PERPLEXITY EVALUATION")
    print("=" * 60)

    model = load_model(args.checkpoint, args.device)

    results = {}
    print("\nLoading held-out chunks (last chunk per domain):")
    for domain_name, subdir in DOMAINS.items():
        print(f"\n--- {domain_name.upper()} ---")
        chunk_data = load_held_out_chunk(args.data_dir, subdir)

        t0 = time.time()
        domain_result = compute_perplexity(
            model, chunk_data,
            seq_length=4096,
            num_sequences=args.num_sequences,
            device=args.device,
        )
        elapsed = time.time() - t0
        domain_result["wall_time_sec"] = elapsed

        results[domain_name] = domain_result
        print(f"  Perplexity: {domain_result['perplexity']:.2f}")
        print(f"  Avg loss: {domain_result['avg_loss']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("DOMAIN PERPLEXITY SUMMARY")
    print("=" * 60)
    print(f"{'Domain':<10} {'Perplexity':>12} {'Avg Loss':>10} {'Tokens':>12}")
    print("-" * 50)
    for domain_name, r in results.items():
        print(f"{domain_name:<10} {r['perplexity']:>12.2f} {r['avg_loss']:>10.4f} {r['tokens_evaluated']:>12,}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
