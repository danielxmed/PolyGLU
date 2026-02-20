"""Tokenization pipeline: HF datasets → binary uint32 chunks.

Streams HF datasets, tokenizes with Qwen3 tokenizer, appends EOS between
documents, and writes binary chunks of ~100M tokens each (~400MB per chunk).

Supports resume by counting existing chunks and skipping.

Usage:
    # Single dataset
    python -m src.data.tokenize_dataset \
        --dataset nvidia/Nemotron-CC-Math-v1 \
        --config 4plus \
        --output data/tokenized/math \
        --max-tokens 7000000000

    # All datasets
    python -m src.data.tokenize_dataset --all --output-base data/tokenized
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


CHUNK_SIZE = 100_000_000  # ~100M tokens per chunk
EOS_TOKEN_ID = 151643  # Qwen3 <|endoftext|>

# Verified HF dataset configurations
DATASET_CONFIGS = {
    "math": {
        "dataset_id": "nvidia/Nemotron-CC-Math-v1",
        "config": "4plus",
        "text_field": "text",
        "max_tokens": 7_000_000_000,
    },
    "stem": {
        "dataset_id": "nvidia/Nemotron-CC-v2",
        "config": None,
        "text_field": "text",
        "max_tokens": 2_500_000_000,
    },
    "code": {
        "dataset_id": "nvidia/Nemotron-CC-Code-v1",
        "config": None,
        "text_field": "text",
        "max_tokens": 500_000_000,
    },
}


def get_tokenizer():
    """Load Qwen3 tokenizer (reused as-is, never retrained)."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    return tokenizer


def count_existing_chunks(output_dir: str) -> int:
    """Count existing .bin chunk files for resume support."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0
    return len(list(output_path.glob("chunk_*.bin")))


def tokenize_and_write(
    dataset_id: str,
    config: str | None,
    text_field: str,
    output_dir: str,
    max_tokens: int,
    token: str | None = None,
):
    """Stream a HF dataset, tokenize, and write binary uint32 chunks.

    Each document is tokenized and separated by an EOS token.
    Chunks are written as raw uint32 arrays (~400MB each at 100M tokens).

    Args:
        dataset_id: HuggingFace dataset ID.
        config: Dataset config name (e.g., "4plus") or None.
        text_field: Name of the text column in the dataset.
        output_dir: Directory to write chunk files.
        max_tokens: Maximum total tokens to process.
        token: HF token for gated dataset access.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing_chunks = count_existing_chunks(output_dir)
    tokens_to_skip = existing_chunks * CHUNK_SIZE

    tokenizer = get_tokenizer()

    print(f"Loading dataset: {dataset_id}" + (f" (config={config})" if config else ""))
    ds = load_dataset(
        dataset_id,
        name=config,
        split="train",
        streaming=True,
        token=token,
    )

    buffer = []
    total_tokens = 0
    chunk_idx = existing_chunks
    skipped_tokens = 0

    if tokens_to_skip > 0:
        print(f"Resuming: skipping ~{tokens_to_skip:,} tokens ({existing_chunks} existing chunks)")

    for example in ds:
        text = example.get(text_field, "")
        if not text or not text.strip():
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(EOS_TOKEN_ID)  # Document separator

        # Skip tokens covered by existing chunks
        if skipped_tokens < tokens_to_skip:
            skipped_tokens += len(token_ids)
            continue

        buffer.extend(token_ids)
        total_tokens += len(token_ids)

        # Write chunk when buffer is full
        while len(buffer) >= CHUNK_SIZE:
            chunk_data = np.array(buffer[:CHUNK_SIZE], dtype=np.uint32)
            chunk_path = output_path / f"chunk_{chunk_idx:05d}.bin"
            chunk_data.tofile(str(chunk_path))
            print(f"  Wrote {chunk_path.name} ({CHUNK_SIZE:,} tokens, total: {total_tokens:,})")

            buffer = buffer[CHUNK_SIZE:]
            chunk_idx += 1

        if total_tokens >= max_tokens:
            print(f"Reached max_tokens limit ({max_tokens:,})")
            break

    # Write remaining tokens as final partial chunk
    if buffer:
        chunk_data = np.array(buffer, dtype=np.uint32)
        chunk_path = output_path / f"chunk_{chunk_idx:05d}.bin"
        chunk_data.tofile(str(chunk_path))
        print(f"  Wrote {chunk_path.name} ({len(buffer):,} tokens, final partial chunk)")
        chunk_idx += 1

    # Write manifest
    manifest = {
        "dataset_id": dataset_id,
        "config": config,
        "total_tokens": total_tokens,
        "num_chunks": chunk_idx,
        "chunk_size": CHUNK_SIZE,
        "eos_token_id": EOS_TOKEN_ID,
    }
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done: {total_tokens:,} tokens in {chunk_idx} chunks → {output_dir}")
    return total_tokens


def tokenize_all(output_base: str, token: str | None = None):
    """Tokenize all 3 sources (math, STEM, code) in sequence."""
    for source_name, cfg in DATASET_CONFIGS.items():
        output_dir = os.path.join(output_base, source_name)
        print(f"\n{'='*60}")
        print(f"Tokenizing: {source_name} ({cfg['dataset_id']})")
        print(f"{'='*60}")
        tokenize_and_write(
            dataset_id=cfg["dataset_id"],
            config=cfg["config"],
            text_field=cfg["text_field"],
            output_dir=output_dir,
            max_tokens=cfg["max_tokens"],
            token=token,
        )


def main():
    parser = argparse.ArgumentParser(description="Tokenize HF datasets into binary chunks")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset ID")
    parser.add_argument("--config", type=str, default=None, help="Dataset config name")
    parser.add_argument("--text-field", type=str, default="text", help="Text column name")
    parser.add_argument("--output", type=str, help="Output directory for chunks")
    parser.add_argument("--max-tokens", type=int, default=10_000_000_000, help="Max tokens to process")
    parser.add_argument("--all", action="store_true", help="Tokenize all 3 sources")
    parser.add_argument("--output-base", type=str, default="data/tokenized", help="Base output dir for --all")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.all:
        tokenize_all(args.output_base, token=token)
    elif args.dataset and args.output:
        tokenize_and_write(
            dataset_id=args.dataset,
            config=args.config,
            text_field=args.text_field,
            output_dir=args.output,
            max_tokens=args.max_tokens,
            token=token,
        )
    else:
        parser.error("Either --all or both --dataset and --output are required")


if __name__ == "__main__":
    main()
