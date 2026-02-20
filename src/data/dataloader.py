"""Streaming dataloader with document masking and data mix ratios.

Core components:
- ChunkReader: Memory-maps binary chunks, yields continuous token streams
- MixingScheduler: Returns current (math, stem, code) ratios based on step
- PretrainingDataLoader: Produces micro-batches with cu_seqlens for Flash Attention

Output format per batch:
    {
        "input_ids": (bs, seq_length),        # Token IDs
        "targets": (bs, seq_length),          # Shifted targets
        "cu_seqlens": (num_docs + 1,),        # Cumulative doc lengths for flash_attn_varlen_func
        "max_seqlen": int,                    # Max document length in batch
    }
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


EOS_TOKEN_ID = 151643


class ChunkReader:
    """Memory-maps binary uint32 chunks and yields continuous token streams.

    Supports seek for training resume. Chunks are read sequentially with
    optional shuffling of chunk order.
    """

    def __init__(self, data_dir: str, shuffle: bool = True, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.chunk_paths = sorted(self.data_dir.glob("chunk_*.bin"))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")

        self.shuffle = shuffle
        self.rng = random.Random(seed)

        # Load manifest for metadata
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

        self._reset()

    def _reset(self):
        """Reset to start of data, optionally reshuffling chunk order."""
        self._chunk_order = list(range(len(self.chunk_paths)))
        if self.shuffle:
            self.rng.shuffle(self._chunk_order)
        self._chunk_idx = 0
        self._pos_in_chunk = 0
        self._current_chunk = None

    def _load_chunk(self, idx: int) -> np.ndarray:
        """Memory-map a chunk file as uint32 array."""
        return np.memmap(str(self.chunk_paths[idx]), dtype=np.uint32, mode="r")

    def read_sequence(self, length: int) -> Optional[np.ndarray]:
        """Read `length` tokens from the stream, crossing chunk boundaries.

        Returns None if data is exhausted (one epoch done).
        """
        tokens = []
        remaining = length

        while remaining > 0:
            if self._chunk_idx >= len(self._chunk_order):
                # Exhausted all chunks — signal end of epoch
                if tokens:
                    # Pad with EOS if we have partial data
                    tokens.extend([EOS_TOKEN_ID] * remaining)
                    return np.array(tokens, dtype=np.uint32)
                return None

            chunk_file_idx = self._chunk_order[self._chunk_idx]
            if self._current_chunk is None:
                self._current_chunk = self._load_chunk(chunk_file_idx)

            available = len(self._current_chunk) - self._pos_in_chunk
            take = min(remaining, available)

            tokens.extend(self._current_chunk[self._pos_in_chunk:self._pos_in_chunk + take].tolist())
            self._pos_in_chunk += take
            remaining -= take

            if self._pos_in_chunk >= len(self._current_chunk):
                self._chunk_idx += 1
                self._pos_in_chunk = 0
                self._current_chunk = None

        return np.array(tokens, dtype=np.uint32)

    def seek_tokens(self, n_tokens: int):
        """Skip ahead n_tokens for training resume."""
        chunk_size = self.manifest.get("chunk_size", 100_000_000)
        chunks_to_skip = n_tokens // chunk_size
        remainder = n_tokens % chunk_size

        self._chunk_idx = min(chunks_to_skip, len(self._chunk_order))
        self._pos_in_chunk = 0
        self._current_chunk = None

        if remainder > 0 and self._chunk_idx < len(self._chunk_order):
            chunk_file_idx = self._chunk_order[self._chunk_idx]
            self._current_chunk = self._load_chunk(chunk_file_idx)
            self._pos_in_chunk = min(remainder, len(self._current_chunk))


class MixingScheduler:
    """Returns current (math, stem, code) sampling probabilities based on step.

    Base proportions: 70/25/5
    After anneal_start_frac of training: linearly anneal to 85/10/5
    """

    def __init__(
        self,
        total_steps: int,
        base_math: float = 0.70,
        base_stem: float = 0.25,
        base_code: float = 0.05,
        anneal_start_frac: float = 0.80,
        anneal_math: float = 0.85,
        anneal_stem: float = 0.10,
        anneal_code: float = 0.05,
    ):
        self.total_steps = total_steps
        self.base = (base_math, base_stem, base_code)
        self.anneal_start_step = int(total_steps * anneal_start_frac)
        self.anneal_target = (anneal_math, anneal_stem, anneal_code)

    def get_ratios(self, step: int) -> tuple[float, float, float]:
        """Return (math, stem, code) sampling probabilities for given step."""
        if step < self.anneal_start_step:
            return self.base

        # Linear interpolation from base to anneal_target
        anneal_steps = self.total_steps - self.anneal_start_step
        if anneal_steps <= 0:
            return self.anneal_target

        progress = min((step - self.anneal_start_step) / anneal_steps, 1.0)
        ratios = tuple(
            b + (t - b) * progress
            for b, t in zip(self.base, self.anneal_target)
        )
        return ratios


def _extract_cu_seqlens(token_ids: np.ndarray, seq_length: int) -> tuple[torch.Tensor, int]:
    """Extract cumulative sequence lengths from EOS-delimited token sequence.

    Documents within a sequence are delimited by EOS tokens. cu_seqlens marks
    document boundaries for flash_attn_varlen_func to prevent cross-document attention.

    Args:
        token_ids: Array of token IDs for one sequence (length seq_length).
        seq_length: Sequence length.

    Returns:
        (cu_seqlens, max_seqlen): cu_seqlens is int32 tensor starting at 0,
        ending at seq_length. max_seqlen is the longest document in the sequence.
    """
    eos_positions = np.where(token_ids == EOS_TOKEN_ID)[0]

    boundaries = [0]
    for pos in eos_positions:
        # EOS at position pos means document ends after pos (inclusive)
        next_start = pos + 1
        if next_start < seq_length:
            boundaries.append(next_start)
    boundaries.append(seq_length)

    # Deduplicate and sort (handles consecutive EOS tokens)
    boundaries = sorted(set(boundaries))

    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)

    # Max document length
    doc_lengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
    max_seqlen = max(doc_lengths) if doc_lengths else seq_length

    return cu_seqlens, max_seqlen


class PretrainingDataLoader:
    """Streaming dataloader with document masking and data mix enforcement.

    For each micro-batch slot, samples a source via MixingScheduler,
    reads seq_length+1 tokens from that source's ChunkReader,
    and extracts cu_seqlens from EOS positions within each sequence.
    """

    def __init__(
        self,
        data_dir: str,
        math_subdir: str,
        stem_subdir: str,
        code_subdir: str,
        seq_length: int,
        micro_batch_size: int,
        total_steps: int,
        grad_accum_steps: int,
        mix_math: float = 0.70,
        mix_stem: float = 0.25,
        mix_code: float = 0.05,
        anneal_start_frac: float = 0.80,
        anneal_math: float = 0.85,
        anneal_stem: float = 0.10,
        anneal_code: float = 0.05,
        seed: int = 42,
        start_step: int = 0,
    ):
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps

        # Source readers
        self.readers = {
            "math": ChunkReader(os.path.join(data_dir, math_subdir), seed=seed),
            "stem": ChunkReader(os.path.join(data_dir, stem_subdir), seed=seed + 1),
            "code": ChunkReader(os.path.join(data_dir, code_subdir), seed=seed + 2),
        }
        self.source_names = ["math", "stem", "code"]

        self.scheduler = MixingScheduler(
            total_steps=total_steps,
            base_math=mix_math,
            base_stem=mix_stem,
            base_code=mix_code,
            anneal_start_frac=anneal_start_frac,
            anneal_math=anneal_math,
            anneal_stem=anneal_stem,
            anneal_code=anneal_code,
        )

        self.rng = random.Random(seed)
        self._current_step = start_step

    def set_step(self, step: int):
        """Update current step for mix ratio scheduling."""
        self._current_step = step

    def _sample_source(self) -> str:
        """Sample a data source based on current mixing ratios."""
        ratios = self.scheduler.get_ratios(self._current_step)
        return self.rng.choices(self.source_names, weights=ratios, k=1)[0]

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Yield one micro-batch with document masking info.

        Returns dict with:
            input_ids: (micro_batch_size, seq_length) int64 tensor
            targets: (micro_batch_size, seq_length) int64 tensor
            cu_seqlens: (total_boundaries,) int32 tensor (batched cu_seqlens)
            max_seqlen: int
        """
        input_ids_list = []
        targets_list = []
        all_cu_seqlens = []
        max_seqlen = 0
        offset = 0  # Running offset for batched cu_seqlens

        for _ in range(self.micro_batch_size):
            source = self._sample_source()
            reader = self.readers[source]

            # Read seq_length + 1 tokens (input + shifted target)
            tokens = reader.read_sequence(self.seq_length + 1)
            if tokens is None:
                raise StopIteration("Data exhausted")

            input_tokens = tokens[:self.seq_length]
            target_tokens = tokens[1:self.seq_length + 1]

            # Extract document boundaries for this sequence
            cu_seqlens_seq, max_seqlen_seq = _extract_cu_seqlens(input_tokens, self.seq_length)
            max_seqlen = max(max_seqlen, max_seqlen_seq)

            # Offset cu_seqlens for batched varlen attention
            all_cu_seqlens.append(cu_seqlens_seq + offset)
            offset += self.seq_length

            input_ids_list.append(torch.from_numpy(input_tokens.astype(np.int64)))
            targets_list.append(torch.from_numpy(target_tokens.astype(np.int64)))

        # Merge cu_seqlens: take first element from first sequence,
        # then remaining elements from all sequences
        merged_cu_seqlens = [all_cu_seqlens[0]]
        for cs in all_cu_seqlens[1:]:
            merged_cu_seqlens.append(cs[1:])  # Skip duplicate 0-offsets
        cu_seqlens = torch.cat(merged_cu_seqlens)

        return {
            "input_ids": torch.stack(input_ids_list),
            "targets": torch.stack(targets_list),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
