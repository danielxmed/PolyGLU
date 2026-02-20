"""SFT dataset for supervised fine-tuning on Nemotron-Math-v2.

Formats conversations with ChatML-style tags and builds loss masks
(1 for assistant tokens, 0 for user/system tokens).

Format:
    <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>\n<|endoftext|>
"""

import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


class SFTDataset(Dataset):
    """SFT dataset with loss masking on assistant tokens only.

    Loads nvidia/Nemotron-Math-v2, formats conversations, tokenizes,
    and builds loss masks.
    """

    def __init__(
        self,
        dataset_id: str = "nvidia/Nemotron-Math-v2",
        splits: list[str] | None = None,
        max_seq_length: int = 4096,
        token: str | None = None,
    ):
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
        )

        # Special token IDs for ChatML formatting
        self.im_start_text = "<|im_start|>"
        self.im_end_text = "<|im_end|>"
        self.eos_text = "<|endoftext|>"

        splits = splits or ["high_part00", "high_part01", "high_part02"]
        hf_token = token or os.environ.get("HF_TOKEN")

        # Load and concatenate splits
        ds_parts = []
        for split in splits:
            ds_part = load_dataset(dataset_id, split=split, token=hf_token)
            ds_parts.append(ds_part)
        self.dataset = concatenate_datasets(ds_parts)

        print(f"Loaded {len(self.dataset)} examples from {dataset_id}")

    def _format_conversation(self, example: dict) -> tuple[str, list[tuple[int, int]]]:
        """Format a conversation and track assistant token spans.

        Args:
            example: Dataset example with 'conversations' field containing
                     list of {"role": str, "content": str} dicts.

        Returns:
            (formatted_text, assistant_spans) where assistant_spans is a list
            of (start_char, end_char) tuples marking assistant content.
        """
        messages = example.get("conversations", example.get("messages", []))
        text_parts = []
        assistant_spans = []
        current_pos = 0

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            header = f"{self.im_start_text}{role}\n"
            body = f"{content}{self.im_end_text}\n"

            text_parts.append(header)
            current_pos += len(header)

            if role == "assistant":
                start = current_pos
                text_parts.append(body)
                current_pos += len(body)
                assistant_spans.append((start, current_pos))
            else:
                text_parts.append(body)
                current_pos += len(body)

        text_parts.append(self.eos_text)
        return "".join(text_parts), assistant_spans

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized example with loss mask.

        Returns:
            {
                "input_ids": (seq_length,) int64 tensor,
                "targets": (seq_length,) int64 tensor,
                "loss_mask": (seq_length,) float32 tensor (1.0 for assistant, 0.0 otherwise),
            }
        """
        example = self.dataset[idx]
        formatted_text, assistant_spans = self._format_conversation(example)

        # Tokenize the full text
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_length + 1,
            truncation=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        # Build loss mask using character offsets
        loss_mask = [0.0] * len(token_ids)
        for i, (char_start, char_end) in enumerate(offsets):
            if char_start is None or char_end is None:
                continue
            for asp_start, asp_end in assistant_spans:
                if char_start >= asp_start and char_end <= asp_end:
                    loss_mask[i] = 1.0
                    break

        # Ensure we have at least 2 tokens for input/target split
        if len(token_ids) < 2:
            # Pad to minimum length
            token_ids = token_ids + [self.tokenizer.pad_token_id or 0] * (2 - len(token_ids))
            loss_mask = loss_mask + [0.0] * (2 - len(loss_mask))

        # Pad or truncate to max_seq_length + 1
        if len(token_ids) > self.max_seq_length + 1:
            token_ids = token_ids[:self.max_seq_length + 1]
            loss_mask = loss_mask[:self.max_seq_length + 1]
        else:
            pad_len = self.max_seq_length + 1 - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            loss_mask = loss_mask + [0.0] * pad_len

        input_ids = torch.tensor(token_ids[:self.max_seq_length], dtype=torch.long)
        targets = torch.tensor(token_ids[1:self.max_seq_length + 1], dtype=torch.long)
        # Loss mask aligned with targets (shifted by 1)
        loss_mask_tensor = torch.tensor(loss_mask[1:self.max_seq_length + 1], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "targets": targets,
            "loss_mask": loss_mask_tensor,
        }
