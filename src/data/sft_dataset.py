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
from datasets import load_dataset
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
        max_examples: int = 0,
        token: str | None = None,
    ):
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
        )
        self.pad_token_id = self.tokenizer.pad_token_id or 0

        # Special token IDs for ChatML formatting
        self.im_start_text = "<|im_start|>"
        self.im_end_text = "<|im_end|>"
        self.eos_text = "<|endoftext|>"

        splits = splits or ["high_part00", "high_part01", "high_part02"]
        hf_token = token or os.environ.get("HF_TOKEN")

        # Load via streaming to avoid 47GB arrow cache on disk
        # Collect conversations into memory (~2-5GB RAM for 700K examples)
        conversations = []
        for split in splits:
            print(f"Streaming {split}...")
            ds_stream = load_dataset(dataset_id, split=split, streaming=True, token=hf_token)
            count = 0
            skipped = 0
            for example in ds_stream:
                messages = example.get("conversations", example.get("messages", []))
                # Filter tool-using examples inline
                has_tool = any(m["role"] not in ("user", "assistant", "system") for m in messages)
                if has_tool:
                    skipped += 1
                    continue
                conversations.append(messages)
                count += 1
                if max_examples > 0 and len(conversations) >= max_examples:
                    break
                if count % 100_000 == 0:
                    print(f"  ...{count} examples loaded", flush=True)
            print(f"  {split}: {count} kept, {skipped} tool-examples skipped")
            if max_examples > 0 and len(conversations) >= max_examples:
                break

        print(f"Total: {len(conversations)} examples in memory")
        self.conversations = conversations

    def _format_conversation(self, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
        """Format a conversation and track assistant token spans.

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            (formatted_text, assistant_spans) where assistant_spans is a list
            of (start_char, end_char) tuples marking assistant content.
        """
        text_parts = []
        assistant_spans = []
        current_pos = 0

        for msg in messages:
            role = msg["role"]
            # Skip non-user/assistant/system roles (safety net)
            if role not in ("user", "assistant", "system"):
                continue
            # Use content field only (not reasoning_content)
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
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized example with loss mask (variable length, no padding).

        Returns:
            {
                "input_ids": (seq_len,) int64 tensor,
                "targets": (seq_len,) int64 tensor,
                "loss_mask": (seq_len,) float32 tensor (1.0 for assistant, 0.0 otherwise),
            }
        """
        messages = self.conversations[idx]
        formatted_text, assistant_spans = self._format_conversation(messages)

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
            token_ids = token_ids + [self.pad_token_id] * (2 - len(token_ids))
            loss_mask = loss_mask + [0.0] * (2 - len(loss_mask))

        # Truncate to max_seq_length + 1 (input/target shift needs +1)
        if len(token_ids) > self.max_seq_length + 1:
            token_ids = token_ids[:self.max_seq_length + 1]
            loss_mask = loss_mask[:self.max_seq_length + 1]

        # No padding here — collate_fn handles dynamic padding
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_mask[1:], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "targets": targets,
            "loss_mask": loss_mask_tensor,
        }


def sft_collate_fn(batch: list[dict]) -> dict:
    """Dynamic padding collate — pads to max length in batch, not 4096.

    Math conversations vary widely in length. Dynamic padding avoids
    wasting compute on padding tokens (which are masked out anyway).
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    targets_list = []
    loss_mask_list = []

    # Use pad_token_id=0 (matching tokenizer default)
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            input_ids_list.append(torch.cat([
                item["input_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            targets_list.append(torch.cat([
                item["targets"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            loss_mask_list.append(torch.cat([
                item["loss_mask"],
                torch.zeros(pad_len, dtype=torch.float32),
            ]))
        else:
            input_ids_list.append(item["input_ids"])
            targets_list.append(item["targets"])
            loss_mask_list.append(item["loss_mask"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "targets": torch.stack(targets_list),
        "loss_mask": torch.stack(loss_mask_list),
    }
