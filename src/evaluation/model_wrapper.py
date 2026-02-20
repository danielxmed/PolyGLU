"""lm-evaluation-harness LM interface for PolychromaticLM.

Implements the lm_eval.api.model.LM interface for benchmark evaluation.
No KV cache — simple but sufficient for one-time evaluation runs.

Supports:
- loglikelihood(): Forward pass, extract log-probs for continuation (MMLU)
- generate_until(): Autoregressive generation with stop strings (GSM8K, MATH-500)
- loglikelihood_rolling(): Rolling log-likelihood for perplexity
"""

import torch
import torch.nn.functional as F
from typing import Optional
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from src.model.architecture import PolychromaticLM
from src.model.config import ModelConfig
from transformers import AutoTokenizer


@register_model("polychromatic")
class PolychromaticLMWrapper(LM):
    """lm-eval-harness wrapper for PolychromaticLM."""

    def __init__(
        self,
        checkpoint_path: str,
        model_config: Optional[ModelConfig] = None,
        batch_size: int = 1,
        max_gen_length: int = 512,
        device: str = "cuda",
    ):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_gen_length = max_gen_length

        config = model_config or ModelConfig()
        self.config = config

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
        )
        self.eos_token_id = self.tokenizer.eos_token_id or 151643

        # Load model (no Flash Attention for eval simplicity)
        config_no_flash = ModelConfig(**{
            k: (False if k == "use_flash_attn" else v)
            for k, v in config.__dict__.items()
        })

        self.model = PolychromaticLM(
            vocab_size=config_no_flash.vocab_size,
            d_model=config_no_flash.d_model,
            eps=config_no_flash.eps,
            head_dim=config_no_flash.head_dim,
            seq_length=config_no_flash.seq_length,
            n_activations=config_no_flash.n_activations,
            n_q_heads=config_no_flash.n_q_heads,
            n_kv_heads=config_no_flash.n_kv_heads,
            d_ff=config_no_flash.d_ff,
            n_layers=config_no_flash.n_layers,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])

        # Set tau to stored value
        tau = checkpoint.get("tau", 0.1)
        for block in self.model.model_core:
            block.polyglu.tau = tau

        self.model = self.model.to(dtype=torch.bfloat16, device=self._device)
        self.model.eval()

        print(f"Loaded PolychromaticLM from {checkpoint_path} (step {checkpoint.get('step', '?')}, tau={tau:.3f})")

    @property
    def eot_token_id(self):
        return self.eos_token_id

    @property
    def max_length(self):
        return self.config.seq_length

    @property
    def max_gen_toks(self):
        return self._max_gen_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run model forward pass, return logits."""
        with torch.no_grad():
            return self.model(input_ids)

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Autoregressive generation without KV cache.

        Simple but correct for evaluation purposes.
        """
        generated = context.clone()
        stop_token_ids = set()
        if stop:
            for s in stop:
                ids = self.tok_encode(s, add_special_tokens=False)
                if ids:
                    stop_token_ids.add(ids[0])

        for _ in range(max_length):
            if generated.shape[1] >= self.config.seq_length:
                break

            with torch.no_grad():
                logits = self.model(generated)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            token_id = next_token.item()
            if token_id == self.eos_token_id or token_id in stop_token_ids:
                break

        return generated

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood for each (context, continuation) pair."""
        results = []
        for request in requests:
            context, continuation = request.args
            context_ids = self.tok_encode(context)
            continuation_ids = self.tok_encode(continuation)
            full_ids = context_ids + continuation_ids

            # Truncate to max length
            if len(full_ids) > self.config.seq_length:
                full_ids = full_ids[-self.config.seq_length:]
                # Recompute where continuation starts after truncation
                cont_start = max(0, len(full_ids) - len(continuation_ids))
            else:
                cont_start = len(context_ids)

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self._device)
            logits = self._model_call(input_tensor)  # (1, seq_len, vocab)
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Extract log-probs for continuation tokens
            total_logprob = 0.0
            is_greedy = True
            for i in range(cont_start, len(full_ids)):
                if i == 0:
                    continue
                token_id = full_ids[i]
                token_logprob = log_probs[0, i - 1, token_id].item()
                total_logprob += token_logprob

                # Check if this was the greedy choice
                if torch.argmax(log_probs[0, i - 1]).item() != token_id:
                    is_greedy = False

            results.append((total_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        """Compute rolling log-likelihood for perplexity."""
        results = []
        for request in requests:
            (text,) = request.args
            token_ids = self.tok_encode(text)

            total_logprob = 0.0
            # Process in chunks of max_length
            for start in range(0, len(token_ids), self.config.seq_length):
                chunk = token_ids[start:start + self.config.seq_length]
                if len(chunk) < 2:
                    continue

                input_tensor = torch.tensor([chunk], dtype=torch.long, device=self._device)
                logits = self._model_call(input_tensor)
                log_probs = F.log_softmax(logits.float(), dim=-1)

                for i in range(1, len(chunk)):
                    total_logprob += log_probs[0, i - 1, chunk[i]].item()

            results.append((total_logprob,))

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until stop condition is met."""
        results = []
        for request in requests:
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self._max_gen_length)

            context_ids = self.tok_encode(context)
            # Truncate context if needed
            if len(context_ids) > self.config.seq_length - max_gen:
                context_ids = context_ids[-(self.config.seq_length - max_gen):]

            context_tensor = torch.tensor(
                [context_ids], dtype=torch.long, device=self._device
            )

            generated = self._model_generate(
                context_tensor, max_length=max_gen, stop=until
            )

            # Decode only the generated part
            gen_ids = generated[0, len(context_ids):].tolist()
            gen_text = self.tok_decode(gen_ids)

            # Truncate at first stop string
            for stop_str in until:
                if stop_str in gen_text:
                    gen_text = gen_text[:gen_text.index(stop_str)]
                    break

            results.append(gen_text)

        return results
