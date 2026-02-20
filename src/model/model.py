"""Production model factory and utilities.

This is the ONLY entry point for model creation. It:
1. Instantiates the frozen PolychromaticLM
2. Applies Flash Attention patches (optional)
3. Patches forward methods to pass cu_seqlens through the layer stack
4. Casts to BFloat16
"""

import types
import torch
import torch.nn as nn

from src.model.architecture import PolychromaticLM
from src.model.config import ModelConfig


def _make_block_forward_with_doc_masking(block):
    """Patch TransformerBlock.forward to accept and pass cu_seqlens/max_seqlen."""
    original_polyglu_forward = block.polyglu.forward

    def block_forward(self, x, rope, cu_seqlens=None, max_seqlen=None):
        x_1 = self.rmsnorm_1(x)
        x_2 = self.gqa(x_1, rope, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x_3 = x + x_2
        x_4 = self.rmsnorm_2(x_3)
        x_5 = self.polyglu(x_4)
        output = x_5 + x_3
        return output

    return block_forward


def _make_model_forward_with_doc_masking(model):
    """Patch PolychromaticLM.forward to accept and pass cu_seqlens/max_seqlen."""

    def model_forward(self, token_ids, cu_seqlens=None, max_seqlen=None):
        x = self.embeddings(token_ids)
        for block in self.model_core:
            x = block(x, self.rope, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.rmsnorm(x)
        output = self.output_head(x)
        return output

    return model_forward


def create_model(config: ModelConfig) -> PolychromaticLM:
    """Create a production-ready PolychromaticLM instance.

    Args:
        config: ModelConfig with architecture hyperparameters.

    Returns:
        PolychromaticLM in BFloat16, optionally with Flash Attention patches.
    """
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

    if config.use_flash_attn:
        from src.model.flash_attention import patch_model_for_flash_attention
        patch_model_for_flash_attention(model)

        # Patch TransformerBlock forwards to pass cu_seqlens through
        for block in model.model_core:
            block.forward = types.MethodType(
                _make_block_forward_with_doc_masking(block), block
            )

        # Patch model forward to accept cu_seqlens
        model.forward = types.MethodType(
            _make_model_forward_with_doc_masking(model), model
        )

    model = model.to(dtype=torch.bfloat16)
    return model


def get_routing_entropy(model: PolychromaticLM) -> dict:
    """Compute routing entropy from learned alpha parameters.

    Returns per-layer entropy and mean entropy for WandB logging.

    Entropy is computed as: H = -sum(p * log(p)) where p = softmax(alpha)
    High entropy = diverse activation usage (good).
    Low entropy = collapsed to single activation (bad, reduces to SwiGLU).
    """
    entropies = {}
    for i, block in enumerate(model.model_core):
        alpha = block.polyglu.alpha  # [d_ff, K]
        probs = torch.softmax(alpha.float(), dim=-1)  # Use float32 for numerical stability
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # [d_ff]
        entropies[f"routing_entropy/layer_{i}"] = entropy.mean().item()

    entropies["routing_entropy/mean"] = sum(entropies.values()) / len(entropies)
    return entropies


def load_checkpoint(path: str, config: ModelConfig, device: str = "cpu") -> tuple:
    """Load a portable checkpoint (.pt format matching frozen Trainer).

    Args:
        path: Path to checkpoint file.
        config: ModelConfig for model instantiation.
        device: Device to load to.

    Returns:
        (model, step, tau) tuple.
    """
    # Create model WITHOUT flash attention for portability
    config_no_flash = ModelConfig(**{
        k: (False if k == "use_flash_attn" else v)
        for k, v in config.__dict__.items()
    })
    model = create_model(config_no_flash)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint.get("step", 0)
    tau = checkpoint.get("tau", 0.1)
    model.update_tau(step, 1)  # Just sets tau directly via the stored value path
    # Manually set tau to the exact stored value
    for block in model.model_core:
        block.polyglu.tau = tau

    return model, step, tau
