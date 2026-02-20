"""Flash Attention 2 wrapper that monkey-patches GQA forward methods.

Preserves all frozen parameters (W_q, W_k, W_v, W_o, QK-norm, RoPE).
Replaces the manual attention computation (lines 158-173 of full_model.py)
with flash_attn_func / flash_attn_varlen_func.

Key optimizations:
- Flash Attention handles GQA natively (16 Q heads, 8 KV heads) — NO repeat_interleave
- causal=True replaces the manual torch.triu mask
- 1/sqrt(head_dim) scaling is handled internally by Flash Attention
- flash_attn_varlen_func enables document masking via cu_seqlens
"""

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func


def _make_flash_forward(gqa_module):
    """Create a Flash Attention forward function bound to a specific GQA instance.

    The returned function preserves the frozen QKV projections, QK-norm, and RoPE,
    then dispatches to flash_attn_varlen_func (with document masking) or
    flash_attn_func (standard causal).

    Args:
        gqa_module: A frozen GQA instance whose weights we use.

    Returns:
        A forward function with signature: forward(x, rope, cu_seqlens=None, max_seqlen=None)
    """
    def flash_forward(x, rope, cu_seqlens=None, max_seqlen=None):
        bs, seq_length, _ = x.shape

        # Frozen QKV projections + reshape — identical to frozen lines 148-150
        Q = gqa_module.W_q(x).reshape(bs, seq_length, gqa_module.n_q_heads, gqa_module.head_dim)
        K = gqa_module.W_k(x).reshape(bs, seq_length, gqa_module.n_kv_heads, gqa_module.head_dim)
        V = gqa_module.W_v(x).reshape(bs, seq_length, gqa_module.n_kv_heads, gqa_module.head_dim)

        # Frozen QK-norm — identical to frozen lines 152-153
        Q = gqa_module.rmsnorm_q(Q)
        K = gqa_module.rmsnorm_k(K)

        # Frozen RoPE — identical to frozen lines 155-156
        Q = rope(Q)
        K = rope(K)

        # Flash Attention replaces frozen lines 158-173
        # No repeat_interleave needed — Flash Attention handles GQA natively
        if cu_seqlens is not None and max_seqlen is not None:
            # Document masking: flatten batch into (total_tokens, heads, dim)
            Q_flat = Q.reshape(-1, gqa_module.n_q_heads, gqa_module.head_dim)
            K_flat = K.reshape(-1, gqa_module.n_kv_heads, gqa_module.head_dim)
            V_flat = V.reshape(-1, gqa_module.n_kv_heads, gqa_module.head_dim)

            attn_out = flash_attn_varlen_func(
                Q_flat, K_flat, V_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
            # Reshape back to (bs, seq_length, n_q_heads, head_dim)
            attn_out = attn_out.reshape(bs, seq_length, gqa_module.n_q_heads, gqa_module.head_dim)
        else:
            # Standard causal attention (no document masking)
            # flash_attn_func expects (bs, seq_len, heads, dim)
            attn_out = flash_attn_func(Q, K, V, causal=True)

        # Reshape and output projection — identical to frozen lines 171-173
        attn_out = attn_out.reshape(bs, seq_length, -1)
        return gqa_module.W_o(attn_out)

    return flash_forward


def patch_model_for_flash_attention(model):
    """Monkey-patch all GQA instances in the model to use Flash Attention 2.

    This replaces each GQA.forward with a Flash Attention version that:
    - Uses the same frozen weight matrices (W_q, W_k, W_v, W_o)
    - Applies the same QK-norm and RoPE
    - Replaces manual attention with flash_attn_func/flash_attn_varlen_func
    - Accepts optional cu_seqlens/max_seqlen for document masking

    Args:
        model: A PolychromaticLM instance.
    """
    import types
    for block in model.model_core:
        flash_fn = _make_flash_forward(block.gqa)
        block.gqa.forward = types.MethodType(lambda self, x, rope, cu_seqlens=None, max_seqlen=None, _fn=flash_fn: _fn(x, rope, cu_seqlens, max_seqlen), block.gqa)
