# =============================================================================
# FROZEN — DO NOT MODIFY
#
# These classes are copied verbatim from docs_for_agents/daniels_base_work/full_model.py.
# They represent the author's novel scientific contribution and are final.
# They may be WRAPPED (e.g., Flash Attention, DeepSpeed) but their forward pass
# logic, parameter shapes, and initialization schemes must remain exactly as-is.
# =============================================================================

import torch
import torch.nn as nn
import math


class PolyGLU(nn.Module):
  def __init__(self, d_model, d_ff, n_activations=4):
    super().__init__()

    self.W_gate = nn.Linear(d_model, d_ff, bias=False)
    self.W_up = nn.Linear(d_model, d_ff, bias=False)
    self.W_down = nn.Linear(d_ff, d_model, bias=False)

    self.activations = [
        nn.functional.relu, # ~ Glutamate -> hard threshold or hard activation
        torch.tanh, # ~ GABA -> simetric compression or inhibitory
        nn.functional.silu, # ~ dopamine -> self-gated or reward system
        nn.functional.gelu # ~ acetylcholine -> probabilistic gate or cognition conductor
    ]

    self.alpha = nn.Parameter(torch.zeros(d_ff, n_activations))
    self.beta = nn.Parameter(torch.ones(n_activations))

    self.tau = 1.0

    self.gate_net = nn.Sequential(
        nn.Linear(d_model, 32),
        nn.ReLU(),
        nn.Linear(32, n_activations)
    )

  def forward (self, x):
    mean_pool_h = torch.mean(x, 1)

    logits = self.alpha.unsqueeze(0) + (self.beta * self.gate_net(mean_pool_h).unsqueeze(1))

    g_k = nn.functional.gumbel_softmax(logits, tau=self.tau).unsqueeze(1)

    gate_x = self.W_gate(x)
    value_x = self.W_up(x)

    nt_out = torch.stack([nt(gate_x) for nt in self.activations], dim=-1)

    polyglu_sum = (g_k * nt_out).sum(dim=-1)

    polyglu_output = polyglu_sum * value_x
    polyglu_output = self.W_down(polyglu_output)

    return polyglu_output

class RMSNorm(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()

    self.gama = nn.Parameter(torch.ones(d_model))
    self.eps = eps

  def forward(self, x):

    denominator = torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)

    output = (x / denominator) * self.gama
    return output

class RoPE(nn.Module):
  def __init__(self, head_dim, seq_length):
    super().__init__()

    self.head_dim = head_dim

    i = torch.arange(0, head_dim//2, 1, dtype=torch.float32) # [32]
    freqs = 1 / 10000**(2*i/head_dim) # [32]
    pos = torch.arange(0, seq_length, 1, dtype=torch.float32) # [4096]
    thetas = pos.unsqueeze(1) * freqs.unsqueeze(0) # [4096, 32]

    cos_cached = torch.cos(thetas)
    sin_cached = torch.sin(thetas)

    self.register_buffer('cosines', cos_cached)
    self.register_buffer('sins', sin_cached)

  def forward(self, x):  # x ~ [bs, seq_length, num_heads, head_dim]
    seq_len = x.shape[1]

    x_1 = x[:, :, :, :self.head_dim // 2] #  ~ bs, seq_length, num_heads, 32]
    x_2 = x[:, :, :, self.head_dim // 2:]#  ~ bs, seq_length, num_heads, 32]

    cos = self.cosines[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = self.sins[:seq_len].unsqueeze(0).unsqueeze(2)


    x_1_rotated = x_1 * cos - x_2 * sin
    x_2_rotated = x_1 * sin + x_2 * cos

    output = torch.cat((x_1_rotated, x_2_rotated), dim=-1)

    return output

class GQA(nn.Module):
  def __init__(self, model_dim, n_q_heads, n_kv_heads, eps):
    super().__init__()

    self.head_dim = model_dim // n_q_heads
    self.model_dim = model_dim
    self.n_q_heads = n_q_heads
    self.n_kv_heads = n_kv_heads
    self.n_rep = n_q_heads // n_kv_heads
    self.rmsnorm_q = RMSNorm(self.head_dim, eps)
    self.rmsnorm_k = RMSNorm(self.head_dim, eps)

    self.W_q = nn.Linear(model_dim, self.head_dim * n_q_heads, bias=False)
    self.W_k = nn.Linear(model_dim, self.head_dim * n_kv_heads, bias=False)
    self.W_v = nn.Linear(model_dim, self.head_dim * n_kv_heads, bias=False)
    self.W_o = nn.Linear(self.head_dim * n_q_heads, model_dim, bias=False)

  def forward(self, x, rope):
    bs, seq_length, _ = x.shape

    Q = self.W_q(x).reshape(bs, seq_length, self.n_q_heads, self.head_dim) #[bs, seq_len, 16, 64]
    K = self.W_k(x).reshape(bs, seq_length, self.n_kv_heads, self.head_dim) #[bs, seq_len, 8, 64]
    V = self.W_v(x).reshape(bs, seq_length, self.n_kv_heads, self.head_dim) #[bs, seq_len, 8, 64]

    Q = self.rmsnorm_q(Q)
    K = self.rmsnorm_k(K)

    Q_rotated = rope(Q)
    K_rotated = rope(K)

    K_expanded = torch.repeat_interleave(K_rotated, self.n_rep, 2)
    V_expanded = torch.repeat_interleave(V, self.n_rep, 2)

    Q_rotated = Q_rotated.transpose(1, 2)
    K_expanded = K_expanded.transpose(1, 2)
    V_expanded = V_expanded.transpose(1, 2)

    product_qk = (Q_rotated @ K_expanded.transpose(-1, -2)) / math.sqrt(self.head_dim)

    mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
    masked_qk = product_qk.masked_fill(mask, float('-inf'))

    qkv = torch.softmax(masked_qk, dim=-1) @ V_expanded # [bs, 16, seq_len, head_dim]
    qkv = qkv.transpose(1, 2)
    qkv = qkv.reshape(bs, seq_length, -1) # [bs, seq_length,  head_dim * n_q_heads]
    attention_output = self.W_o(qkv)

    return attention_output

class TransformerBlock(nn.Module):
  def __init__(self, d_model, head_dim, seq_length, n_activations, eps, n_q_heads, n_kv_heads, d_ff):
    super().__init__()

    self.rmsnorm_1 = RMSNorm(d_model, eps)
    self.rmsnorm_2 = RMSNorm(d_model, eps)
    self.polyglu = PolyGLU(d_model, d_ff, n_activations)
    self.gqa = GQA(d_model, n_q_heads, n_kv_heads, eps)

  def forward(self, x, rope):
    x_1 = self.rmsnorm_1(x)
    x_2 = self.gqa(x_1, rope)
    x_3 = x + x_2
    x_4 = self.rmsnorm_2(x_3)
    x_5 = self.polyglu(x_4)
    output = x_5 + x_3

    return output

class PolychromaticLM(nn.Module):

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  def __init__(self, vocab_size, d_model, eps, head_dim, seq_length, n_activations, n_q_heads, n_kv_heads, d_ff, n_layers):
    super().__init__()

    self.rope = RoPE(head_dim, seq_length)

    self.embeddings = nn.Embedding(vocab_size, d_model)

    self.model_core = nn.ModuleList ([
        TransformerBlock(d_model, head_dim, seq_length, n_activations, eps, n_q_heads, n_kv_heads, d_ff) for _ in range(n_layers)
    ])

    self.rmsnorm = RMSNorm(d_model, eps)

    self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    self.output_head.weight = self.embeddings.weight

    self.apply(self._init_weights)

    scale = 1.0 / math.sqrt(2 * n_layers)

    for block in self.model_core:
      block.gqa.W_o.weight.data.mul_(scale)
      block.polyglu.W_down.weight.data.mul_(scale)


  def update_tau(self, step, total_steps):
      tau_max = 1.0
      tau_min = 0.1
      tau = tau_max - (tau_max - tau_min) * (step / total_steps)
      tau = max(tau, tau_min)
      for block in self.model_core:
          block.polyglu.tau = tau


  def forward(self, token_ids):
    x = self.embeddings(token_ids) #bs, seq_len, d_model

    for block in self.model_core:
      x = block(x, self.rope)

    x = self.rmsnorm(x)

    output = self.output_head(x)

    return output
