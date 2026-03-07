---
license: apache-2.0
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - pretrained
  - transformer
  - PolyGLU
  - activation-routing
  - math
  - research
  - from-scratch
model-index:
  - name: PolyChromaticLM-1.0-base-0.6B
    results:
      - task:
          type: multiple-choice
          name: HellaSwag
        dataset:
          name: HellaSwag
          type: hellaswag
        metrics:
          - type: acc_norm
            value: 28.51
            name: Normalized Accuracy
      - task:
          type: multiple-choice
          name: ARC-Easy
        dataset:
          name: ARC-Easy
          type: ai2_arc
          config: ARC-Easy
        metrics:
          - type: acc_norm
            value: 41.04
            name: Normalized Accuracy
      - task:
          type: multiple-choice
          name: ARC-Challenge
        dataset:
          name: ARC-Challenge
          type: ai2_arc
          config: ARC-Challenge
        metrics:
          - type: acc_norm
            value: 22.27
            name: Normalized Accuracy
      - task:
          type: multiple-choice
          name: PIQA
        dataset:
          name: PIQA
          type: piqa
        metrics:
          - type: acc_norm
            value: 58.87
            name: Normalized Accuracy
      - task:
          type: multiple-choice
          name: WinoGrande
        dataset:
          name: WinoGrande
          type: winogrande
        metrics:
          - type: acc
            value: 52.17
            name: Accuracy
      - task:
          type: multiple-choice
          name: BoolQ
        dataset:
          name: BoolQ
          type: boolq
        metrics:
          - type: acc
            value: 61.13
            name: Accuracy
      - task:
          type: multiple-choice
          name: SciQ
        dataset:
          name: SciQ
          type: sciq
        metrics:
          - type: acc_norm
            value: 61.20
            name: Normalized Accuracy
      - task:
          type: multiple-choice
          name: MMLU-STEM
        dataset:
          name: MMLU-STEM
          type: mmlu
          config: stem
        metrics:
          - type: acc
            value: 25.28
            name: Accuracy (5-shot)
---

<div align="center">

# PolyChromaticLM 1.0 Base (0.6B)

**A 597M-parameter transformer with biologically-inspired activation routing**

*Instead of a fixed activation function, each neuron dynamically selects among ReLU, Tanh, SiLU, and GELU — like biological neurons selecting neurotransmitters based on context.*

[![Paper](https://img.shields.io/badge/arXiv-2026-b31b1b.svg)](https://arxiv.org/)
[![Code](https://img.shields.io/badge/GitHub-PolyGLU-blue.svg)](https://github.com/danielxmed/PolyGLU)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

---

## Overview

PolyChromaticLM is a research language model built from scratch in PyTorch whose core innovation is **PolyGLU** (Polychromatic Gated Linear Unit) — a drop-in SwiGLU replacement that implements **state-conditional activation routing**. Rather than applying a single fixed activation function across all neurons, PolyGLU lets each FFN neuron dynamically choose among K=4 activation functions via a differentiable Gumbel-Softmax routing mechanism.

This is the **base pre-trained checkpoint** (no instruction tuning / SFT). It was trained on ~10B tokens with a math-heavy data mix on a single A100 80GB GPU.

**Author**: Daniel Nobrega (independent research)

### Key Results

- Routing converges to **near-deterministic selections** (entropy = 0.03% of maximum) without any explicit sparsity regularization — an emergent property
- Clear **depth-dependent activation specialization**: early layers prefer GELU, deep layers strongly prefer Tanh
- Achieves **62–89% of Qwen3-0.6B-Base** benchmark performance using **3,600x fewer training tokens**
- The routing mechanism adds only **0.23% parameter overhead** (~1.4M params)

---

## Architecture

| | |
|---|---|
| **Parameters** | 597M total (~1.4M routing, 0.23% overhead) |
| **Hidden dim** | 1,024 |
| **FFN dim** | 4,096 |
| **Layers** | 28 |
| **Attention** | GQA (16 query / 8 KV heads, head dim 64) |
| **Context** | 4,096 tokens |
| **Vocab** | 151,669 ([Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B-Base) tokenizer) |
| **Position encoding** | RoPE (θ=10,000) |
| **Normalization** | RMSNorm (pre-norm) + QK-Norm |
| **FFN** | **PolyGLU** (K=4: ReLU, Tanh, SiLU, GELU) |
| **Weight tying** | Embedding ↔ output head |

### How PolyGLU Works

Standard SwiGLU uses a fixed SiLU activation. PolyGLU generalizes this:

```
PolyGLU(x) = [Σ_k  g_k · σ_k(x · W_gate)] ⊙ (x · W_up)
```

where `g_k = GumbelSoftmax(α_k + β_k · f(h̄), τ)` and `σ_k ∈ {ReLU, Tanh, SiLU, GELU}`.

Each neuron has:
- **Static preference (α)**: a learned bias toward specific activations
- **Dynamic gating (β · f(h̄))**: a lightweight MLP that reads the mean-pooled hidden state and modulates routing based on context
- **Temperature (τ)**: annealed from 1.0→0.1 during training, controlling routing sharpness

The biological analogy: just as neurons select specific neurotransmitters (glutamate, GABA, dopamine, acetylcholine) depending on circuit state, PolyGLU neurons select activation functions depending on input context.

---

## Training

| | |
|---|---|
| **Tokens** | ~10.24B |
| **Steps** | 19,531 |
| **Hardware** | 1× NVIDIA A100 80GB |
| **Wall time** | ~12.5 days (~300 GPU-hours) |
| **Precision** | BFloat16 |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, ε=1e-8) |
| **Peak LR** | 1e-4 (cosine decay, 2K warmup) |
| **Weight decay** | 0.1 (weight matrices only) |
| **Batch size** | ~524K tokens/step |
| **Gradient clipping** | 1.0 max norm |
| **Final loss** | 1.31 |

### Data Mix

| Domain | Dataset | Share | Tokens |
|--------|---------|------:|-------:|
| Math | [`nvidia/Nemotron-CC-Math-v1`](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) (4+ subset) | 70% | ~7.0B |
| STEM | [`openbmb/Ultra-FineWeb`](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | 25% | ~2.5B |
| Code | [`lumees/github-code-2025-language-split`](https://huggingface.co/datasets/lumees/github-code-2025-language-split) (Python) | 5% | ~0.5B |

The final 20% of training anneals the mix to 85% math / 10% STEM / 5% code for math-focused refinement.

### Training Dynamics

<div align="center">
<img src="figures/combined_training_dynamics.png" alt="Training dynamics: loss, learning rate, tau annealing, and throughput" width="90%">
</div>

<details>
<summary><b>Loss curve detail</b></summary>
<img src="figures/loss_curve.png" alt="Loss curve from 12.13 to 1.31" width="80%">

| Step | Tokens | Loss |
|-----:|-------:|-----:|
| 0 | 0 | 12.13 |
| 2,000 | 1.05B | 3.50 |
| 10,000 | 5.24B | 2.26 |
| 15,000 | 7.86B | 1.68 |
| 19,531 | 10.24B | **1.31** |

</details>

---

## Evaluation

All benchmarks via [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-eval) v0.4.11, 0-shot unless noted.

### Benchmarks

| Benchmark | Metric | PolyChromaticLM | Random | Qwen3-0.6B-Base |
|-----------|--------|----------------:|-------:|----------------:|
| **HellaSwag** | acc_norm | 28.51 | 25.00 | 41.10 |
| **ARC-Easy** | acc_norm | 41.04 | 25.00 | 65.60 |
| **ARC-Challenge** | acc_norm | 22.27 | 25.00 | 33.90 |
| **PIQA** | acc_norm | 58.87 | 50.00 | 70.00 |
| **WinoGrande** | acc | 52.17 | 50.00 | 58.50 |
| **BoolQ** | acc | 61.13 | 50.00 | 69.70 |
| **SciQ** | acc_norm | 61.20 | 25.00 | — |
| **OpenBookQA** | acc_norm | 29.00 | 25.00 | — |
| **MMLU-STEM** | acc (5-shot) | 25.28 | 25.00 | — |
| **LAMBADA** | acc | 15.35 | ~0 | — |

<div align="center">
<img src="figures/eval_benchmark_comparison.png" alt="Benchmark comparison vs Qwen3-0.6B-Base" width="80%">
</div>

**Context**: Qwen3-0.6B-Base was trained on ~36T tokens — approximately **3,600× our budget**. Achieving 62–89% of its scores at 0.028% of the training compute demonstrates strong token efficiency for the PolyGLU architecture.

### Domain Perplexity

| Domain | Training Share | Perplexity | Bits/Token |
|--------|:-------------:|-----------:|-----------:|
| Math | 70% → 85% | **3.56** | 1.83 |
| Code | 5% | **7.08** | 2.82 |
| STEM | 25% → 10% | **31.93** | 5.00 |

<div align="center">
<img src="figures/eval_domain_perplexity.png" alt="Domain perplexity across math, code, and STEM" width="70%">
</div>

Code perplexity (7.08) is significantly lower than STEM (31.93) despite receiving 5× less data — evidence that mathematical structure transfers effectively to code patterns.

---

## Emergent Routing Behavior

The most striking finding from training: **the routing mechanism converges to near-deterministic activation selections without any explicit sparsity loss or entropy regularization.**

At convergence, mean dynamic routing entropy is **0.0004** (just 0.03% of the theoretical maximum), meaning the gate network makes near-one-hot activation choices for virtually every neuron.

<div align="center">
<img src="figures/dynamic_routing_entropy_final.png" alt="Per-layer dynamic routing entropy at convergence" width="80%">
</div>

### Layer-wise Activation Specialization

The model discovers a clear depth-dependent activation gradient:

- **Early layers (0–5)**: GELU-dominant (~35–40%) — smooth, probabilistic activations for initial feature extraction
- **Middle layers (6–14)**: Mixed — gradual transition with increasing Tanh representation
- **Deep layers (15–27)**: Tanh-dominant (~50–65%) — bounded compression for deep representational processing

<div align="center">
<img src="figures/layer_distribution.png" alt="Activation function preference by layer" width="90%">
</div>

Three layers (9, 16, 17) maintain elevated routing entropy, suggesting they benefit from activation diversity. Layer 17 notably *increases* its entropy during the second half of training — counter to the global trend toward determinism.

<div align="center">
<img src="figures/neurotransmitter_heatmap.png" alt="Neurotransmitter map: preferred activation per neuron across all layers" width="90%">
</div>

---

## Usage

This model was trained from scratch in pure PyTorch (no HuggingFace model wrappers). To load and use it:

```python
import torch
from transformers import AutoTokenizer

# Clone the training repo for model code
# git clone https://github.com/danielxmed/PolyGLU.git
from src.model.config import ModelConfig
from src.model.model import load_checkpoint

# Load model
config = ModelConfig(use_flash_attn=False)
model, step, tau = load_checkpoint("path/to/portable_final.pt", config, device="cuda")
model.eval()

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
input_ids = tokenizer("The derivative of x squared is", return_tensors="pt")["input_ids"].cuda()

# Generate (greedy, no KV cache)
with torch.no_grad():
    for _ in range(50):
        logits = model(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

print(tokenizer.decode(input_ids[0]))
```

> **Note**: This is a base model — it produces raw continuations, not instruction-following responses. An SFT version fine-tuned on math problem-solving is forthcoming.

---

## Limitations

- **Base model only** — no instruction tuning, no chat capability, no RLHF. Outputs are raw text continuations.
- **10B token training budget** — significantly less than comparable-size production models (Qwen3-0.6B: ~36T tokens). General knowledge and factual recall are limited.
- **Math-heavy distribution** (70% math) — strong on mathematical language modeling, weaker on general NLU tasks.
- **No KV cache** — inference requires the full training codebase; generation is slow without a dedicated inference implementation.
- **English only** — trained exclusively on English-language data.

---

## Citation

```bibtex
@misc{nobrega2026polychromaticLM,
  title   = {PolychromaticLM: State-Conditional Activation Routing via Neurotransmitter-Inspired Gated Linear Units},
  author  = {Daniel Nobrega},
  year    = {2026},
  url     = {https://huggingface.co/tylerxdurden/PolyChromaticLM-1.0-base-0.6B}
}
```

---

## Links

| | |
|---|---|
| **Code** | [github.com/danielxmed/PolyGLU](https://github.com/danielxmed/PolyGLU) |
| **Model** | [huggingface.co/tylerxdurden/PolyChromaticLM-1.0-base-0.6B](https://huggingface.co/tylerxdurden/PolyChromaticLM-1.0-base-0.6B) |
| **Weights & Biases** | [polychromatic-lm](https://wandb.ai/danielmedeiros-medeiros-nobrega-medtech/polychromatic-lm) |

---

<div align="center">
<i>Built from scratch on a single A100. Independent research by Daniel Nobrega.</i>
</div>
