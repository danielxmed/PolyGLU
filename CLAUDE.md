# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup (RunPod A100 80GB)
```bash
bash scripts/setup_runpod.sh   # pip install, HF login, WandB login, create dirs
```

### Data Pipeline
```bash
# Tokenize all datasets → binary uint32 chunks in data/tokenized/{math,stem,code}/
python -m src.data.tokenize_dataset --all --output-base data/tokenized

# Or tokenize individually:
python -m src.data.tokenize_dataset --source math --output-base data/tokenized
python -m src.data.tokenize_dataset --source stem --output-base data/tokenized
python -m src.data.tokenize_dataset --source code --output-base data/tokenized
```

### Training
```bash
# Pre-training (DeepSpeed single GPU)
deepspeed --num_gpus=1 -m src.training.train --config configs/train_config.yaml

# SFT (requires pre-trained checkpoint at checkpoints/portable_final.pt)
deepspeed --num_gpus=1 -m src.sft.sft --config configs/sft_config.yaml
```

### Evaluation
```bash
# Benchmark eval (gsm8k, minerva_math, mmlu_stem)
python -m src.evaluation.run_eval --checkpoint checkpoints/portable_final.pt \
    --tasks gsm8k minerva_math mmlu_stem --output results/pretrain_eval.json

# With example limit for quick testing
python -m src.evaluation.run_eval --checkpoint checkpoints/portable_final.pt \
    --tasks gsm8k --limit 10
```

### Interpretability
```bash
python -m src.interpretability.run_analysis --checkpoint checkpoints/portable_final.pt \
    --output figures/
```

### Shell Script Wrappers
```bash
bash scripts/tokenize_data.sh   # tokenize all sources
bash scripts/run_train.sh       # pre-training
bash scripts/run_sft.sh         # SFT
bash scripts/run_eval.sh        # eval + interpretability (both checkpoints)
```

---

## Code Architecture

### Module Dependency Flow
```
configs/*.yaml → src/model/config.py (ModelConfig, TrainConfig, SFTConfig dataclasses)
                        ↓
         src/model/model.py  ←  src/model/architecture.py (FROZEN classes)
         create_model()      ←  src/model/flash_attention.py (monkey-patches GQA)
                ↓
    ┌───────────┼──────────────────┐
    ↓           ↓                  ↓
src/training/   src/sft/           src/evaluation/
train.py        sft.py             run_eval.py → model_wrapper.py
    ↓           ↓
src/data/       src/data/
dataloader.py   sft_dataset.py
```

### Key Entry Points
- **Model creation**: Always via `src/model/model.py:create_model(config)` — handles Flash Attention patching, gradient checkpointing, BFloat16 casting
- **Model loading**: `src/model/model.py:load_checkpoint(path, config, device)` — returns (model, step, tau), creates model WITHOUT Flash Attention for portability
- **Routing entropy**: `src/model/model.py:get_routing_entropy(model)` — returns WandB-compatible dict

### Frozen Architecture (`src/model/architecture.py`)
These classes are copied verbatim from `docs_for_agents/daniels_base_work/full_model.py`. **DO NOT modify their forward pass, parameter shapes, or initialization:**
- `PolyGLU` — state-conditional activation routing (the core innovation)
- `GQA` — grouped query attention with QK-norm
- `RoPE` — rotary position embeddings
- `RMSNorm` — root mean square normalization
- `TransformerBlock` — pre-norm block (RMSNorm → GQA → residual → RMSNorm → PolyGLU → residual)
- `PolychromaticLM` — full model (embeddings → 28 blocks → RMSNorm → output head, weight tying)

These may be **wrapped** (Flash Attention, DeepSpeed, dtype) but never modified.

### Cross-Cutting Design Patterns

**Checkpoint Duality**: Two checkpoint formats serve different purposes:
- DeepSpeed checkpoints (`checkpoints/ds_checkpoint/`): For training resume (full optimizer state)
- Portable `.pt` checkpoints (`checkpoints/portable_*.pt`): For eval/SFT/interpretability. Format: `{model_state_dict, optimizer_state_dict, scheduler_state_dict, step, tau}`

**Document Masking Flow**: EOS tokens (ID 151643) delimit documents in binary chunks. `dataloader.py` extracts `cu_seqlens` from EOS positions → passed through `model.py` patched forward → down to `flash_attention.py` which calls `flash_attn_varlen_func`. This prevents cross-document attention within packed sequences.

**Chunked Cross-Entropy** (`train.py:chunked_cross_entropy`): Splits the output head computation along sequence dimension (chunk_size=2048) with per-chunk gradient checkpointing. Avoids materializing the full `[B*T, 151669]` logit tensor (~15-20GB saving on A100).

**Tau Annealing**: Gumbel-Softmax temperature (1.0 → 0.1, linear, clamped at 0.1). Updated each step via `model.update_tau(step, total_steps)`. Stored in checkpoints for reproducibility. Frozen at 0.1 during SFT.

**Data Mix Annealing** (`dataloader.py:MixingScheduler`): First 80% of training uses baseline ratios (70% math, 25% STEM, 5% code). Final 20% anneals to (85% math, 10% STEM, 5% code). Scheduler takes current step, returns sampling probabilities.

**Configuration**: YAML files → dataclass instances via `.from_yaml()`. Zero magic numbers in training scripts. All hyperparameters configurable with spec-matching defaults.

---

## Project Overview

**PolychromaticLM** is a ~600M-parameter transformer whose core innovation is **PolyGLU** — a drop-in SwiGLU replacement using state-conditional activation routing inspired by neurotransmitter-receptor diversity. Each FFN neuron maintains a learnable static preference (α) over K=4 activation functions (ReLU, Tanh, SiLU, GELU), dynamically modulated by a lightweight gating network conditioned on the hidden state. Routing is differentiable via Gumbel-Softmax with temperature annealing.

**Author**: Daniel Nobrega (independent research). **arXiv**: submitted Feb 20, 2026.

### The Macro Task
Convert `docs_for_agents/daniels_base_work/full_model.py` (validated Colab prototype) into production-ready training/inference code targeting a single A100 80GB on RunPod. The model architecture is final — the work is building production infrastructure around it.

---

## Hard Constraints

### Technology Stack
- **PyTorch 2.x** (model from scratch, no HuggingFace model wrappers)
- **Flash Attention 2** (efficient attention with document masking)
- **DeepSpeed ZeRO** (memory optimization for single GPU)
- **BFloat16** mixed precision throughout
- **Weights & Biases** for experiment tracking
- **EleutherAI lm-evaluation-harness** for benchmarks

### Hardware & Budget
- Single GPU: A100 80GB, RunPod community cloud (~$1.64/hr)
- Total budget: ~$346 — no room for wasted compute

---

## Model Specifications

| Parameter | Value |
|---|---|
| Total Parameters | ~600M (+~1.4M routing, ~0.23%) |
| Hidden Dimension (d_model) | 1,024 |
| FFN Intermediate (d_ff) | 4,096 |
| Layers | 28 |
| Query / KV Heads | 16 / 8 (GQA) |
| Head Dimension | 64 |
| Position Encoding | RoPE (θ=10000) |
| Normalization | RMSNorm (pre-norm) + QK-Norm |
| FFN Activation | PolyGLU (K=4: ReLU, Tanh, SiLU, GELU) |
| Context Length | 4,096 tokens |
| Vocabulary | 151,669 (Qwen3 tokenizer, reuse as-is) |

### PolyGLU Routing Details
- **Static (α)**: per-neuron preference vector ∈ R⁴, shape [d_ff, K] = [4096, 4]
- **Dynamic (β · f(h))**: gate_net receives mean-pooled hidden state → Linear(d_model→32) → ReLU → Linear(32→K); β: learnable scalar per activation, init 1.0
- **Routing**: g_k = GumbelSoftmax(α_k + β_k · f(h), τ)
- **Output**: PolyGLU(x) = [Σ_k g_k · σ_k(x · W_gate)] ⊙ (x · W_up), then W_down

### Weight Initialization
- `nn.Linear` weights: Normal(0, 0.02), biases: zeros
- `nn.Embedding`: Normal(0, 0.02)
- Residual scaling: W_o and W_down scaled by 1/√(2·n_layers)

---

## Training Specifications

### Pre-Training
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| Peak LR | 1e-4 (cosine decay, 2000-step warmup) |
| Weight Decay | 0.1 (2D+ params only; 1D exempt) |
| Effective Batch | ~512K tokens (micro_batch=16, grad_accum=8, seq=4096) |
| Gradient Clipping | 1.0 (max_norm) |
| Gumbel-Softmax τ | 1.0 → 0.1 (linear annealing, clamped) |
| Training Tokens | ~10B (~19,531 steps) |

### Data Mix
| Dataset | Share | Tokens |
|---|---|---|
| nvidia/Nemotron-CC-Math-v1 (4+ subset) | 70% | ~7B |
| nvidia/Nemotron-CC-v2 / v2.1 | 25% | ~2.5B |
| nvidia/Nemotron-CC-Code-v1 | 5% | ~0.5B |

Annealing: last 20% of training shifts to 85% math, 10% STEM, 5% code.

### Data Format
- Binary uint32 chunks (~100M tokens each) in `data/tokenized/{math,stem,code}/`
- EOS token (151643) separates documents within chunks
- Each chunk has a `manifest.json` with metadata

### SFT
- Dataset: nvidia/Nemotron-Math-v2 (~347K problems)
- 1–2 epochs, LR 2e-5, micro_batch=2, grad_accum=16
- Loss on assistant tokens only (ChatML format)

---

## Verified HuggingFace Resources

**CRITICAL: Use ONLY these exact repo IDs. Do NOT hallucinate or fabricate HF paths.**

- **Tokenizer**: `Qwen/Qwen3-0.6B-Base`
- **Math**: `nvidia/Nemotron-CC-Math-v1` (config="4plus")
- **STEM**: `nvidia/Nemotron-CC-v2`, `nvidia/Nemotron-CC-v2.1`
- **Code**: `nvidia/Nemotron-CC-Code-v1`
- **SFT**: `nvidia/Nemotron-Math-v2`
- **SFT (optional)**: `meta-math/MetaMathQA`, `AI-MO/NuminaMath-CoT`

All gated datasets accessed via `HF_TOKEN` in `.env`.

---

## Task Phases

1. **Data Pipeline**: Tokenize → binary chunks → streaming dataloader with document masking and mix annealing
2. **Training Loop**: DeepSpeed + Flash Attention + WandB + checkpointing + τ annealing
3. **SFT**: Load pretrained → fine-tune on math conversations → loss masking
4. **Evaluation**: lm-evaluation-harness (GSM8K, MATH-500, MMLU-STEM, perplexity)
5. **Interpretability**: Neurotransmitter maps, routing entropy, dynamic routing analysis
6. **VanillaLM Baseline** (if budget permits): Same architecture with standard SwiGLU for controlled comparison

---

## Success Criteria

1. Stable training to ~10B tokens (no loss spikes/NaN)
2. Quantitative benchmark scores (GSM8K, MATH-500)
3. High routing entropy (activation diversity, not collapse to single function)
4. Interpretable layer-wise specialization patterns in neurotransmitter maps

---

## What NOT to Do

- **DO NOT** modify forward pass logic, parameter names, or tensor shapes of frozen architecture classes
- **DO NOT** use HuggingFace model wrappers (transformers.AutoModel, etc.)
- **DO NOT** guess or fabricate HuggingFace dataset/model paths
- **DO NOT** retrain the tokenizer
- **DO NOT** over-engineer — budget-constrained research project, not enterprise software
- **DO NOT** add speculative features or "nice to have" infrastructure
- **DO NOT** commit `.env` or any file containing HF_TOKEN

## Coding Conventions

- Python 3.10+, PyTorch 2.x idioms
- Type hints on public function signatures
- Configuration via dataclasses + YAML — no magic numbers in training scripts
- Module-style execution: `python -m src.training.train`, `deepspeed -m src.training.train`
