# PolyGLU — Project Instructions for Claude Code

## 1. Project Overview

**PolychromaticLM** is a ~600M-parameter transformer language model whose core
innovation is **PolyGLU** (PolychromaticGLU) — a drop-in replacement for SwiGLU
in transformer FFN blocks. PolyGLU uses state-conditional activation routing
inspired by neurotransmitter-receptor diversity in biological neural systems.

Each FFN neuron maintains a learnable static preference (α) over K=4 qualitatively
distinct activation functions (ReLU, Tanh, SiLU, GELU), dynamically modulated by
a lightweight gating network conditioned on the layer's hidden state. Routing is
made differentiable via Gumbel-Softmax with temperature annealing (τ: 1.0 → 0.1).

The routing overhead is ~0.23% of total parameters (~1.4M across 28 layers).

**Author**: Daniel Nobrega (independent research).
**arXiv preprint**: submitted February 20, 2026 (cs.LG, cross-listed cs.NE, cs.AI).

### Scientific Claim

By enriching the informational content per connection (multiple qualitatively
distinct activation functions per neuron) rather than scaling parameter count,
PolyGLU achieves greater expressive power per parameter. The biological analogy:
synaptic behavior depends on neurotransmitter type, receptor subtype, and context
— not just connection weight.

---

## 2. Repository Structure

```
PolyGLU/
├── CLAUDE.md                          # This file — project instructions for agents
├── LICENSE
├── .env                               # HF_TOKEN for gated dataset access (DO NOT COMMIT)
│
├── docs_for_people/                   # Human-facing documentation
│   ├── README.md                      # Project overview, usage, citation
│   └── (paper link, usage guides)
│
├── docs_for_agents/                   # Agent-facing technical specs
│   ├── BASE_REPORT.md                 # Full technical report (v1.1, Feb 2026)
│   └── daniels_base_work/
│       └── full_model.py              # Validated model + training loop (Colab origin)
│
├── src/                               # Production source code (to be created)
│   ├── model/                         # Model architecture (wrapped from full_model.py)
│   ├── data/                          # Data pipeline (download, tokenize, dataloader)
│   ├── training/                      # Training loop (DeepSpeed, checkpointing, logging)
│   ├── sft/                           # Supervised fine-tuning pipeline
│   ├── evaluation/                    # Benchmark evaluation scripts
│   └── interpretability/              # Neurotransmitter maps, routing entropy analysis
│
├── configs/                           # Training/eval configuration files
├── scripts/                           # Shell scripts for RunPod execution
└── tests/                             # Unit tests for data pipeline, model, training
```

---

## 3. The Macro Task

Convert `docs_for_agents/daniels_base_work/full_model.py` (a validated Colab
prototype) into **production-ready training and inference code** targeting a
**single A100 80GB on RunPod** (~$1.64/hr, community cloud).

The model architecture is the author's novel scientific contribution. It has been
validated and is final. The work is to build production infrastructure around it.

---

## 4. Hard Constraints

### Architecture Is Frozen
**DO NOT refactor, "improve", rename, or restructure these classes:**
- `PolyGLU` — the core innovation (state-conditional activation routing)
- `GQA` — grouped query attention with QK-norm
- `RoPE` — rotary position embeddings
- `RMSNorm` — root mean square normalization
- `TransformerBlock` — pre-norm transformer block (RMSNorm → GQA → residual → RMSNorm → PolyGLU → residual)
- `PolychromaticLM` — the full model (embeddings → N blocks → RMSNorm → output head, weight tying)

These classes may be **wrapped** (e.g., for Flash Attention integration, DeepSpeed
compatibility, or dtype handling) but their forward pass logic, parameter shapes,
and initialization schemes must remain exactly as implemented in `full_model.py`.

The `Trainer` class in `full_model.py` is a reference implementation. Production
training code should replicate its logic (optimizer config, LR schedule, τ annealing,
gradient accumulation, checkpoint format) while adding DeepSpeed, WandB, proper
data loading, and robustness.

### Technology Stack
- **PyTorch 2.x** (model from scratch, no HuggingFace model wrappers)
- **Flash Attention 2** (efficient attention with document masking support)
- **DeepSpeed ZeRO Stage 2** (memory optimization for single GPU)
- **BFloat16** mixed precision throughout
- **Weights & Biases** for experiment tracking
- **EleutherAI lm-evaluation-harness** for benchmarks

### Hardware & Budget
- **Single GPU**: A100 80GB, RunPod community cloud (~$1.64/hr)
- **Total budget**: ~$346 (~R$2,000)
  - Pre-training: ~85–90h (~$148)
  - SFT: ~4–6h (~$10)
  - VanillaLM baseline (if budget permits): ~65–70h (~$115)
  - Debugging/ablations: ~35h (~$57)
  - Evaluation: ~10h (~$16)
- **No room for wasted compute.** Every design decision must be budget-aware.

---

## 5. Model Specifications

| Parameter | Value |
|---|---|
| Total Parameters | ~600M (+~1.4M routing, ~0.23%) |
| Non-Embedding Parameters | ~440M |
| Hidden Dimension (d_model) | 1,024 |
| FFN Intermediate Dimension (d_ff) | 4,096 |
| Number of Layers | 28 |
| Query Heads | 16 |
| KV Heads | 8 (GQA) |
| Head Dimension | 64 |
| Position Encoding | RoPE (θ=10000) |
| Normalization | RMSNorm (pre-norm) + QK-Norm in attention |
| FFN Activation | PolyGLU (K=4: ReLU, Tanh, SiLU, GELU) |
| Embedding Tying | Yes (input/output share weights) |
| Context Length | 4,096 tokens |
| Precision | BFloat16 |
| Vocabulary Size | 151,669 (Qwen3 tokenizer) |

### PolyGLU Routing Details
- **Static component (α)**: per-neuron learnable preference vector ∈ R⁴, shape [d_ff, K] = [4096, 4]
- **Dynamic component (β · f(h))**: gating network receives mean-pooled hidden state
  - f(h) = Linear₂(ReLU(Linear₁(mean_pool(h))))
  - Linear₁: d_model → 32, Linear₂: 32 → K
  - β: learnable scalar per activation, initialized to 1.0
- **Routing**: g_k = GumbelSoftmax(α_k + β_k · f(h), τ)
- **Output**: PolyGLU(x) = [Σ_k g_k · σ_k(x · W_gate)] ⊙ (x · W_up), then W_down

### Weight Initialization
- All `nn.Linear` weights: Normal(0, 0.02)
- All `nn.Linear` biases: zeros (where present)
- `nn.Embedding` weights: Normal(0, 0.02)
- Residual scaling: W_o and W_down scaled by 1/√(2·n_layers) after init

---

## 6. Training Specifications

### Pre-Training
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| Peak Learning Rate | 1e-4 |
| LR Schedule | Cosine decay after warmup |
| Warmup Steps | 2,000 |
| Weight Decay | 0.1 (applied to 2D+ params only; 1D params exempt) |
| Effective Batch Size | ~512K tokens |
| Gradient Clipping | 1.0 (max_norm) |
| Gumbel-Softmax τ | 1.0 → 0.1 (linear annealing, clamped at 0.1) |
| Training Tokens | ~10B |
| Context Length | 4,096 tokens |
| Estimated Time | ~85–90 hours on A100 80GB |

### Data Mix (Pre-Training)
| Dataset | Share | Tokens | Role |
|---|---|---|---|
| Nemotron-CC-Math-v1 (4+ subset) | 70% | ~7B | Core math (LaTeX, equations, proofs) |
| Nemotron-CC-v2 / v2.1 | 25% | ~2.5B | General STEM (linguistic diversity) |
| Nemotron-CC-Code-v1 | 5% | ~0.5B | Code (Python, NumPy, SageMath) |

### Data Mix Annealing
- **First 80% of training**: baseline proportions (70/25/5)
- **Final 20% (annealing phase)**: math increases to ~85%, STEM decreases to ~10%, code stays at 5%

### Tokenization
- Qwen3 tokenizer (byte-level BPE, vocab 151,669) — reuse as-is, do NOT retrain
- Pre-training data stored as concatenated uint32 arrays in binary chunks (~100M tokens each)
- `<EOS>` tokens separate documents within chunks
- Dataloader implements document masking via Flash Attention 2 (no cross-document attention)

### SFT (Supervised Fine-Tuning)
- **Dataset**: Nemotron-Math-v2 (~347K problems, 7M reasoning trajectories)
- **Epochs**: 1–2
- **Duration**: ~4–6 hours on A100
- **Loss masking**: Loss computed on assistant tokens only (user prompt tokens masked)
- **Optional exploration**: MetaMathQA, NuminaMath-CoT (only if primary SFT results warrant it)

---

## 7. Verified HuggingFace Resources

**CRITICAL: Use ONLY these exact repo IDs. Do NOT hallucinate, guess, or construct HF paths.**

### Tokenizer (reuse as-is)
- `Qwen/Qwen3-0.6B-Base` — https://hf.co/Qwen/Qwen3-0.6B-Base

### Pre-Training Data
- `nvidia/Nemotron-CC-Math-v1` — https://hf.co/datasets/nvidia/Nemotron-CC-Math-v1 (70%, math core)
- `nvidia/Nemotron-CC-v2` — https://hf.co/datasets/nvidia/Nemotron-CC-v2 (25%, general STEM)
- `nvidia/Nemotron-CC-v2.1` — https://hf.co/datasets/nvidia/Nemotron-CC-v2.1 (complementary to v2)
- `nvidia/Nemotron-CC-Code-v1` — https://hf.co/datasets/nvidia/Nemotron-CC-Code-v1 (5%, code)

### SFT Data (Primary)
- `nvidia/Nemotron-Math-v2` — https://hf.co/datasets/nvidia/Nemotron-Math-v2

### SFT Data (Optional — only if primary results warrant exploration)
- `meta-math/MetaMathQA` — https://hf.co/datasets/meta-math/MetaMathQA
- `AI-MO/NuminaMath-CoT` — https://hf.co/datasets/AI-MO/NuminaMath-CoT

### Evaluation
- `lm-evaluation-harness` (EleutherAI) — install via pip

### Dataset Access
All gated datasets are accessed via `HF_TOKEN` stored in `.env`.
Use `huggingface_hub` login or pass token directly to `load_dataset()`.

---

## 8. Task Breakdown (Phases)

### Phase 1: Data Pipeline
- Download and inspect Nemotron datasets (streaming where possible)
- Tokenize with Qwen3 tokenizer → binary uint32 chunks (~100M tokens each)
- Build streaming dataloader with:
  - Document masking (EOS-delimited, Flash Attention 2 compatible)
  - Data mix ratio enforcement (70/25/5)
  - Data mix annealing logic (last 20% of training)
  - Effective batch size ~512K tokens via micro-batching + gradient accumulation

### Phase 2: Production Training Loop
- Wrap model classes for DeepSpeed ZeRO Stage 2 + BFloat16
- Integrate Flash Attention 2 (replace manual attention in GQA)
- Replicate `Trainer` logic from full_model.py:
  - AdamW with param group separation (decay vs no-decay)
  - Cosine LR schedule with 2000-step warmup
  - Gumbel-Softmax τ annealing (linear, clamped at 0.1)
  - Gradient accumulation and clipping
- Add WandB logging (loss, LR, τ, gradient norm, throughput, routing entropy)
- Checkpointing: save model + optimizer + scheduler + step + τ state
- Resume from checkpoint support

### Phase 3: SFT Pipeline
- Load pre-trained checkpoint
- Format Nemotron-Math-v2 for chat (user/assistant turns)
- Loss masking on assistant tokens only
- 1–2 epochs, lower LR (likely 1e-5 or 2e-5 range)
- Save SFT checkpoint

### Phase 4: Evaluation
- Integrate lm-evaluation-harness
- Benchmarks: GSM8K, MATH-500, MMLU-STEM, perplexity
- Run on both pre-trained and SFT checkpoints
- Compare against external references: Qwen3-0.6B-Base, Pythia-410M, SmolLM-360M

### Phase 5: Interpretability
- Extract neurotransmitter maps: argmax of learned α per neuron per layer
- Visualize activation function distribution across layers
- Compute routing entropy (per-layer and aggregate)
- Dynamic routing analysis: how β · f(h) shifts for different input types
- Verify high entropy (activation diversity, not collapse to single function)

### Phase 6: VanillaLM Baseline (if budget permits)
- Identical architecture but replace PolyGLU with standard SwiGLU
- Train on same data for same number of tokens
- Most scientifically rigorous comparison for the paper

---

## 9. Success Criteria

1. **Stable training**: PolyGLU model trains to completion on ~10B tokens without loss spikes or NaN
2. **Measurable benchmark scores**: Quantitative results on GSM8K and MATH-500
3. **Routing diversity**: Entropy of activation selections remains high across layers (not collapsed to a single activation — which would reduce PolyGLU to standard SwiGLU)
4. **Controlled comparison**: VanillaLM baseline (if budget permits) demonstrates PolyGLU advantage
5. **Interpretable specialization**: Neurotransmitter maps show meaningful layer-wise patterns (hypothesis: early layers → ReLU, middle → GELU/SiLU, final → Tanh)

---

## 10. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Activation collapse (low routing entropy) | PolyGLU reduces to SwiGLU, nullifying contribution | Entropy regularization; diversity reward; monitor routing entropy in WandB |
| Compute overhead exceeds budget | Insufficient training tokens | Hard activation selection path; reduce K to 2 if needed |
| Gumbel-Softmax instability | Loss spikes, NaN gradients | Careful τ schedule; gradient clipping; consider separate LR for routing params |
| Unfair comparison vs baselines | Reviewers question validity | Always report tokens trained; prioritize controlled VanillaLM comparison |

---

## 11. Coding Conventions

- Python 3.10+, PyTorch 2.x idioms
- Type hints on public function signatures
- Configuration via dataclasses or YAML — no magic numbers in training scripts
- All hyperparameters from Section 6 must be configurable, with defaults matching the spec
- Scripts should be runnable via `python -m src.training.train` or similar module paths
- Shell scripts in `scripts/` for RunPod-specific setup (pip installs, env vars, launch commands)
- `.env` is in `.gitignore` — never commit tokens

---

## 12. What NOT to Do

- **DO NOT** modify the forward pass logic of any frozen architecture class
- **DO NOT** rename parameters or change tensor shapes in the model
- **DO NOT** use HuggingFace model wrappers (transformers.AutoModel, etc.) — the model is from scratch
- **DO NOT** guess or fabricate HuggingFace dataset/model paths — use only those listed in Section 7
- **DO NOT** retrain the tokenizer — reuse Qwen3's as-is
- **DO NOT** over-engineer — this is a budget-constrained research project, not enterprise software
- **DO NOT** add speculative features, abstractions, or "nice to have" infrastructure
- **DO NOT** commit .env or any file containing HF_TOKEN
