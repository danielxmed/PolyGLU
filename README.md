# PolychromaticLM

> **Work in Progress** — The model is currently in pre-training. Training infrastructure is built; benchmarks and results are not yet available.

A ~600M-parameter transformer language model featuring **PolyGLU** (PolychromaticGLU) — a drop-in replacement for SwiGLU in transformer FFN blocks that uses state-conditional activation routing inspired by neurotransmitter-receptor diversity in biological neural systems.

**Author**: Daniel Nobrega (independent research)
**arXiv preprint**: submitted February 20, 2026 (cs.LG, cross-listed cs.NE, cs.AI)

## What is PolyGLU?

Standard transformer FFN blocks use a single activation function (SiLU in SwiGLU). PolyGLU gives each FFN neuron access to K=4 qualitatively distinct activation functions (ReLU, Tanh, SiLU, GELU), with routing determined by:

- **Static preference (α)**: A learnable per-neuron preference vector over the 4 activations
- **Dynamic modulation (β · f(h))**: A lightweight gating network that conditions routing on the layer's hidden state

Routing is made differentiable via Gumbel-Softmax with temperature annealing (τ: 1.0 → 0.1 during training).

The routing overhead is ~0.23% of total parameters (~1.4M across 28 layers).

### Scientific Claim

By enriching the informational content per connection (multiple qualitatively distinct activation functions per neuron) rather than scaling parameter count, PolyGLU achieves greater expressive power per parameter. The biological analogy: synaptic behavior depends on neurotransmitter type, receptor subtype, and context — not just connection weight.

## Model Architecture

| Parameter | Value |
|---|---|
| Total Parameters | ~600M (+~1.4M routing) |
| Hidden Dimension | 1,024 |
| FFN Intermediate | 4,096 |
| Layers | 28 |
| Attention | GQA (16 query heads, 8 KV heads) |
| Position Encoding | RoPE (θ=10000) |
| Normalization | RMSNorm (pre-norm) + QK-Norm |
| Context Length | 4,096 tokens |
| Vocabulary | 151,669 (Qwen3 tokenizer) |
| Precision | BFloat16 |

## Training

### Pre-Training
- ~10B tokens on a math-focused data mix (70% math, 25% STEM, 5% code)
- Data sources: Nemotron-CC-Math-v1, Nemotron-CC-v2/v2.1, Nemotron-CC-Code-v1
- Single A100 80GB, ~85-90 hours
- AdamW optimizer, cosine LR schedule, Gumbel-Softmax τ annealing

### Supervised Fine-Tuning
- Nemotron-Math-v2 (~347K math problems)
- Loss masking on assistant tokens only
- 1-2 epochs, ~4-6 hours on A100

### Evaluation
- Benchmarks: GSM8K, MATH-500, MMLU-STEM
- Evaluated via EleutherAI lm-evaluation-harness

## Quick Start

### Setup
```bash
pip install -r requirements.txt
bash scripts/setup_runpod.sh
```

### Data Preparation
```bash
python -m src.data.tokenize_dataset --all --output-base data/tokenized
```

### Training
```bash
deepspeed --num_gpus=1 -m src.training.train --config configs/train_config.yaml
```

### SFT
```bash
deepspeed --num_gpus=1 -m src.sft.sft --config configs/sft_config.yaml
```

### Evaluation
```bash
python -m src.evaluation.run_eval --checkpoint checkpoints/portable_final.pt \
    --tasks gsm8k minerva_math mmlu_stem
```

### Interpretability
```bash
python -m src.interpretability.run_analysis --checkpoint checkpoints/portable_final.pt \
    --output figures/
```

## Repository Structure

```
PolyGLU/
├── src/
│   ├── model/           # Architecture (frozen) + production wrappers (Flash Attn, checkpointing)
│   ├── data/            # Tokenization pipeline, streaming dataloader, SFT dataset
│   ├── training/        # Pre-training loop (DeepSpeed + WandB)
│   ├── sft/             # Supervised fine-tuning pipeline
│   ├── evaluation/      # lm-evaluation-harness integration
│   └── interpretability/ # Neurotransmitter maps, routing entropy, dynamic routing
├── configs/             # YAML configs for training and SFT
├── scripts/             # Shell scripts for RunPod execution
└── docs_for_agents/     # Technical report + original Colab prototype
```

## Technology Stack

- PyTorch 2.x (model from scratch)
- Flash Attention 2 (with document masking)
- DeepSpeed ZeRO (single GPU memory optimization)
- Weights & Biases (experiment tracking)
- EleutherAI lm-evaluation-harness (benchmarks)

## License

See [LICENSE](LICENSE) for details.
