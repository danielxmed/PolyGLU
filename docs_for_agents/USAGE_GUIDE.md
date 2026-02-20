# PolyGLU Production Codebase — Usage Guide

Step-by-step instructions for running the full training pipeline on a RunPod A100 80GB instance.

---

## Prerequisites

- RunPod account with A100 80GB community cloud (~$1.64/hr)
- HuggingFace account with access to gated Nemotron datasets
- Weights & Biases account for experiment tracking

## 1. Environment Setup

SSH into your RunPod instance, clone the repo, and run:

```bash
cd PolyGLU

# Create .env with your tokens
echo "HF_TOKEN=hf_your_token_here" > .env
echo "WANDB_API_KEY=your_wandb_key_here" >> .env

# Install everything and authenticate
bash scripts/setup_runpod.sh
```

This installs all dependencies (`requirements.txt`), logs into HuggingFace and WandB, and creates the data/checkpoint directories.

---

## 2. Tokenize Data

Before training, convert HuggingFace datasets into binary uint32 chunks. This runs on CPU — no GPU needed.

```bash
bash scripts/tokenize_data.sh
```

This tokenizes all 3 sources:
- **Math** (nvidia/Nemotron-CC-Math-v1, 4+ subset) → `data/tokenized/math/` (~7B tokens, ~28GB)
- **STEM** (nvidia/Nemotron-CC-v2) → `data/tokenized/stem/` (~2.5B tokens, ~10GB)
- **Code** (nvidia/Nemotron-CC-Code-v1) → `data/tokenized/code/` (~0.5B tokens, ~2GB)

Total: ~40GB of binary chunks. Each chunk is ~100M tokens (~400MB).

**Resume support**: If tokenization is interrupted, re-running the same command skips existing chunks automatically.

**Individual source** (if you want to tokenize one at a time):
```bash
python -m src.data.tokenize_dataset \
    --dataset nvidia/Nemotron-CC-Math-v1 \
    --config 4plus \
    --output data/tokenized/math \
    --max-tokens 7000000000
```

---

## 3. Pre-Training

```bash
bash scripts/run_train.sh
```

Or directly:
```bash
deepspeed --num_gpus=1 -m src.training.train --config configs/train_config.yaml
```

### What happens

- Creates the PolychromaticLM model (~600M params) with Flash Attention 2
- Initializes DeepSpeed ZeRO Stage 2 for memory optimization
- Trains for ~19,531 steps on ~10B tokens (~85-90 hours)
- Data mix: 70% math / 25% STEM / 5% code, annealing to 85/10/5 in the final 20%
- Gumbel-Softmax temperature anneals from 1.0 → 0.1

### Monitoring (WandB)

The following metrics are logged every 100 steps:
- `train/loss` — training loss
- `train/lr` — learning rate (linear warmup → cosine decay)
- `train/tau` — Gumbel-Softmax temperature
- `train/tokens_per_sec` — throughput
- `routing_entropy/layer_N` — per-layer routing entropy
- `routing_entropy/mean` — aggregate entropy (watch for collapse!)

**Key metric to watch**: `routing_entropy/mean` should stay well above 0.5. If it drops below 0.3, activation collapse may be occurring (PolyGLU reducing to SwiGLU).

### Checkpoints

Two checkpoint formats are saved:
- **DeepSpeed** (`checkpoints/ds_checkpoint/`): For resume. Saved every 1,000 steps.
- **Portable** (`checkpoints/portable_stepN.pt`): For eval/SFT. Saved every 5,000 steps.
- **Final** (`checkpoints/portable_final.pt`): Saved at training completion.

### Resume from checkpoint

Edit `configs/train_config.yaml`:
```yaml
training:
  resume_from: checkpoints/ds_checkpoint  # or path to portable .pt
```

Then re-run `bash scripts/run_train.sh`.

---

## 4. Supervised Fine-Tuning (SFT)

After pre-training completes:

```bash
bash scripts/run_sft.sh
```

### What happens

- Loads `checkpoints/portable_final.pt`
- Freezes tau at 0.1
- Fine-tunes on nvidia/Nemotron-Math-v2 (~347K problems)
- Loss is computed on assistant tokens only (user prompt tokens masked)
- Lower learning rate (2e-5), 1 epoch, ~4-6 hours

### Output

- Checkpoints in `checkpoints_sft/`
- Final model: `checkpoints_sft/sft_final.pt`

---

## 5. Evaluation

```bash
bash scripts/run_eval.sh
```

This runs benchmarks on both pre-trained and SFT models, plus interpretability analysis.

### Individual evaluation

```bash
# Pre-trained model
python -m src.evaluation.run_eval \
    --checkpoint checkpoints/portable_final.pt \
    --tasks gsm8k minerva_math mmlu_stem \
    --output results/pretrain_eval.json

# SFT model
python -m src.evaluation.run_eval \
    --checkpoint checkpoints_sft/sft_final.pt \
    --tasks gsm8k minerva_math \
    --output results/sft_eval.json
```

### Quick test (limited examples)

```bash
python -m src.evaluation.run_eval \
    --checkpoint checkpoints/portable_final.pt \
    --tasks gsm8k \
    --limit 10 \
    --output results/test_eval.json
```

### Benchmarks

| Task | What it measures | Method |
|---|---|---|
| gsm8k | Grade school math (multi-step reasoning) | generate_until |
| minerva_math | Competition math (MATH-500) | generate_until |
| mmlu_stem | Broad STEM knowledge | loglikelihood |

---

## 6. Interpretability Analysis

```bash
python -m src.interpretability.run_analysis \
    --checkpoint checkpoints/portable_final.pt \
    --output figures/
```

### Outputs

- `figures/neurotransmitter_heatmap.png` — Which activation each neuron prefers (per layer)
- `figures/layer_distribution.png` — Stacked bar chart of activation type proportions
- `figures/entropy_per_layer.png` — Mean routing entropy per layer (with error bars)
- `figures/entropy_histogram.png` — Distribution of per-neuron entropy
- `figures/dynamic_routing_comparison.png` — How gate_net shifts for arithmetic/algebra/geometry
- `figures/routing_shift_summary.png` — Gate sensitivity per layer
- `figures/entropy_stats.json` — Numerical entropy statistics

---

## Configuration Reference

All hyperparameters are in YAML configs. Defaults match the paper specification.

### `configs/train_config.yaml`

Key settings you might adjust:
```yaml
training:
  total_steps: 19531      # Reduce for shorter runs
  checkpoint_every: 1000   # More frequent if budget is tight
  log_every: 100           # WandB logging frequency
  micro_batch_size: 2      # Increase if memory allows (unlikely on A100 80GB)
  grad_accum_steps: 64     # Decrease to reduce effective batch size
```

### `configs/ds_config.json`

DeepSpeed ZeRO Stage 2 config. The key design decision: **no optimizer or scheduler sections**. We provide our own to exactly match the frozen Trainer's AdamW settings. If you add optimizer/scheduler to this JSON, DeepSpeed will override ours and training behavior will differ.

---

## Quick Verification (on RunPod)

Before committing to a full training run:

```bash
# 1. Verify model creation and parameter count
python -c "
from src.model.model import create_model
from src.model.config import ModelConfig
m = create_model(ModelConfig())
print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')
"
# Expected: ~600M params

# 2. Short training run (100 steps) on real data
# Edit train_config.yaml: total_steps: 100, checkpoint_every: 50
# Then: bash scripts/run_train.sh
# Verify: loss decreases, WandB logs appear, checkpoint saves/loads

# 3. Eval pipeline smoke test
python -m src.evaluation.run_eval \
    --checkpoint checkpoints/portable_step50.pt \
    --tasks gsm8k \
    --limit 5 \
    --output results/smoke_test.json
```

---

## File Structure

```
src/
├── model/
│   ├── architecture.py      # FROZEN classes (DO NOT MODIFY)
│   ├── config.py             # ModelConfig, TrainConfig, SFTConfig
│   ├── flash_attention.py    # Flash Attention 2 monkey-patch for GQA
│   └── model.py              # create_model() factory, entropy utils
├── data/
│   ├── tokenize_dataset.py   # HF → binary uint32 chunks
│   ├── dataloader.py         # Streaming loader with doc masking + mix ratios
│   └── sft_dataset.py        # SFT dataset with loss masking
├── training/
│   └── train.py              # Pre-training loop (DeepSpeed + WandB)
├── sft/
│   └── sft.py                # SFT training script
├── evaluation/
│   ├── model_wrapper.py      # lm-eval-harness LM interface
│   └── run_eval.py           # Evaluation entry point
└── interpretability/
    ├── neurotransmitter_maps.py
    ├── routing_entropy.py
    ├── dynamic_routing.py
    └── run_analysis.py
```

---

## Troubleshooting

**OOM during training**: Reduce `micro_batch_size` to 1 and increase `grad_accum_steps` to 128 (keeps effective batch size the same).

**NaN loss**: Likely Gumbel-Softmax instability in BF16. Check if loss spikes correlate with low tau values. Workaround: cast gate_net computation to FP32 in `architecture.py`'s PolyGLU.forward (wrap the gumbel_softmax call with `.float()` and cast back).

**Low routing entropy**: If `routing_entropy/mean` drops below 0.3, activation collapse is occurring. Consider adding entropy regularization (add `- lambda * mean_entropy` to the loss).

**Tokenization interrupted**: Re-run the same command — it counts existing chunks and resumes.

**DeepSpeed checkpoint corruption**: Fall back to a portable `.pt` checkpoint and restart training with `resume_from` pointing to it.
