#!/bin/bash
# Base pre-trained model evaluation pipeline
# Runs: smoke test → Phase A (loglikelihood) → Phase B (GSM8K) → Phase C (domain perplexity)
set -e

CHECKPOINT="checkpoints/portable_final.pt"
OUTDIR="results/base_eval"
mkdir -p "$OUTDIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "============================================================"
echo "BASE MODEL EVALUATION PIPELINE"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTDIR/"
echo "============================================================"

# ------------------------------------------------------------------
# SMOKE TEST (limit=5, verify everything runs)
# ------------------------------------------------------------------
echo ""
echo "[SMOKE TEST] Running 3 tasks with limit=5..."
python -m src.evaluation.run_eval \
    --checkpoint "$CHECKPOINT" \
    --tasks hellaswag gsm8k lambada_openai \
    --batch-size 1 \
    --limit 5 \
    --output "$OUTDIR/smoke_test.json"
echo "[SMOKE TEST] PASSED"

# ------------------------------------------------------------------
# PHASE A: Loglikelihood benchmarks (~40 min)
# ------------------------------------------------------------------
echo ""
echo "[PHASE A] Loglikelihood benchmarks (10 tasks)..."
python -m src.evaluation.run_eval \
    --checkpoint "$CHECKPOINT" \
    --tasks hellaswag arc_easy arc_challenge piqa winogrande boolq \
           mmlu_stem lambada_openai openbookqa sciq \
    --batch-size 4 \
    --output "$OUTDIR/loglikelihood_benchmarks.json"
echo "[PHASE A] COMPLETE"

# ------------------------------------------------------------------
# PHASE B: GSM8K generation (~3.5 hours)
# ------------------------------------------------------------------
echo ""
echo "[PHASE B] GSM8K generation (1319 examples)..."
python -m src.evaluation.run_eval \
    --checkpoint "$CHECKPOINT" \
    --tasks gsm8k \
    --batch-size 1 \
    --output "$OUTDIR/gsm8k_results.json"
echo "[PHASE B] COMPLETE"

# ------------------------------------------------------------------
# PHASE C: Domain perplexity (~1 hour)
# ------------------------------------------------------------------
echo ""
echo "[PHASE C] Domain perplexity (math, stem, code)..."
python -m src.evaluation.domain_perplexity \
    --checkpoint "$CHECKPOINT" \
    --data-dir data/tokenized \
    --num-sequences 244 \
    --output "$OUTDIR/domain_perplexity.json"
echo "[PHASE C] COMPLETE"

echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE"
echo "Results in: $OUTDIR/"
ls -la "$OUTDIR/"
echo "============================================================"
