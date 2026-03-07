#!/bin/bash
# SFT model evaluation pipeline
# Runs: smoke test → Phase A (loglikelihood) → Phase B (GSM8K SFT) → Phase C (GSM8K Base)
set -e

SFT_CHECKPOINT="checkpoints_sft/sft_final.pt"
BASE_CHECKPOINT="checkpoints/portable_final.pt"
SFT_OUTDIR="results/sft_eval"
BASE_OUTDIR="results/base_eval"
mkdir -p "$SFT_OUTDIR"
mkdir -p "$BASE_OUTDIR"

if [ ! -f "$SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found at $SFT_CHECKPOINT"
    exit 1
fi

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "ERROR: Base checkpoint not found at $BASE_CHECKPOINT"
    exit 1
fi

echo "============================================================"
echo "SFT MODEL EVALUATION PIPELINE"
echo "SFT Checkpoint:  $SFT_CHECKPOINT"
echo "Base Checkpoint: $BASE_CHECKPOINT"
echo "SFT Output:      $SFT_OUTDIR/"
echo "Base Output:     $BASE_OUTDIR/"
echo "============================================================"

# ------------------------------------------------------------------
# SMOKE TEST (limit=5, verify SFT checkpoint loads)
# ------------------------------------------------------------------
echo ""
echo "[SMOKE TEST] Running 3 tasks with limit=5 on SFT checkpoint..."
python -m src.evaluation.run_eval \
    --checkpoint "$SFT_CHECKPOINT" \
    --tasks hellaswag gsm8k lambada_openai \
    --batch-size 1 \
    --limit 5 \
    --output "$SFT_OUTDIR/smoke_test.json"
echo "[SMOKE TEST] PASSED"

# ------------------------------------------------------------------
# PHASE A: Loglikelihood benchmarks on SFT (~45 min)
# ------------------------------------------------------------------
echo ""
echo "[PHASE A] Loglikelihood benchmarks on SFT (10 tasks)..."
python -m src.evaluation.run_eval \
    --checkpoint "$SFT_CHECKPOINT" \
    --tasks hellaswag arc_easy arc_challenge piqa winogrande boolq \
           mmlu_stem lambada_openai openbookqa sciq \
    --batch-size 4 \
    --output "$SFT_OUTDIR/loglikelihood_benchmarks.json"
echo "[PHASE A] COMPLETE"

# ------------------------------------------------------------------
# PHASE B: GSM8K generation on SFT (~3.5 hours)
# ------------------------------------------------------------------
echo ""
echo "[PHASE B] GSM8K generation on SFT checkpoint (1319 examples)..."
python -m src.evaluation.run_eval \
    --checkpoint "$SFT_CHECKPOINT" \
    --tasks gsm8k \
    --batch-size 1 \
    --output "$SFT_OUTDIR/gsm8k_results.json"
echo "[PHASE B] COMPLETE"

# ------------------------------------------------------------------
# PHASE C: GSM8K generation on Base (~3.5 hours)
# ------------------------------------------------------------------
echo ""
echo "[PHASE C] GSM8K generation on Base checkpoint (1319 examples)..."
python -m src.evaluation.run_eval \
    --checkpoint "$BASE_CHECKPOINT" \
    --tasks gsm8k \
    --batch-size 1 \
    --output "$BASE_OUTDIR/gsm8k_results.json"
echo "[PHASE C] COMPLETE"

echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE"
echo "SFT results in: $SFT_OUTDIR/"
ls -la "$SFT_OUTDIR/"
echo ""
echo "Base results in: $BASE_OUTDIR/"
ls -la "$BASE_OUTDIR/"
echo "============================================================"
