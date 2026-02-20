#!/bin/bash
# Run evaluation on pre-trained and SFT checkpoints.
set -e

echo "=== PolychromaticLM Evaluation ==="

# Pre-trained model evaluation
if [ -f checkpoints/portable_final.pt ]; then
    echo "--- Evaluating pre-trained model ---"
    python -m src.evaluation.run_eval \
        --checkpoint checkpoints/portable_final.pt \
        --tasks gsm8k minerva_math mmlu_stem \
        --output results/pretrain_eval.json
fi

# SFT model evaluation
if [ -f checkpoints_sft/sft_final.pt ]; then
    echo "--- Evaluating SFT model ---"
    python -m src.evaluation.run_eval \
        --checkpoint checkpoints_sft/sft_final.pt \
        --tasks gsm8k minerva_math mmlu_stem \
        --output results/sft_eval.json
fi

# Interpretability analysis
if [ -f checkpoints/portable_final.pt ]; then
    echo "--- Running interpretability analysis ---"
    python -m src.interpretability.run_analysis \
        --checkpoint checkpoints/portable_final.pt \
        --output figures/
fi

echo "=== Evaluation Complete ==="
echo "Results in results/ and figures/"
