#!/bin/bash
# Tokenize all 3 data sources (math, STEM, code) before training.
# Runs on CPU — no GPU needed.
set -e

echo "=== Tokenizing Pre-Training Data ==="

# Load HF_TOKEN
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

python -m src.data.tokenize_dataset --all --output-base data/tokenized --hf-token "$HF_TOKEN"

echo "=== Tokenization Complete ==="
echo "Check data/tokenized/{math,stem,code}/ for binary chunks"
