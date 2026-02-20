#!/bin/bash
# Setup script for RunPod A100 80GB environment
set -e

echo "=== PolyGLU RunPod Setup ==="

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Login to HuggingFace (requires HF_TOKEN in .env or environment)
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace login successful"
else
    echo "WARNING: HF_TOKEN not found. Set it in .env or environment for gated dataset access."
fi

# Login to WandB (requires WANDB_API_KEY in environment)
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "WandB login successful"
else
    echo "WARNING: WANDB_API_KEY not found. Set it for experiment tracking."
fi

# Create data directories
mkdir -p data/tokenized/{math,stem,code}
mkdir -p checkpoints
mkdir -p checkpoints_sft
mkdir -p results
mkdir -p figures

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. bash scripts/tokenize_data.sh    # Tokenize datasets"
echo "  2. bash scripts/run_train.sh        # Start pre-training"
