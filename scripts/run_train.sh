#!/bin/bash
# Launch pre-training with DeepSpeed on single A100 80GB.
set -e

echo "=== Starting PolychromaticLM Pre-Training ==="

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

deepspeed --num_gpus=1 --module src.training.train \
    --config configs/train_config.yaml

echo "=== Pre-Training Complete ==="
