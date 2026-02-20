#!/bin/bash
# Launch SFT with DeepSpeed on single A100 80GB.
set -e

echo "=== Starting PolychromaticLM SFT ==="

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

deepspeed --num_gpus=1 -m src.sft.sft \
    --config configs/sft_config.yaml

echo "=== SFT Complete ==="
