"""Evaluation entry point using lm-evaluation-harness.

Usage:
    python -m src.evaluation.run_eval \
        --checkpoint checkpoints/portable_final.pt \
        --tasks gsm8k minerva_math mmlu_stem \
        --output results/pretrain_eval.json

    python -m src.evaluation.run_eval \
        --checkpoint checkpoints_sft/sft_final.pt \
        --tasks gsm8k minerva_math \
        --output results/sft_eval.json
"""

import argparse
import json
import os

import lm_eval

from src.model.config import ModelConfig
from src.evaluation.model_wrapper import PolychromaticLMWrapper


def main():
    parser = argparse.ArgumentParser(description="PolychromaticLM evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "minerva_math", "mmlu_stem"],
                        help="Benchmark tasks to run")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (for testing)")
    parser.add_argument("--output", type=str, default="results/eval_results.json", help="Output path")
    parser.add_argument("--config", type=str, default=None, help="Model config YAML (uses defaults if not provided)")
    args = parser.parse_args()

    # Load model config
    if args.config:
        model_config = ModelConfig.from_yaml(args.config)
    else:
        model_config = ModelConfig()

    # Create model wrapper
    print(f"Loading model from {args.checkpoint}...")
    lm = PolychromaticLMWrapper(
        checkpoint_path=args.checkpoint,
        model_config=model_config,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Run evaluation
    print(f"Running evaluation on tasks: {args.tasks}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
