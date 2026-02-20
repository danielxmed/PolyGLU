"""Entry point for all interpretability analyses.

Runs neurotransmitter maps, routing entropy, and dynamic routing analysis.
Saves all figures to the output directory.

Usage:
    python -m src.interpretability.run_analysis \
        --checkpoint checkpoints/portable_final.pt \
        --output figures/
"""

import argparse
import json
import os

import torch

from src.model.architecture import PolychromaticLM
from src.model.config import ModelConfig
from src.interpretability.neurotransmitter_maps import (
    extract_alpha_preferences,
    plot_neurotransmitter_heatmap,
    plot_layer_distribution,
)
from src.interpretability.routing_entropy import (
    compute_entropy,
    plot_entropy_per_layer,
    plot_entropy_histogram,
)
from src.interpretability.dynamic_routing import (
    analyze_dynamic_routing,
    plot_dynamic_routing_comparison,
    plot_routing_shift_summary,
)


def load_model(checkpoint_path: str, config: ModelConfig, device: str) -> PolychromaticLM:
    """Load model from portable checkpoint (no Flash Attention for analysis)."""
    model = PolychromaticLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        eps=config.eps,
        head_dim=config.head_dim,
        seq_length=config.seq_length,
        n_activations=config.n_activations,
        n_q_heads=config.n_q_heads,
        n_kv_heads=config.n_kv_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    tau = checkpoint.get("tau", 0.1)
    for block in model.model_core:
        block.polyglu.tau = tau

    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path} (step {checkpoint.get('step', '?')}, tau={tau:.3f})")
    return model


def main():
    parser = argparse.ArgumentParser(description="PolychromaticLM interpretability analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-dynamic", action="store_true",
                        help="Skip dynamic routing analysis (requires GPU memory)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    config = ModelConfig.from_yaml(args.config) if args.config else ModelConfig()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, config, device)

    # --- 1. Neurotransmitter Maps ---
    print("\n=== Neurotransmitter Maps ===")
    preferences = extract_alpha_preferences(model)
    plot_neurotransmitter_heatmap(preferences, os.path.join(args.output, "neurotransmitter_heatmap.png"))
    plot_layer_distribution(preferences, os.path.join(args.output, "layer_distribution.png"))

    # --- 2. Routing Entropy ---
    print("\n=== Routing Entropy ===")
    entropy_data = compute_entropy(model)
    plot_entropy_per_layer(entropy_data, os.path.join(args.output, "entropy_per_layer.png"))
    plot_entropy_histogram(entropy_data, os.path.join(args.output, "entropy_histogram.png"))

    # Save entropy stats as JSON
    stats = {k: v for k, v in entropy_data.items() if k != "per_neuron"}
    with open(os.path.join(args.output, "entropy_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Aggregate entropy: {entropy_data['aggregate_mean']:.4f} "
          f"(normalized: {entropy_data['normalized_mean']:.4f})")

    # --- 3. Dynamic Routing ---
    if not args.skip_dynamic:
        print("\n=== Dynamic Routing Analysis ===")
        model_bf16 = model.to(dtype=torch.bfloat16)
        routing_data = analyze_dynamic_routing(model_bf16, device)
        plot_dynamic_routing_comparison(routing_data, os.path.join(args.output, "dynamic_routing_comparison.png"))
        plot_routing_shift_summary(routing_data, os.path.join(args.output, "routing_shift_summary.png"))
    else:
        print("\nSkipping dynamic routing analysis (--skip-dynamic)")

    print(f"\nAll figures saved to {args.output}")


if __name__ == "__main__":
    main()
