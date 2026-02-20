"""Routing entropy analysis for PolychromaticLM.

Computes per-layer and aggregate entropy of softmax(alpha) to measure
activation diversity.

High entropy = diverse activation usage (good, leverages PolyGLU capacity).
Low entropy = collapsed to single activation (bad, equivalent to SwiGLU).

Max entropy for K=4: log(4) ≈ 1.386 (uniform distribution).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model.architecture import PolychromaticLM


def compute_entropy(model: PolychromaticLM) -> dict:
    """Compute routing entropy from learned alpha parameters.

    Returns:
        {
            "per_layer_mean": list of floats (mean entropy per layer),
            "per_layer_std": list of floats (std entropy per layer),
            "per_neuron": list of (d_ff,) arrays (per-neuron entropy per layer),
            "aggregate_mean": float,
            "aggregate_std": float,
            "max_entropy": float (log(K)),
            "normalized_mean": float (aggregate_mean / max_entropy),
        }
    """
    max_entropy = np.log(4)  # K=4
    per_layer_mean = []
    per_layer_std = []
    per_neuron = []
    all_entropies = []

    for block in model.model_core:
        alpha = block.polyglu.alpha.detach().float().cpu()  # [d_ff, K]
        probs = torch.softmax(alpha, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # [d_ff]

        entropy_np = entropy.numpy()
        per_neuron.append(entropy_np)
        per_layer_mean.append(float(entropy_np.mean()))
        per_layer_std.append(float(entropy_np.std()))
        all_entropies.extend(entropy_np.tolist())

    aggregate_mean = float(np.mean(all_entropies))
    aggregate_std = float(np.std(all_entropies))

    return {
        "per_layer_mean": per_layer_mean,
        "per_layer_std": per_layer_std,
        "per_neuron": per_neuron,
        "aggregate_mean": aggregate_mean,
        "aggregate_std": aggregate_std,
        "max_entropy": float(max_entropy),
        "normalized_mean": aggregate_mean / max_entropy,
    }


def plot_entropy_per_layer(entropy_data: dict, output_path: str):
    """Plot mean entropy per layer with error bars."""
    n_layers = len(entropy_data["per_layer_mean"])
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        x,
        entropy_data["per_layer_mean"],
        yerr=entropy_data["per_layer_std"],
        color="#3498db",
        alpha=0.8,
        capsize=2,
    )
    ax.axhline(
        y=entropy_data["max_entropy"],
        color="red",
        linestyle="--",
        label=f"Max entropy (log(4) = {entropy_data['max_entropy']:.3f})",
    )
    ax.axhline(
        y=entropy_data["aggregate_mean"],
        color="green",
        linestyle="--",
        label=f"Aggregate mean = {entropy_data['aggregate_mean']:.3f}",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Routing Entropy per Layer")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved entropy per layer plot to {output_path}")


def plot_entropy_histogram(entropy_data: dict, output_path: str):
    """Plot histogram of per-neuron entropy across all layers."""
    all_entropies = np.concatenate(entropy_data["per_neuron"])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(all_entropies, bins=100, color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(
        x=entropy_data["max_entropy"],
        color="red",
        linestyle="--",
        label=f"Max entropy = {entropy_data['max_entropy']:.3f}",
    )
    ax.axvline(
        x=entropy_data["aggregate_mean"],
        color="blue",
        linestyle="--",
        label=f"Mean = {entropy_data['aggregate_mean']:.3f}",
    )

    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Count (neurons)")
    ax.set_title(f"Distribution of Per-Neuron Routing Entropy (N={len(all_entropies):,})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved entropy histogram to {output_path}")
