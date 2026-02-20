"""Neurotransmitter maps: extract and visualize activation preferences.

For each layer, extracts argmax of learned alpha vectors to show which
activation function each neuron "prefers". Produces heatmaps and
stacked bar charts showing the distribution of activation types across layers.

Activation mapping:
    0: ReLU (Glutamate) — hard threshold
    1: Tanh (GABA) — symmetric compression
    2: SiLU (Dopamine) — self-gated
    3: GELU (Acetylcholine) — probabilistic gate
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.model.architecture import PolychromaticLM

ACTIVATION_NAMES = ["ReLU", "Tanh", "SiLU", "GELU"]
ACTIVATION_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
NT_NAMES = ["Glutamate", "GABA", "Dopamine", "Acetylcholine"]


def extract_alpha_preferences(model: PolychromaticLM) -> dict:
    """Extract per-neuron activation preferences from all layers.

    Returns:
        {
            "alphas": list of (d_ff, K) arrays (raw alpha per layer),
            "argmax": list of (d_ff,) arrays (preferred activation per neuron),
            "softmax": list of (d_ff, K) arrays (probability distribution per neuron),
        }
    """
    alphas = []
    argmax_maps = []
    softmax_maps = []

    for block in model.model_core:
        alpha = block.polyglu.alpha.detach().float().cpu()  # [d_ff, K]
        probs = torch.softmax(alpha, dim=-1)
        preferred = torch.argmax(alpha, dim=-1)

        alphas.append(alpha.numpy())
        argmax_maps.append(preferred.numpy())
        softmax_maps.append(probs.numpy())

    return {
        "alphas": alphas,
        "argmax": argmax_maps,
        "softmax": softmax_maps,
    }


def plot_neurotransmitter_heatmap(preferences: dict, output_path: str):
    """Plot heatmap of preferred activation per neuron per layer.

    Rows = layers, columns = neurons (d_ff). Color = preferred activation.
    """
    argmax_maps = preferences["argmax"]
    n_layers = len(argmax_maps)
    d_ff = argmax_maps[0].shape[0]

    # Build matrix: (n_layers, d_ff)
    matrix = np.stack(argmax_maps)

    fig, ax = plt.subplots(figsize=(20, 8))

    # Custom colormap for 4 activations
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(ACTIVATION_COLORS)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")

    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Layer")
    ax.set_title("Neurotransmitter Map: Preferred Activation per Neuron")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(i) for i in range(n_layers)])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACTIVATION_COLORS[i], label=f"{ACTIVATION_NAMES[i]} ({NT_NAMES[i]})")
        for i in range(4)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved neurotransmitter heatmap to {output_path}")


def plot_layer_distribution(preferences: dict, output_path: str):
    """Plot stacked bar chart of activation distribution per layer."""
    argmax_maps = preferences["argmax"]
    n_layers = len(argmax_maps)

    # Count per layer
    counts = np.zeros((n_layers, 4))
    for i, am in enumerate(argmax_maps):
        for k in range(4):
            counts[i, k] = np.sum(am == k)

    # Normalize to percentages
    totals = counts.sum(axis=1, keepdims=True)
    percentages = counts / totals * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_layers)
    bottom = np.zeros(n_layers)

    for k in range(4):
        ax.bar(
            x, percentages[:, k], bottom=bottom,
            color=ACTIVATION_COLORS[k],
            label=f"{ACTIVATION_NAMES[k]} ({NT_NAMES[k]})",
        )
        bottom += percentages[:, k]

    ax.set_xlabel("Layer")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Activation Distribution Across Layers")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer distribution chart to {output_path}")
