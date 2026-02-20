"""Dynamic routing analysis: how beta*f(h) shifts routing for different inputs.

Uses forward hooks on gate_net to capture the dynamic modulation signal
for different input types. Compares routing patterns across input categories
(arithmetic, algebra, geometry) using hardcoded English example texts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from src.model.architecture import PolychromaticLM
from transformers import AutoTokenizer

ACTIVATION_NAMES = ["ReLU", "Tanh", "SiLU", "GELU"]
ACTIVATION_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

# Hardcoded examples for each category
CATEGORY_EXAMPLES = {
    "arithmetic": [
        "Calculate 347 + 892 step by step. First, add the ones digits: 7 + 2 = 9.",
        "What is 15 × 23? We can use the distributive property: 15 × 20 + 15 × 3.",
        "Divide 1,248 by 16 using long division.",
    ],
    "algebra": [
        "Solve for x: 3x + 7 = 22. Subtract 7 from both sides to get 3x = 15.",
        "Factor the quadratic expression x² - 5x + 6 = (x - 2)(x - 3).",
        "Find the roots of 2x² + 3x - 5 = 0 using the quadratic formula.",
    ],
    "geometry": [
        "The area of a circle with radius r = 5 is A = πr² = 25π ≈ 78.54 square units.",
        "In a right triangle with legs a = 3 and b = 4, the hypotenuse c = √(9 + 16) = 5.",
        "Calculate the volume of a sphere with radius 6: V = (4/3)πr³ = 288π.",
    ],
}


def capture_gate_activations(
    model: PolychromaticLM,
    input_ids: torch.Tensor,
) -> list[np.ndarray]:
    """Run forward pass and capture gate_net outputs via hooks.

    Args:
        model: PolychromaticLM instance.
        input_ids: (1, seq_len) tensor.

    Returns:
        List of (K,) arrays — gate_net output per layer (mean-pooled over batch).
    """
    gate_outputs = []

    hooks = []
    for block in model.model_core:
        def hook_fn(module, input, output, _list=gate_outputs):
            # gate_net output shape: (bs, K) — take mean over batch
            _list.append(output.detach().float().cpu().mean(dim=0).numpy())

        h = block.polyglu.gate_net.register_forward_hook(hook_fn)
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return gate_outputs


def analyze_dynamic_routing(
    model: PolychromaticLM,
    device: torch.device,
    tokenizer_name: str = "Qwen/Qwen3-0.6B-Base",
) -> dict:
    """Analyze how dynamic routing differs across input categories.

    Returns:
        Dict mapping category → list of per-layer gate activations (averaged over examples).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model.eval()

    results = {}
    for category, texts in CATEGORY_EXAMPLES.items():
        category_activations = []

        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            # Pad/truncate to reasonable length
            if len(token_ids) > 512:
                token_ids = token_ids[:512]
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

            gate_outputs = capture_gate_activations(model, input_tensor)
            category_activations.append(gate_outputs)

        # Average across examples: list of (K,) per layer
        n_layers = len(category_activations[0])
        avg_activations = []
        for layer_idx in range(n_layers):
            layer_gates = [ca[layer_idx] for ca in category_activations]
            avg_activations.append(np.mean(layer_gates, axis=0))

        results[category] = avg_activations

    return results


def plot_dynamic_routing_comparison(routing_data: dict, output_path: str):
    """Plot how gate_net outputs differ across categories for each layer."""
    categories = list(routing_data.keys())
    n_layers = len(routing_data[categories[0]])
    n_cats = len(categories)

    fig, axes = plt.subplots(4, 7, figsize=(24, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    for layer_idx in range(min(n_layers, 28)):
        ax = axes[layer_idx]
        x = np.arange(4)  # K=4 activations
        width = 0.25

        for i, cat in enumerate(categories):
            values = routing_data[cat][layer_idx]
            ax.bar(x + i * width, values, width, label=cat, alpha=0.8)

        ax.set_title(f"Layer {layer_idx}", fontsize=8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(ACTIVATION_NAMES, fontsize=6, rotation=45)

        if layer_idx == 0:
            ax.legend(fontsize=6)

    plt.suptitle("Dynamic Routing: gate_net(h) by Input Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dynamic routing comparison to {output_path}")


def plot_routing_shift_summary(routing_data: dict, output_path: str):
    """Plot summary of how much dynamic routing shifts across categories.

    For each layer, computes the max absolute difference in gate_net output
    between any two categories, per activation.
    """
    categories = list(routing_data.keys())
    n_layers = len(routing_data[categories[0]])

    # Compute max shift per layer per activation
    max_shifts = np.zeros((n_layers, 4))
    for layer_idx in range(n_layers):
        for k in range(4):
            values = [routing_data[cat][layer_idx][k] for cat in categories]
            max_shifts[layer_idx, k] = max(values) - min(values)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_layers)

    for k in range(4):
        ax.plot(x, max_shifts[:, k], marker="o", markersize=3,
                color=ACTIVATION_COLORS[k], label=ACTIVATION_NAMES[k])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Gate Shift (across categories)")
    ax.set_title("Dynamic Routing Sensitivity by Layer and Activation")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved routing shift summary to {output_path}")
