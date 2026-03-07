"""Generate all training visualization plots for the pre-training report.

Reads training_metrics.csv and final_dynamic_entropy.json to produce
paper-quality figures. All outputs saved to paper_reporting/figures/.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_CSV = os.path.join(BASE_DIR, "paper_reporting", "training_metrics.csv")
ENTROPY_JSON = os.path.join(BASE_DIR, "paper_reporting", "final_dynamic_entropy.json")
FIGURES_DIR = os.path.join(BASE_DIR, "paper_reporting", "figures")

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

ACTIVATION_NAMES = ["ReLU", "Tanh", "SiLU", "GELU"]
ACTIVATION_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def load_metrics() -> dict:
    steps, losses, lrs, taus, tps, entropies = [], [], [], [], [], []
    with open(METRICS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))
            taus.append(float(row["tau"]))
            tps.append(int(row["tokens_per_sec"]))
            entropies.append(float(row["static_entropy"]))
    return {
        "steps": np.array(steps),
        "losses": np.array(losses),
        "lrs": np.array(lrs),
        "taus": np.array(taus),
        "tps": np.array(tps),
        "entropies": np.array(entropies),
    }


def moving_average(data: np.ndarray, window: int = 50) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_loss_curve(m: dict):
    """Full loss curve with smoothed trendline and annotations."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw data (light)
    ax.plot(m["steps"], m["losses"], alpha=0.15, color="#3498db", linewidth=0.5, label="_raw")

    # Smoothed (50-step window)
    window = 50
    smooth = moving_average(m["losses"], window)
    smooth_steps = m["steps"][window - 1:]
    ax.plot(smooth_steps, smooth, color="#2c3e50", linewidth=1.5, label="Loss (50-step MA)")

    # Annotations
    annotations = [
        (2000, "Warmup ends\n(step 2,000)", "above"),
        (10000, "Weight decay fix\n& resume (step 10,000)", "above"),
        (15625, "Data mix annealing\nstarts (step 15,625)", "above"),
    ]
    for step, text, pos in annotations:
        ax.axvline(x=step, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=0.8)
        y_pos = ax.get_ylim()[1] * 0.85 if pos == "above" else ax.get_ylim()[0]
        ax.annotate(text, xy=(step, smooth[np.searchsorted(smooth_steps, step)] if step >= smooth_steps[0] else m["losses"][np.searchsorted(m["steps"], step)]),
                    xytext=(step + 400, 6.5 if step < 5000 else 2.8),
                    fontsize=8, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("PolychromaticLM Pre-Training Loss Curve (~10B Tokens)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)

    # Token axis on top
    ax2 = ax.twiny()
    ax2.set_xlim(0, 20000)
    token_ticks = [0, 5000, 10000, 15000, 19531]
    token_labels = [f"{t * 524288 / 1e9:.1f}B" for t in token_ticks]
    ax2.set_xticks(token_ticks)
    ax2.set_xticklabels(token_labels)
    ax2.set_xlabel("Tokens Processed")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "loss_curve.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved loss_curve.png")


def plot_lr_schedule(m: dict):
    """Learning rate schedule: cosine decay with warmup."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m["steps"], m["lrs"] * 1e4, color="#e67e22", linewidth=1.2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate (×10⁻⁴)")
    ax.set_title("Learning Rate Schedule (Cosine Decay with Warmup)")
    ax.axvline(x=2000, color="#95a5a6", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.annotate("Warmup\n(2,000 steps)", xy=(2000, 0.5), fontsize=9, color="#7f8c8d")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 20000)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "lr_schedule.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved lr_schedule.png")


def plot_tau_annealing(m: dict):
    """Gumbel-Softmax temperature annealing curve."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m["steps"], m["taus"], color="#8e44ad", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Temperature (τ)")
    ax.set_title("Gumbel-Softmax Temperature Annealing (τ: 1.0 → 0.1)")
    ax.axhline(y=0.1, color="#95a5a6", linestyle=":", alpha=0.6)
    ax.annotate("τ_min = 0.1", xy=(18000, 0.12), fontsize=9, color="#7f8c8d")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "tau_annealing.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved tau_annealing.png")


def plot_throughput(m: dict):
    """Tokens/sec throughput over training."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Smooth throughput
    window = 20
    smooth = moving_average(m["tps"].astype(float), window)
    smooth_steps = m["steps"][window - 1:]

    ax.plot(smooth_steps, smooth, color="#27ae60", linewidth=1.2)
    ax.fill_between(smooth_steps, smooth * 0.98, smooth * 1.02, alpha=0.1, color="#27ae60")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Training Throughput")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 20000)

    mean_tps = m["tps"][m["tps"] > 5000].mean()  # Filter startup anomalies
    ax.axhline(y=mean_tps, color="#c0392b", linestyle="--", alpha=0.5)
    ax.annotate(f"Mean: {mean_tps:,.0f} tok/s", xy=(15000, mean_tps + 100), fontsize=9, color="#c0392b")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "throughput.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved throughput.png")


def plot_dynamic_routing_entropy_heatmap():
    """Per-layer dynamic routing entropy at final step as a bar chart."""
    with open(ENTROPY_JSON) as f:
        entropy = json.load(f)

    layers = list(range(28))
    values = [entropy.get(f"layer_{i}", 0.0) for i in layers]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#e74c3c" if v > 0.001 else "#f39c12" if v > 0.0001 else "#3498db" for v in values]
    bars = ax.bar(layers, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Dynamic Routing Entropy")
    ax.set_title("Per-Layer Dynamic Routing Entropy at Final Step (19,530)")
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 0.05)
    ax.grid(alpha=0.3, axis="y")

    # Max entropy reference
    ax.axhline(y=1.3863, color="#95a5a6", linestyle=":", alpha=0.4)

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="> 10⁻³ (partially specialized)"),
        Patch(facecolor="#f39c12", label="10⁻⁴ – 10⁻³ (mostly specialized)"),
        Patch(facecolor="#3498db", label="< 10⁻⁴ (near-deterministic)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Annotate notable layers
    for i, v in enumerate(values):
        if v > 0.0001:
            ax.annotate(f"{v:.2e}", xy=(i, v), xytext=(i, v * 3),
                        fontsize=7, ha="center", color="#2c3e50",
                        arrowprops=dict(arrowstyle="-", color="#95a5a6", lw=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dynamic_routing_entropy_final.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved dynamic_routing_entropy_final.png")


def plot_combined_training_dynamics(m: dict):
    """Loss + LR + Tau on a single figure with dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Loss (left y-axis)
    window = 50
    smooth_loss = moving_average(m["losses"], window)
    smooth_steps = m["steps"][window - 1:]
    ln1 = ax1.plot(smooth_steps, smooth_loss, color="#2c3e50", linewidth=1.5, label="Loss (50-step MA)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax1.set_xlim(0, 20000)

    # LR and Tau (right y-axis)
    ax2 = ax1.twinx()
    ln2 = ax2.plot(m["steps"], m["lrs"] * 1e4, color="#e67e22", linewidth=1.0, alpha=0.8, label="LR (×10⁻⁴)")
    ln3 = ax2.plot(m["steps"], m["taus"], color="#8e44ad", linewidth=1.0, alpha=0.8, label="τ")
    ax2.set_ylabel("LR (×10⁻⁴) / τ", color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")

    # Combined legend
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")

    ax1.set_title("Training Dynamics: Loss, Learning Rate, and Temperature")
    ax1.grid(alpha=0.2)

    # Key event markers
    for step in [2000, 10000, 15625]:
        ax1.axvline(x=step, color="#bdc3c7", linestyle="--", alpha=0.5, linewidth=0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "combined_training_dynamics.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved combined_training_dynamics.png")


def plot_dynamic_entropy_comparison():
    """Compare dynamic routing entropy at step 10,030 vs final step 19,530."""
    # Step 10,030 data (from red0001_report.md)
    early_entropy = {
        0: 0.000003, 1: 0.000002, 2: 0.000002, 3: 0.000055, 4: 0.000139,
        5: 0.000087, 6: 0.000029, 7: 0.000004, 8: 0.000014, 9: 0.183973,
        10: 0.000121, 11: 0.000727, 12: 0.000566, 13: 0.002424, 14: 0.000502,
        15: 0.001079, 16: 0.032677, 17: 0.000691, 18: 0.000145, 19: 0.000071,
        20: 0.000125, 21: 0.000208, 22: 0.000064, 23: 0.000030, 24: 0.000095,
        25: 0.000124, 26: 0.000077, 27: 0.000401,
    }

    with open(ENTROPY_JSON) as f:
        final_data = json.load(f)
    final_entropy = {i: final_data.get(f"layer_{i}", 0.0) for i in range(28)}

    layers = list(range(28))
    early = [early_entropy[i] for i in layers]
    final = [final_entropy[i] for i in layers]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(28)
    width = 0.35

    bars1 = ax.bar(x - width / 2, early, width, label="Step 10,030 (τ=0.54)", color="#e74c3c", alpha=0.7)
    bars2 = ax.bar(x + width / 2, final, width, label="Step 19,530 (τ=0.10)", color="#3498db", alpha=0.7)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Dynamic Routing Entropy (log scale)")
    ax.set_title("Dynamic Routing Entropy Evolution: Step 10,030 vs Final Step 19,530")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 0.5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dynamic_entropy_evolution.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved dynamic_entropy_evolution.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    m = load_metrics()
    print(f"Loaded {len(m['steps'])} data points")

    print("Generating plots...")
    plot_loss_curve(m)
    plot_lr_schedule(m)
    plot_tau_annealing(m)
    plot_throughput(m)
    plot_dynamic_routing_entropy_heatmap()
    plot_combined_training_dynamics(m)
    plot_dynamic_entropy_comparison()
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
