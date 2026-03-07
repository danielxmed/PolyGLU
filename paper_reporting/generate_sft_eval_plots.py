"""Generate SFT evaluation visualization plots.

Compares base pre-trained model vs SFT model across benchmarks.
All outputs saved to paper_reporting/figures/.

Usage:
    python paper_reporting/generate_sft_eval_plots.py
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_RESULTS_DIR = os.path.join(BASE_DIR, "results", "base_eval")
SFT_RESULTS_DIR = os.path.join(BASE_DIR, "results", "sft_eval")
FIGURES_DIR = os.path.join(BASE_DIR, "paper_reporting", "figures")
SFT_METRICS_CSV = os.path.join(BASE_DIR, "paper_reporting", "sft_training_metrics.csv")

# Style (matches generate_eval_plots.py)
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

# Colors
POLY_COLOR = "#2ecc71"      # Green for base PolychromaticLM
SFT_COLOR = "#e74c3c"       # Red for SFT model
QWEN_COLOR = "#3498db"      # Blue for Qwen3 reference
RANDOM_COLOR = "#bdc3c7"    # Gray for random baseline
LR_COLOR = "#9b59b6"        # Purple for LR overlay

# Reused constants from generate_eval_plots.py
QWEN3_BASELINES = {
    "hellaswag": 0.411,
    "arc_easy": 0.656,
    "arc_challenge": 0.339,
    "piqa": 0.700,
    "winogrande": 0.585,
    "boolq": 0.697,
}

RANDOM_BASELINES = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "piqa": 0.50,
    "winogrande": 0.50,
    "boolq": 0.50,
    "openbookqa": 0.25,
    "sciq": 0.25,
    "mmlu_stem": 0.25,
    "lambada_openai": 0.0,
}

METRIC_MAP = {
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "sciq": "acc_norm,none",
    "mmlu_stem": "acc,none",
    "lambada_openai": "acc,none",
}

DISPLAY_NAMES = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "sciq": "SciQ",
    "mmlu_stem": "MMLU-STEM",
    "lambada_openai": "LAMBADA",
    "gsm8k": "GSM8K",
}


def extract_scores(results: dict) -> dict:
    """Extract scores from lm-eval results JSON."""
    scores = {}
    for task_name, metric_key in METRIC_MAP.items():
        task_results = results.get("results", {}).get(task_name, {})
        if metric_key in task_results:
            scores[task_name] = task_results[metric_key]
        elif task_name == "mmlu_stem":
            for key, val in task_results.items():
                if "acc" in key and "stderr" not in key and isinstance(val, (int, float)):
                    scores[task_name] = val
                    break
    return scores


def extract_gsm8k_score(results: dict) -> float | None:
    """Extract GSM8K exact-match score."""
    gsm8k = results.get("results", {}).get("gsm8k", {})
    return gsm8k.get("exact_match,strict-match", None)


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_sft_training_metrics() -> list[dict]:
    rows = []
    with open(SFT_METRICS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step": int(row["step"]),
                "loss": float(row["loss"]),
                "lr": float(row["lr"]),
                "tokens_per_sec": int(row["tokens_per_sec"]),
                "mem_gb": float(row["mem_gb"]),
                "entropy": float(row["entropy"]),
            })
    return rows


def smooth(values: list[float], window: int = 50) -> list[float]:
    """Simple moving average smoothing."""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def plot_sft_loss_curve(metrics: list[dict]):
    """SFT training loss with smoothed trendline + LR overlay."""
    steps = [r["step"] for r in metrics]
    losses = [r["loss"] for r in metrics]
    lrs = [r["lr"] for r in metrics]
    smoothed = smooth(losses, window=100)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Loss (left axis)
    ax1.scatter(steps, losses, alpha=0.15, s=3, color=SFT_COLOR, label="Loss (raw)")
    ax1.plot(steps, smoothed, color=SFT_COLOR, linewidth=2, label="Loss (smoothed)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss", color=SFT_COLOR)
    ax1.tick_params(axis="y", labelcolor=SFT_COLOR)
    ax1.set_ylim(0.5, 2.0)
    ax1.grid(alpha=0.3)

    # LR (right axis)
    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color=LR_COLOR, linewidth=1.5, alpha=0.7, label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color=LR_COLOR)
    ax2.tick_params(axis="y", labelcolor=LR_COLOR)

    # Title and legend
    ax1.set_title("SFT Training Loss Curve — PolychromaticLM on Nemotron-Math-v2")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Annotations
    ax1.annotate(f"Initial: {losses[0]:.3f}", xy=(steps[0], losses[0]),
                 xytext=(steps[0] + 500, losses[0] + 0.05),
                 fontsize=9, color=SFT_COLOR, fontweight="bold")
    ax1.annotate(f"Final: {losses[-1]:.3f}", xy=(steps[-1], losses[-1]),
                 xytext=(steps[-1] - 3000, losses[-1] + 0.15),
                 fontsize=9, color=SFT_COLOR, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sft_loss_curve.png"))
    plt.close()
    print("Saved sft_loss_curve.png")


def plot_base_vs_sft_benchmarks(base_scores: dict, sft_scores: dict):
    """Grouped bars: base vs SFT vs random on all loglikelihood tasks."""
    # Use tasks available in both
    tasks = [t for t in METRIC_MAP if t in base_scores and t in sft_scores]
    if not tasks:
        print("No shared tasks for base vs SFT comparison, skipping.")
        return

    x = np.arange(len(tasks))
    width = 0.22

    base_vals = [base_scores[t] for t in tasks]
    sft_vals = [sft_scores[t] for t in tasks]
    random_vals = [RANDOM_BASELINES.get(t, 0) for t in tasks]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, base_vals, width, label="Base (Pre-trained)", color=POLY_COLOR, edgecolor="white")
    bars2 = ax.bar(x, sft_vals, width, label="SFT (Nemotron-Math-v2)", color=SFT_COLOR, edgecolor="white")
    bars3 = ax.bar(x + width, random_vals, width, label="Random", color=RANDOM_COLOR, edgecolor="white")

    ax.set_ylabel("Accuracy")
    ax.set_title("Base vs SFT — Loglikelihood Benchmarks")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[t] for t in tasks], rotation=20, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sft_base_vs_sft_benchmarks.png"))
    plt.close()
    print("Saved sft_base_vs_sft_benchmarks.png")


def plot_gsm8k_comparison(base_gsm8k: float | None, sft_gsm8k: float | None):
    """GSM8K exact-match: base vs SFT (+ Qwen3 reference if available)."""
    labels = []
    scores = []
    colors = []

    if base_gsm8k is not None:
        labels.append("Base\n(Pre-trained)")
        scores.append(base_gsm8k)
        colors.append(POLY_COLOR)

    if sft_gsm8k is not None:
        labels.append("SFT\n(Nemotron-Math-v2)")
        scores.append(sft_gsm8k)
        colors.append(SFT_COLOR)

    # Qwen3-0.6B reference (published GSM8K score)
    labels.append("Qwen3-0.6B\n(reference)")
    scores.append(0.621)  # Published GSM8K score for Qwen3-0.6B
    colors.append(QWEN_COLOR)

    if len(scores) < 2:
        print("Insufficient data for GSM8K comparison, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, scores, color=colors, edgecolor="white", width=0.5)

    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("GSM8K Performance — Base vs SFT vs Reference")
    ax.set_ylim(0, max(scores) * 1.3 if max(scores) > 0 else 0.1)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sft_gsm8k_comparison.png"))
    plt.close()
    print("Saved sft_gsm8k_comparison.png")


def plot_delta_chart(base_scores: dict, sft_scores: dict):
    """Horizontal bar chart of per-benchmark delta (green=improved, red=regressed)."""
    tasks = [t for t in METRIC_MAP if t in base_scores and t in sft_scores]
    if not tasks:
        print("No shared tasks for delta chart, skipping.")
        return

    deltas = [(sft_scores[t] - base_scores[t]) for t in tasks]

    # Sort by delta
    sorted_pairs = sorted(zip(tasks, deltas), key=lambda x: x[1])
    tasks_sorted = [p[0] for p in sorted_pairs]
    deltas_sorted = [p[1] for p in sorted_pairs]

    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas_sorted]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(tasks_sorted))
    bars = ax.barh(y, deltas_sorted, color=colors, edgecolor="white", height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES[t] for t in tasks_sorted])
    ax.set_xlabel("Accuracy Delta (SFT - Base)")
    ax.set_title("SFT Impact — Per-Benchmark Change")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, deltas_sorted):
        offset = 0.003 if val >= 0 else -0.003
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1%}", va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sft_delta_chart.png"))
    plt.close()
    print("Saved sft_delta_chart.png")


def plot_training_dynamics(metrics: list[dict]):
    """Combined loss + LR dual-axis plot with training phases annotated."""
    steps = [r["step"] for r in metrics]
    losses = [r["loss"] for r in metrics]
    lrs = [r["lr"] for r in metrics]
    tps = [r["tokens_per_sec"] for r in metrics]
    smoothed_loss = smooth(losses, window=100)
    smoothed_tps = smooth(tps, window=50)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top: Loss + LR
    ax1 = axes[0]
    ax1.plot(steps, smoothed_loss, color=SFT_COLOR, linewidth=2, label="Loss (smoothed)")
    ax1.fill_between(steps, losses, alpha=0.1, color=SFT_COLOR)
    ax1.set_ylabel("Cross-Entropy Loss", color=SFT_COLOR)
    ax1.tick_params(axis="y", labelcolor=SFT_COLOR)
    ax1.set_ylim(0.5, 2.0)
    ax1.grid(alpha=0.3)

    ax1_r = ax1.twinx()
    ax1_r.plot(steps, lrs, color=LR_COLOR, linewidth=1.5, alpha=0.7, label="LR")
    ax1_r.set_ylabel("Learning Rate", color=LR_COLOR)
    ax1_r.tick_params(axis="y", labelcolor=LR_COLOR)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("SFT Training Dynamics — PolychromaticLM")

    # Warmup annotation
    ax1.axvline(x=100, color="gray", linestyle="--", alpha=0.5)
    ax1.annotate("Warmup\nends", xy=(100, 1.8), fontsize=8, color="gray", ha="center")

    # Bottom: Throughput
    ax2 = axes[1]
    ax2.plot(steps, smoothed_tps, color=QWEN_COLOR, linewidth=2, label="Throughput (smoothed)")
    ax2.fill_between(steps, tps, alpha=0.1, color=QWEN_COLOR)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Tokens/sec")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")
    mean_tps = sum(tps) / len(tps)
    ax2.axhline(y=mean_tps, color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(f"Mean: {mean_tps:.0f} tok/s", xy=(steps[-1] * 0.7, mean_tps + 50),
                 fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sft_training_dynamics.png"))
    plt.close()
    print("Saved sft_training_dynamics.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Always generate training dynamics plots (don't depend on eval results)
    print("Loading SFT training metrics...")
    metrics = load_sft_training_metrics()
    print(f"  {len(metrics)} data points")

    print("\nGenerating training plots...")
    plot_sft_loss_curve(metrics)
    plot_training_dynamics(metrics)

    # Load base benchmark results
    print("\nLoading base benchmark results...")
    base_bench = load_json(os.path.join(BASE_RESULTS_DIR, "loglikelihood_benchmarks.json"))
    base_scores = extract_scores(base_bench) if base_bench else {}
    print(f"  Base: {len(base_scores)} tasks")

    # Load SFT benchmark results
    print("Loading SFT benchmark results...")
    sft_bench = load_json(os.path.join(SFT_RESULTS_DIR, "loglikelihood_benchmarks.json"))
    sft_scores = extract_scores(sft_bench) if sft_bench else {}
    print(f"  SFT: {len(sft_scores)} tasks")

    # Comparison plots (require both base and SFT results)
    if base_scores and sft_scores:
        print("\nGenerating comparison plots...")
        plot_base_vs_sft_benchmarks(base_scores, sft_scores)
        plot_delta_chart(base_scores, sft_scores)
    else:
        print("\nSkipping comparison plots (missing base or SFT loglikelihood results)")

    # GSM8K comparison
    base_gsm8k_data = load_json(os.path.join(BASE_RESULTS_DIR, "gsm8k_results.json"))
    sft_gsm8k_data = load_json(os.path.join(SFT_RESULTS_DIR, "gsm8k_results.json"))
    base_gsm8k = extract_gsm8k_score(base_gsm8k_data) if base_gsm8k_data else None
    sft_gsm8k = extract_gsm8k_score(sft_gsm8k_data) if sft_gsm8k_data else None

    if base_gsm8k is not None or sft_gsm8k is not None:
        print("\nGenerating GSM8K comparison plot...")
        plot_gsm8k_comparison(base_gsm8k, sft_gsm8k)
    else:
        print("\nSkipping GSM8K comparison (no results yet)")

    # Print score summary
    if base_scores or sft_scores:
        print("\n" + "=" * 60)
        print("SCORE SUMMARY")
        print("=" * 60)
        all_tasks = sorted(set(list(base_scores.keys()) + list(sft_scores.keys())))
        print(f"{'Task':<20} {'Base':>10} {'SFT':>10} {'Delta':>10}")
        print("-" * 50)
        for t in all_tasks:
            b = base_scores.get(t)
            s = sft_scores.get(t)
            b_str = f"{b:.1%}" if b is not None else "—"
            s_str = f"{s:.1%}" if s is not None else "—"
            d_str = f"{s - b:+.1%}" if (b is not None and s is not None) else "—"
            print(f"{DISPLAY_NAMES.get(t, t):<20} {b_str:>10} {s_str:>10} {d_str:>10}")

        if base_gsm8k is not None or sft_gsm8k is not None:
            print("-" * 50)
            b_str = f"{base_gsm8k:.1%}" if base_gsm8k is not None else "—"
            s_str = f"{sft_gsm8k:.1%}" if sft_gsm8k is not None else "—"
            d_str = f"{sft_gsm8k - base_gsm8k:+.1%}" if (base_gsm8k is not None and sft_gsm8k is not None) else "—"
            print(f"{'GSM8K':<20} {b_str:>10} {s_str:>10} {d_str:>10}")

    print(f"\nAll plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
