"""Generate evaluation visualization plots for the base model performance report.

Reads benchmark results and domain perplexity data to produce paper-quality figures.
All outputs saved to paper_reporting/figures/.

Usage:
    python paper_reporting/generate_eval_plots.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "base_eval")
FIGURES_DIR = os.path.join(BASE_DIR, "paper_reporting", "figures")

# Style (matches generate_plots.py)
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

# Colors
POLY_COLOR = "#2ecc71"     # Green for PolychromaticLM
QWEN_COLOR = "#3498db"     # Blue for Qwen3 reference
RANDOM_COLOR = "#bdc3c7"   # Gray for random baseline
DOMAIN_COLORS = {"math": "#e74c3c", "stem": "#3498db", "code": "#2ecc71"}


def load_benchmark_results() -> dict:
    """Load loglikelihood benchmark results."""
    path = os.path.join(RESULTS_DIR, "loglikelihood_benchmarks.json")
    with open(path) as f:
        return json.load(f)


def load_gsm8k_results() -> dict:
    """Load GSM8K results."""
    path = os.path.join(RESULTS_DIR, "gsm8k_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_domain_perplexity() -> dict:
    """Load domain perplexity results."""
    path = os.path.join(RESULTS_DIR, "domain_perplexity.json")
    with open(path) as f:
        return json.load(f)


# Qwen3-0.6B-Base reference scores (published)
QWEN3_BASELINES = {
    "hellaswag": 0.411,
    "arc_easy": 0.656,
    "arc_challenge": 0.339,
    "piqa": 0.700,
    "winogrande": 0.585,
    "boolq": 0.697,
}

# Random baselines (by number of choices)
RANDOM_BASELINES = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "piqa": 0.50,
    "winogrande": 0.50,
    "boolq": 0.50,
    "openbookqa": 0.25,
    "sciq": 0.25,
    "mathqa": 0.20,  # 5 choices
    "mmlu_stem": 0.25,
    "lambada_openai": 0.0,
}

# Preferred metric for each task
METRIC_MAP = {
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
    "sciq": "acc_norm,none",
    "mathqa": "acc_norm,none",
    "mmlu_stem": "acc,none",
    "lambada_openai": "acc,none",
}

# Display names
DISPLAY_NAMES = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "sciq": "SciQ",
    "mathqa": "MathQA",
    "mmlu_stem": "MMLU-STEM",
    "lambada_openai": "LAMBADA",
}


def extract_scores(results: dict) -> dict:
    """Extract scores from lm-eval results JSON."""
    scores = {}
    for task_name, metric_key in METRIC_MAP.items():
        task_results = results.get("results", {}).get(task_name, {})
        if metric_key in task_results:
            scores[task_name] = task_results[metric_key]
        # Fall back to acc,none for mmlu_stem which may be nested
        elif task_name == "mmlu_stem":
            # mmlu_stem might aggregate sub-tasks
            for key, val in task_results.items():
                if "acc" in key and "stderr" not in key and isinstance(val, (int, float)):
                    scores[task_name] = val
                    break
    return scores


def plot_benchmark_comparison(scores: dict):
    """Grouped bar chart: PolychromaticLM vs Qwen3-0.6B-Base vs random baseline."""
    # Tasks where we have Qwen3 baselines
    shared_tasks = [t for t in ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq"]
                    if t in scores]

    if not shared_tasks:
        print("No shared tasks for comparison plot, skipping.")
        return

    x = np.arange(len(shared_tasks))
    width = 0.25

    poly_scores = [scores[t] for t in shared_tasks]
    qwen_scores = [QWEN3_BASELINES[t] for t in shared_tasks]
    random_scores = [RANDOM_BASELINES[t] for t in shared_tasks]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, poly_scores, width, label="PolychromaticLM (ours)", color=POLY_COLOR, edgecolor="white")
    bars2 = ax.bar(x, qwen_scores, width, label="Qwen3-0.6B-Base", color=QWEN_COLOR, edgecolor="white")
    bars3 = ax.bar(x + width, random_scores, width, label="Random", color=RANDOM_COLOR, edgecolor="white")

    ax.set_ylabel("Accuracy")
    ax.set_title("PolychromaticLM vs Qwen3-0.6B-Base — Common Benchmarks")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[t] for t in shared_tasks], rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_benchmark_comparison.png"))
    plt.close()
    print("Saved eval_benchmark_comparison.png")


def plot_all_benchmarks(scores: dict):
    """Horizontal bar chart of all benchmark scores."""
    tasks = [t for t in METRIC_MAP if t in scores]
    if not tasks:
        print("No scores to plot.")
        return

    y = np.arange(len(tasks))
    vals = [scores[t] for t in tasks]
    randoms = [RANDOM_BASELINES.get(t, 0) for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(y, vals, color=POLY_COLOR, edgecolor="white", height=0.6, label="PolychromaticLM")
    ax.barh(y, randoms, color=RANDOM_COLOR, edgecolor="white", height=0.6, alpha=0.3, label="Random baseline")

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES[t] for t in tasks])
    ax.set_xlabel("Accuracy")
    ax.set_title("PolychromaticLM Base Model — All Benchmarks")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.1%}",
                va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_all_benchmarks.png"))
    plt.close()
    print("Saved eval_all_benchmarks.png")


def plot_domain_perplexity(domain_results: dict):
    """Bar chart of per-domain perplexity."""
    domains = ["math", "stem", "code"]
    ppls = [domain_results[d]["perplexity"] for d in domains]
    losses = [domain_results[d]["avg_loss"] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Perplexity bars
    bars1 = ax1.bar(domains, ppls, color=[DOMAIN_COLORS[d] for d in domains], edgecolor="white", width=0.5)
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Domain Perplexity (lower = better)")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, ppls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")

    # Loss bars
    bars2 = ax2.bar(domains, losses, color=[DOMAIN_COLORS[d] for d in domains], edgecolor="white", width=0.5)
    ax2.set_ylabel("Average Cross-Entropy Loss")
    ax2.set_title("Domain Loss (nats)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Training mix annotation
    ax1.annotate("Training mix: 70% / 25% / 5%", xy=(0.5, 0.02), xycoords="axes fraction",
                 ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_domain_perplexity.png"))
    plt.close()
    print("Saved eval_domain_perplexity.png")


def plot_category_summary(scores: dict):
    """Summary chart grouping benchmarks by category."""
    categories = {
        "Language\nUnderstanding": ["hellaswag", "piqa", "winogrande", "lambada_openai"],
        "Knowledge\n& Reasoning": ["arc_easy", "arc_challenge", "boolq", "openbookqa", "sciq", "mmlu_stem"],
        "Mathematical": ["mathqa"],
    }

    cat_names = []
    cat_scores = []
    cat_counts = []
    for cat, tasks in categories.items():
        available = [scores[t] for t in tasks if t in scores]
        if available:
            cat_names.append(cat)
            cat_scores.append(np.mean(available))
            cat_counts.append(len(available))

    if not cat_names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cat_names, cat_scores, color=[POLY_COLOR, QWEN_COLOR, "#e74c3c"][:len(cat_names)],
                  edgecolor="white", width=0.5)
    ax.set_ylabel("Average Accuracy")
    ax.set_title("PolychromaticLM — Performance by Category")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar, val, n in zip(bars, cat_scores, cat_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}\n(n={n})", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_category_summary.png"))
    plt.close()
    print("Saved eval_category_summary.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading benchmark results...")
    benchmark_results = load_benchmark_results()
    scores = extract_scores(benchmark_results)
    print(f"  Found scores for {len(scores)} tasks: {list(scores.keys())}")

    # Add GSM8K if available
    gsm8k_results = load_gsm8k_results()
    if gsm8k_results:
        gsm8k_task = gsm8k_results.get("results", {}).get("gsm8k", {})
        gsm8k_score = gsm8k_task.get("exact_match,strict-match", None)
        if gsm8k_score is not None:
            scores["gsm8k"] = gsm8k_score
            print(f"  Added GSM8K: {gsm8k_score:.4f}")

    print("\nScores:")
    for task, score in sorted(scores.items()):
        print(f"  {DISPLAY_NAMES.get(task, task)}: {score:.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_benchmark_comparison(scores)
    plot_all_benchmarks(scores)
    plot_category_summary(scores)

    # Domain perplexity
    ppl_path = os.path.join(RESULTS_DIR, "domain_perplexity.json")
    if os.path.exists(ppl_path):
        print("\nLoading domain perplexity results...")
        domain_results = load_domain_perplexity()
        plot_domain_perplexity(domain_results)
    else:
        print("\nDomain perplexity results not found, skipping plot.")

    print("\nAll plots saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
