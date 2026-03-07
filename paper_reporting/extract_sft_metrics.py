"""Extract SFT training metrics from WandB log into CSV.

Parses the step-level log lines from:
  wandb/run-20260305_171955-xhfwjtsr/files/output.log

Output: paper_reporting/sft_training_metrics.csv
  Columns: step, loss, lr, tokens_per_sec, mem_gb, entropy
  ~1,306 data points (every 10 steps)
"""

import csv
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SFT_LOG = os.path.join(BASE_DIR, "wandb", "run-20260305_171955-xhfwjtsr", "files", "output.log")
OUTPUT_CSV = os.path.join(BASE_DIR, "paper_reporting", "sft_training_metrics.csv")

# Pattern: Step 10/13067 | Loss: 1.7734 | LR: 0.000002 | tok/s: 10342 | mem: 8.4GB | entropy: 1.386
STEP_PATTERN = re.compile(
    r"Step (\d+)/\d+ \| Loss: ([\d.]+) \| LR: ([\d.]+) \| tok/s: (\d+) \| mem: ([\d.]+)GB \| entropy: ([\d.]+)"
)


def parse_log(filepath: str) -> list[dict]:
    rows = []
    with open(filepath) as f:
        for line in f:
            m = STEP_PATTERN.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "lr": float(m.group(3)),
                    "tokens_per_sec": int(m.group(4)),
                    "mem_gb": float(m.group(5)),
                    "entropy": float(m.group(6)),
                })
    return rows


def main():
    print(f"Parsing SFT log: {SFT_LOG}")
    rows = parse_log(SFT_LOG)
    print(f"  Found {len(rows)} data points (steps {rows[0]['step']}–{rows[-1]['step']})")

    # Remove duplicates (keep last occurrence for each step)
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    deduped = sorted(seen.values(), key=lambda r: r["step"])
    print(f"  After dedup: {len(deduped)} data points")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "tokens_per_sec", "mem_gb", "entropy"])
        writer.writeheader()
        writer.writerows(deduped)
    print(f"Saved to {OUTPUT_CSV}")

    # Print summary stats
    losses = [r["loss"] for r in deduped]
    print(f"\nSFT Training Summary:")
    print(f"  Initial loss (step {deduped[0]['step']}): {deduped[0]['loss']:.4f}")
    print(f"  Final loss (step {deduped[-1]['step']}): {deduped[-1]['loss']:.4f}")
    print(f"  Min loss: {min(losses):.4f}")
    print(f"  Loss reduction: {deduped[0]['loss']:.4f} → {deduped[-1]['loss']:.4f} ({(1 - deduped[-1]['loss']/deduped[0]['loss'])*100:.1f}%)")
    print(f"  Entropy: {deduped[0]['entropy']:.3f} (constant)")
    tps = [r["tokens_per_sec"] for r in deduped]
    print(f"  Mean throughput: {sum(tps)/len(tps):.0f} tok/s")


if __name__ == "__main__":
    main()
