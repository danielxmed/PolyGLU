"""Extract training metrics from both training log files into a unified CSV.

Parses the step-level log lines from:
  - First half (steps 0–12,360): paper_reporting/red0001_first_half_huge_bug_27_feb_2026.txt
  - Second half (steps 10,010–19,530): logs/train_resume_20260228_074132.log

Merges by taking steps 0–10,000 from first half and 10,010+ from second half.
Also extracts WandB dynamic routing entropy summary from the end of the second log.
"""

import csv
import re
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIRST_HALF_LOG = os.path.join(BASE_DIR, "paper_reporting", "red0001_first_half_huge_bug_27_feb_2026.txt")
SECOND_HALF_LOG = os.path.join(BASE_DIR, "logs", "train_resume_20260228_074132.log")
OUTPUT_CSV = os.path.join(BASE_DIR, "paper_reporting", "training_metrics.csv")
OUTPUT_DYNAMIC_ENTROPY = os.path.join(BASE_DIR, "paper_reporting", "final_dynamic_entropy.json")

# Pattern: Step 10010/19531 | Loss: 2.0527 | LR: 0.000057 | τ: 0.539 | tok/s: 11723 | entropy: 1.386
STEP_PATTERN = re.compile(
    r"Step (\d+)/\d+ \| Loss: ([\d.]+) \| LR: ([\d.]+) \| τ: ([\d.]+) \| tok/s: (\d+) \| entropy: ([\d.]+)"
)

# Pattern for WandB dynamic routing entropy summary
DYNAMIC_ENTROPY_PATTERN = re.compile(
    r"dynamic_routing_entropy/(layer_\d+|mean)\s+([\d.e-]+)"
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
                    "tau": float(m.group(4)),
                    "tokens_per_sec": int(m.group(5)),
                    "static_entropy": float(m.group(6)),
                })
    return rows


def extract_dynamic_entropy(filepath: str) -> dict:
    """Extract final WandB dynamic routing entropy summary from end of log."""
    entropy = {}
    with open(filepath) as f:
        lines = f.readlines()
    # WandB summary is in the last ~40 lines
    for line in lines[-60:]:
        m = DYNAMIC_ENTROPY_PATTERN.search(line)
        if m:
            key = m.group(1)
            val = float(m.group(2))
            entropy[key] = val
    return entropy


def main():
    print("Parsing first half log...")
    first_half = parse_log(FIRST_HALF_LOG)
    print(f"  Found {len(first_half)} data points (steps {first_half[0]['step']}–{first_half[-1]['step']})")

    print("Parsing second half log...")
    second_half = parse_log(SECOND_HALF_LOG)
    print(f"  Found {len(second_half)} data points (steps {second_half[0]['step']}–{second_half[-1]['step']})")

    # Merge: steps 0–10,000 from first half, 10,010+ from second half
    merged = [r for r in first_half if r["step"] <= 10000]
    merged.extend(second_half)
    merged.sort(key=lambda r: r["step"])

    # Remove duplicates (keep first occurrence)
    seen = set()
    deduped = []
    for r in merged:
        if r["step"] not in seen:
            seen.add(r["step"])
            deduped.append(r)

    print(f"Merged: {len(deduped)} data points (steps {deduped[0]['step']}–{deduped[-1]['step']})")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "tau", "tokens_per_sec", "static_entropy"])
        writer.writeheader()
        writer.writerows(deduped)
    print(f"Saved to {OUTPUT_CSV}")

    # Extract dynamic entropy
    print("Extracting dynamic routing entropy from WandB summary...")
    entropy = extract_dynamic_entropy(SECOND_HALF_LOG)
    if entropy:
        with open(OUTPUT_DYNAMIC_ENTROPY, "w") as f:
            json.dump(entropy, f, indent=2)
        print(f"Saved {len(entropy)} entropy values to {OUTPUT_DYNAMIC_ENTROPY}")
    else:
        print("  WARNING: No dynamic entropy data found in log")


if __name__ == "__main__":
    main()
