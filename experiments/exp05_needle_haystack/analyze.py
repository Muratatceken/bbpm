"""Analyze experiment 5 results and generate figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze(results_path: Path, outdir: Path):
    """Analyze results and generate figure."""
    with open(results_path) as f:
        results = json.load(f)

    N_values = results["N_values"]
    bbpm_success = results["bbpm_success"]
    window_success = results["window_success"]

    plt.figure(figsize=(8, 5))
    plt.plot(N_values, bbpm_success, "b-o", label="BBPM", linewidth=2)
    plt.plot(N_values, window_success, "r--o", label="Window Baseline", linewidth=2)
    plt.xlabel("Number of Stored Items (N)")
    plt.ylabel("Retrieval Success Rate")
    plt.title("Exp 5: Needle in a Haystack")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "needle_accuracy.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp05 results")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "exp05_needle_haystack" / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "figures" / "exp05_needle_haystack")

    args = parser.parse_args()
    analyze(args.results, args.outdir)
