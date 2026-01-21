"""Analyze experiment 2 results and generate figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze(results_path: Path, outdir: Path):
    """Analyze results and generate figure."""
    with open(results_path) as f:
        results = json.load(f)

    K_values = results["K_values"]
    H_values = results["H_values"]

    plt.figure(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(K_values))

    for i, H in enumerate(H_values):
        key = f"H={H}"
        accuracies = results["results"][key]["accuracies"]
        offset = width * (i - 0.5)
        plt.bar(x + offset, accuracies, width, label=f"Multi Hash (H={H})")

    plt.xticks(x, K_values)
    plt.xlabel("Sparsity Factor (K)")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Exp 2: Ablation Study (Effect of K and H)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "ablation_K_H.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp02 results")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "exp02_ablation_K_H" / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "figures" / "exp02_ablation_K_H")

    args = parser.parse_args()
    analyze(args.results, args.outdir)
