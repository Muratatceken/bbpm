"""Analyze experiment 3 results and generate figure."""

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
    global_cos = results["global"]["cosines"]
    block_cos = results["block"]["cosines"]

    plt.figure(figsize=(8, 5))
    plt.plot(N_values, global_cos, "r-o", label="Global Hash", linewidth=2)
    plt.plot(N_values, block_cos, "b-o", label="Block Hash", linewidth=2)
    plt.xlabel("Number of Writes (N)")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Exp 3: Block vs Global Hash Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "block_vs_global.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp03 results")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "exp03_block_vs_global" / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "figures" / "exp03_block_vs_global")

    args = parser.parse_args()
    analyze(args.results, args.outdir)
