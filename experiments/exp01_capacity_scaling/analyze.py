"""Analyze experiment 1 results and generate figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze(results_path: Path, outdir: Path):
    """Analyze results and generate figure."""
    # Load results
    with open(results_path) as f:
        results = json.load(f)

    item_counts = results["item_counts"]
    cosine_similarities = results["cosine_similarities"]

    # Create figure
    plt.figure(figsize=(8, 5))
    plt.plot(item_counts, cosine_similarities, "b-o", label="Measured Fidelity (CosSim)", linewidth=2)
    plt.xlabel("Number of Stored Items (N)")
    plt.ylabel("Retrieval Fidelity (Cosine Similarity)")
    plt.title("Exp 1: Capacity vs. Fidelity (D=1M, K=50)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "capacity_vs_fidelity.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp01 results")
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "exp01_capacity_scaling" / "metrics.json",
        help="Path to results JSON",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "exp01_capacity_scaling",
        help="Output directory for figures",
    )

    args = parser.parse_args()
    analyze(args.results, args.outdir)
