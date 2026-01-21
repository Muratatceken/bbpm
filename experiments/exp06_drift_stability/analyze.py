"""Analyze experiment 6 results and generate figure."""

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

    steps = results["steps"]
    stable_acc = results["stable_accuracy"]
    drifting_acc = results["drifting_accuracy"]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, stable_acc, "b-o", label="Stable Keys", linewidth=2, markersize=4)
    plt.plot(steps, drifting_acc, "r--o", label="Drifting Keys", linewidth=2, markersize=4)
    plt.xlabel("Time Step")
    plt.ylabel("Retrieval Accuracy (Cosine Similarity)")
    plt.title("Exp 6: Drift Stability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "drift_stability.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp06 results")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "exp06_drift_stability" / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "figures" / "exp06_drift_stability")

    args = parser.parse_args()
    analyze(args.results, args.outdir)
