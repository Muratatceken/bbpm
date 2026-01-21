"""Analyze experiment 7 results and generate figure."""

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

    if not results.get("transformers_available", False):
        print("Note: Results from synthetic mode (transformers not available)")

    N_values = results["N_values"]
    bbpm_acc = results["bbpm_accuracy"]
    baseline_acc = results["baseline_accuracy"]
    oracle_acc = results.get("oracle_accuracy", [])

    plt.figure(figsize=(8, 5))
    plt.plot(N_values, bbpm_acc, "b-o", label="BBPM-Augmented", linewidth=2)
    plt.plot(N_values, baseline_acc, "r--o", label="Baseline (Window)", linewidth=2)
    if oracle_acc:
        plt.plot(N_values, oracle_acc, "g--o", label="Oracle (Upper Bound)", linewidth=2, alpha=0.7)
    plt.xlabel("Number of Streamed Facts (N)")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Exp 7: LLM Integration - Controller-Based Retrieval Injection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "llm_accuracy_vs_N.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp07 results")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "exp07_llm_integration" / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "figures" / "exp07_llm_integration")

    args = parser.parse_args()
    analyze(args.results, args.outdir)
