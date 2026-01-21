"""Analyze experiment 3 results and generate figure with diagnostics."""

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
    
    # Extract diagnostics
    global_max_loads = results["global"]["max_loads"]
    block_max_loads = results["block"]["max_loads"]
    global_q2 = [occ.get("q2_estimate", 0.0) for occ in results["global"].get("occupancy_summary", [])]
    block_q2 = [occ.get("q2_estimate", 0.0) for occ in results["block"].get("occupancy_summary", [])]

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: Cosine similarity
    ax1.plot(N_values, global_cos, "r-o", label="Global Hash", linewidth=2)
    ax1.plot(N_values, block_cos, "b-o", label="Block Hash", linewidth=2)
    ax1.set_xlabel("Number of Writes (N)")
    ax1.set_ylabel("Mean Cosine Similarity")
    ax1.set_title("Exp 3: Block vs Global Hash - Fidelity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Diagnostics (max_load or q2_estimate)
    if len(global_q2) > 0 and len(block_q2) > 0:
        ax2.plot(N_values, global_q2, "r-o", label="Global Hash (q2)", linewidth=2)
        ax2.plot(N_values, block_q2, "b-o", label="Block Hash (q2)", linewidth=2)
        ax2.set_xlabel("Number of Writes (N)")
        ax2.set_ylabel("q2_estimate (Collision Proxy)")
        ax2.set_title("Exp 3: Block vs Global Hash - Diagnostics")
    else:
        # Fallback to max_load if q2 not available
        ax2.plot(N_values, global_max_loads, "r-o", label="Global Hash (max_load)", linewidth=2)
        ax2.plot(N_values, block_max_loads, "b-o", label="Block Hash (max_load)", linewidth=2)
        ax2.set_xlabel("Number of Writes (N)")
        ax2.set_ylabel("Max Load")
        ax2.set_title("Exp 3: Block vs Global Hash - Diagnostics")
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)

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
