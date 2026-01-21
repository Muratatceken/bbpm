"""Analyze experiment 1 results and generate figure with error bars."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze(results_path: Path, outdir: Path):
    """Analyze results and generate figure with error bars."""
    with open(results_path) as f:
        results = json.load(f)

    # Aggregate data across seeds
    # Structure: {N: [cosine_values, capacity_units, N_over_D]}
    data_by_N = defaultdict(lambda: {"cosine": [], "capacity_units": None, "N_over_D": None})

    seeds_data = results.get("seeds", {})
    num_seeds = len(seeds_data)

    # Collect data from all seeds
    for seed_key, seed_data in seeds_data.items():
        item_counts = seed_data.get("item_counts", [])
        cosine_sims = seed_data.get("cosine_similarities", [])
        diagnostics_list = seed_data.get("diagnostics", [])
        N_over_D_vals = seed_data.get("N_over_D", [])

        for i, N in enumerate(item_counts):
            if i < len(cosine_sims) and i < len(diagnostics_list) and i < len(N_over_D_vals):
                data_by_N[N]["cosine"].append(cosine_sims[i])
                # Extract capacity_units from diagnostics
                cap_units = diagnostics_list[i].get("capacity_units", None)
                if cap_units is not None:
                    data_by_N[N]["capacity_units"] = cap_units
                data_by_N[N]["N_over_D"] = N_over_D_vals[i]

    # Prepare plotting data
    N_vals = sorted(data_by_N.keys())
    capacity_units_vals = []
    N_over_D_vals = []
    cosine_means = []
    cosine_stds = []

    for N in N_vals:
        vals = data_by_N[N]
        if len(vals["cosine"]) > 0:
            # Prefer capacity_units, fallback to N_over_D if not available
            x_val = vals["capacity_units"] if vals["capacity_units"] is not None else vals["N_over_D"]
            capacity_units_vals.append(x_val)
            N_over_D_vals.append(vals["N_over_D"])
            cosine_means.append(np.mean(vals["cosine"]))
            cosine_stds.append(np.std(vals["cosine"]))

    # Create figure
    plt.figure(figsize=(8, 5))
    
    if num_seeds >= 3:
        plt.errorbar(
            capacity_units_vals, cosine_means, yerr=cosine_stds,
            fmt="b-o", label="Measured Fidelity (CosSim)", linewidth=2, capsize=3
        )
    else:
        plt.plot(capacity_units_vals, cosine_means, "b-o", label="Measured Fidelity (CosSim)", linewidth=2)

    # Add vertical line at theoretical capacity
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, label='Theoretical Capacity (N*=1)', alpha=0.7)

    plt.xlabel("Capacity Units (N / (D/(K·H)))")
    plt.ylabel("Retrieval Fidelity (Cosine Similarity)")
    plt.title("Exp 1: Capacity vs. Fidelity (D=1M, K=50, H=1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
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
