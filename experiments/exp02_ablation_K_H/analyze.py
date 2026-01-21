"""Analyze experiment 2 results and generate figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze(results_path: Path, outdir: Path):
    """Analyze results and generate figure with error bars and capacity metrics."""
    with open(results_path) as f:
        results = json.load(f)

    K_values = results["K_values"]
    H_values = results["H_values"]
    seeds_data = results.get("seeds", {})

    # Aggregate data across seeds
    # Structure: {H: {K: [accuracies], capacity_units: value}}
    data_by_H = {}
    num_seeds = len(seeds_data)

    for H in H_values:
        data_by_H[H] = {}
        for K in K_values:
            data_by_H[H][K] = {
                "accuracies": [],
                "avg_cosines": [],
                "capacity_units": None,
            }

        # Collect from all seeds
        for seed_key, seed_data in seeds_data.items():
            H_key = f"H_{H}"
            if H_key not in seed_data:
                continue
            H_data = seed_data[H_key]
            K_vals = H_data.get("K_values", [])
            accs = H_data.get("accuracies", [])
            cap_units = H_data.get("capacity_units", [])

            for i, K in enumerate(K_vals):
                if i < len(accs) and i < len(cap_units):
                    data_by_H[H][K]["accuracies"].append(accs[i])
                    if data_by_H[H][K]["capacity_units"] is None:
                        data_by_H[H][K]["capacity_units"] = cap_units[i]

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    colors = ["b", "g", "r", "m"]
    for h_idx, H in enumerate(H_values):
        color = colors[h_idx % len(colors)]
        K_vals = []
        acc_means = []
        acc_stds = []
        cap_units_vals = []

        for K in K_values:
            if K in data_by_H[H]:
                vals = data_by_H[H][K]
                if len(vals["accuracies"]) > 0:
                    K_vals.append(K)
                    acc_means.append(np.mean(vals["accuracies"]))
                    acc_stds.append(np.std(vals["accuracies"]))
                    cap_units_vals.append(vals["capacity_units"])

        # Plot 1: Accuracy vs K
        if len(K_vals) > 0:
            if num_seeds >= 3:
                ax1.errorbar(
                    K_vals, acc_means, yerr=acc_stds,
                    fmt=f"{color}-o", label=f"H={H}", linewidth=2, capsize=3
                )
            else:
                ax1.plot(K_vals, acc_means, f"{color}-o", label=f"H={H}", linewidth=2)

        # Plot 2: Accuracy vs Capacity Units
        if len(cap_units_vals) > 0:
            if num_seeds >= 3:
                ax2.errorbar(
                    cap_units_vals, acc_means, yerr=acc_stds,
                    fmt=f"{color}-o", label=f"H={H}", linewidth=2, capsize=3
                )
            else:
                ax2.plot(cap_units_vals, acc_means, f"{color}-o", label=f"H={H}", linewidth=2)

    # Add vertical line at capacity_units = 1.0
    ax2.axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, label='Theoretical Capacity (N*=1)', alpha=0.7)

    ax1.set_xlabel("Sparsity Factor (K)")
    ax1.set_ylabel("Retrieval Accuracy")
    ax1.set_title("Exp 2: Accuracy vs K")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Capacity Units (N / (D/(K·H)))")
    ax2.set_ylabel("Retrieval Accuracy")
    ax2.set_title("Exp 2: Accuracy vs Capacity Units")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

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
