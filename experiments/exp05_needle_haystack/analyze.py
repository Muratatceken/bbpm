"""Analyze experiment 5 results and generate figure with error bars."""

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
    # Structure: {(D, N): [bbpm_success_values, window_success_values, N_over_D, q2_values]}
    data_by_D = defaultdict(lambda: defaultdict(lambda: {"bbpm": [], "window": [], "N_over_D": None, "q2": []}))

    seeds_data = results.get("seeds", {})
    num_seeds = len(seeds_data)

    # Collect data from all seeds
    for seed_key, seed_data in seeds_data.items():
        for D_key, D_data in seed_data.items():
            if not D_key.startswith("D_"):
                continue
            D = D_data["D"]
            for run in D_data["runs"]:
                N = run["N"]
                key = (D, N)
                data_by_D[D][N]["bbpm"].append(run["bbpm_success"])
                data_by_D[D][N]["window"].append(run["window_success"])
                data_by_D[D][N]["N_over_D"] = run["N_over_D"]
                data_by_D[D][N]["q2"].append(run["q2_estimate"])

    # Prepare plotting data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: Success vs N/D with error bars
    colors = ["b", "g", "r", "m", "c"]
    for idx, D in enumerate(sorted(data_by_D.keys())):
        D_data = data_by_D[D]
        N_over_D_vals = []
        bbpm_means = []
        bbpm_stds = []
        window_means = []
        window_stds = []

        for N in sorted(D_data.keys()):
            vals = D_data[N]
            if len(vals["bbpm"]) > 0:
                N_over_D_vals.append(vals["N_over_D"])
                bbpm_means.append(np.mean(vals["bbpm"]))
                bbpm_stds.append(np.std(vals["bbpm"]))
                window_means.append(np.mean(vals["window"]))
                window_stds.append(np.std(vals["window"]))

        if len(N_over_D_vals) > 0:
            color = colors[idx % len(colors)]
            if num_seeds >= 3:
                ax1.errorbar(
                    N_over_D_vals, bbpm_means, yerr=bbpm_stds,
                    fmt=f"{color}-o", label=f"BBPM (D={D:,})", linewidth=2, capsize=3
                )
            else:
                ax1.plot(N_over_D_vals, bbpm_means, f"{color}-o", label=f"BBPM (D={D:,})", linewidth=2)

            # Window baseline (same for all D, plot once)
            if idx == 0:
                if num_seeds >= 3:
                    ax1.errorbar(
                        N_over_D_vals, window_means, yerr=window_stds,
                        fmt="r--o", label="Window Baseline", linewidth=2, capsize=3
                    )
                else:
                    ax1.plot(N_over_D_vals, window_means, "r--o", label="Window Baseline", linewidth=2)

    ax1.set_xlabel("N/D (Load Ratio)")
    ax1.set_ylabel("Retrieval Success Rate")
    ax1.set_title("Exp 5: Needle in a Haystack - Success vs N/D")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success vs q2_estimate (optional)
    for idx, D in enumerate(sorted(data_by_D.keys())):
        D_data = data_by_D[D]
        q2_vals = []
        bbpm_means = []
        bbpm_stds = []

        for N in sorted(D_data.keys()):
            vals = D_data[N]
            if len(vals["bbpm"]) > 0 and len(vals["q2"]) > 0:
                q2_vals.append(np.mean(vals["q2"]))
                bbpm_means.append(np.mean(vals["bbpm"]))
                bbpm_stds.append(np.std(vals["bbpm"]))

        if len(q2_vals) > 0:
            color = colors[idx % len(colors)]
            if num_seeds >= 3:
                ax2.errorbar(
                    q2_vals, bbpm_means, yerr=bbpm_stds,
                    fmt=f"{color}-o", label=f"BBPM (D={D:,})", linewidth=2, capsize=3
                )
            else:
                ax2.plot(q2_vals, bbpm_means, f"{color}-o", label=f"BBPM (D={D:,})", linewidth=2)

    ax2.set_xlabel("q2_estimate (Collision Proxy)")
    ax2.set_ylabel("Retrieval Success Rate")
    ax2.set_title("Exp 5: Success vs q2_estimate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

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
