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
    # Structure: {D: {N: {bbpm: [], window_W_*: [], oracle: [], capacity_units: None, ...}}}
    data_by_D = defaultdict(lambda: defaultdict(lambda: {
        "bbpm": [],
        "oracle": [],
        "capacity_units": None,
        "N_over_D": None,
        "q2": [],
        "window": {},  # Dict of {W: [values]}
    }))

    seeds_data = results.get("seeds", {})
    num_seeds = len(seeds_data)

    # Collect data from all seeds
    window_sizes_found = set()
    for seed_key, seed_data in seeds_data.items():
        for D_key, D_data in seed_data.items():
            if not D_key.startswith("D_"):
                continue
            D = D_data["D"]
            for run in D_data["runs"]:
                N = run["N"]
                data_by_D[D][N]["bbpm"].append(run.get("bbpm_success", 0.0))
                data_by_D[D][N]["oracle"].append(run.get("oracle_success", 1.0))
                if data_by_D[D][N]["capacity_units"] is None:
                    data_by_D[D][N]["capacity_units"] = run.get("capacity_units", run.get("N_over_D", 0.0))
                    data_by_D[D][N]["N_over_D"] = run.get("N_over_D", 0.0)
                data_by_D[D][N]["q2"].append(run.get("q2_estimate", 0.0))

                # Collect window successes
                for key, value in run.items():
                    if key.startswith("window_success_W_"):
                        W = int(key.replace("window_success_W_", ""))
                        window_sizes_found.add(W)
                        if W not in data_by_D[D][N]["window"]:
                            data_by_D[D][N]["window"][W] = []
                        data_by_D[D][N]["window"][W].append(value)

    # Prepare plotting data
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot 1: Success vs Capacity Units with error bars
    colors = ["b", "g", "r", "m", "c", "y"]
    window_colors = ["orange", "purple", "brown", "pink"]
    window_styles = ["--", "-.", ":", "-"]

    # Plot BBPM for each D
    for idx, D in enumerate(sorted(data_by_D.keys())):
        D_data = data_by_D[D]
        capacity_units_vals = []
        bbpm_means = []
        bbpm_stds = []

        for N in sorted(D_data.keys()):
            vals = D_data[N]
            if len(vals["bbpm"]) > 0:
                capacity_units_vals.append(vals["capacity_units"])
                bbpm_means.append(np.mean(vals["bbpm"]))
                bbpm_stds.append(np.std(vals["bbpm"]))

        if len(capacity_units_vals) > 0:
            color = colors[idx % len(colors)]
            if num_seeds >= 3:
                ax1.errorbar(
                    capacity_units_vals, bbpm_means, yerr=bbpm_stds,
                    fmt=f"{color}-o", label=f"BBPM (D={D:,})", linewidth=2, capsize=3
                )
            else:
                ax1.plot(capacity_units_vals, bbpm_means, f"{color}-o", label=f"BBPM (D={D:,})", linewidth=2)

    # Plot window baselines (use first D's data, but they should be similar)
    first_D = sorted(data_by_D.keys())[0] if data_by_D else None
    if first_D:
        D_data = data_by_D[first_D]
        sorted_window_sizes = sorted(window_sizes_found)
        for w_idx, W in enumerate(sorted_window_sizes):
            capacity_units_vals = []
            window_means = []
            window_stds = []

            for N in sorted(D_data.keys()):
                vals = D_data[N]
                if W in vals["window"] and len(vals["window"][W]) > 0:
                    capacity_units_vals.append(vals["capacity_units"])
                    window_means.append(np.mean(vals["window"][W]))
                    window_stds.append(np.std(vals["window"][W]))

            if len(capacity_units_vals) > 0:
                w_color = window_colors[w_idx % len(window_colors)]
                w_style = window_styles[w_idx % len(window_styles)]
                if num_seeds >= 3:
                    ax1.errorbar(
                        capacity_units_vals, window_means, yerr=window_stds,
                        fmt=f"{w_color}{w_style}o", label=f"Window (W={W:,})", linewidth=1.5, capsize=2, alpha=0.7
                    )
                else:
                    ax1.plot(capacity_units_vals, window_means, f"{w_color}{w_style}o", 
                           label=f"Window (W={W:,})", linewidth=1.5, alpha=0.7)

    # Plot oracle baseline (should be flat at 1.0)
    if first_D:
        D_data = data_by_D[first_D]
        capacity_units_vals = []
        oracle_means = []
        for N in sorted(D_data.keys()):
            vals = D_data[N]
            if len(vals["oracle"]) > 0:
                capacity_units_vals.append(vals["capacity_units"])
                oracle_means.append(np.mean(vals["oracle"]))
        if len(capacity_units_vals) > 0:
            ax1.plot(capacity_units_vals, oracle_means, "k-", label="Oracle (Perfect)", 
                    linewidth=2, alpha=0.8, linestyle=":")

    # Add vertical line at theoretical capacity
    ax1.axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, label='Theoretical Capacity (N*=1)', alpha=0.7)

    ax1.set_xlabel("Capacity Units (N / (D/(K·H)))")
    ax1.set_ylabel("Retrieval Success Rate")
    ax1.set_title("Exp 5: Needle in a Haystack - Success vs Capacity Units")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

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
