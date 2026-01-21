"""Analyze experiment 7 results and generate figure with error bars."""

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

    if not results.get("transformers_available", False):
        print("Note: Results from synthetic mode (transformers not available)")

    # Aggregate data across seeds
    # Structure: {(D, K, H): {N: [accuracy_values, N_over_D, tokens_per_sec_values]}}
    data_by_config = defaultdict(lambda: defaultdict(lambda: {
        "bbpm": [], "baseline": [], "oracle": [], "N_over_D": None, "tokens_per_sec": []
    }))

    seeds_data = results.get("seeds", {})
    num_seeds = len(seeds_data)

    # Collect data from all seeds
    for seed_key, seed_data in seeds_data.items():
        for config_key, config_data in seed_data.items():
            D = config_data["D"]
            K = config_data["K"]
            H = config_data["H"]
            N = config_data["N"]
            key = (D, K, H)
            
            data_by_config[key][N]["bbpm"].append(config_data["bbpm_accuracy"])
            data_by_config[key][N]["baseline"].append(config_data.get("baseline_accuracy", 0.0))
            data_by_config[key][N]["oracle"].append(config_data.get("oracle_accuracy", 0.0))
            data_by_config[key][N]["N_over_D"] = config_data["N_over_D"]
            data_by_config[key][N]["tokens_per_sec"].append(config_data.get("tokens_per_sec", 0.0))

    # Plot 1: Accuracy vs N_over_D with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    colors = ["b", "g", "r", "m", "c"]
    color_idx = 0

    # Group by D for main plot
    for (D, K, H) in sorted(data_by_config.keys()):
        config_data = data_by_config[(D, K, H)]
        N_over_D_vals = []
        bbpm_means = []
        bbpm_stds = []
        baseline_means = []
        baseline_stds = []
        oracle_means = []
        oracle_stds = []

        for N in sorted(config_data.keys()):
            vals = config_data[N]
            if len(vals["bbpm"]) > 0:
                N_over_D_vals.append(vals["N_over_D"])
                bbpm_means.append(np.mean(vals["bbpm"]))
                bbpm_stds.append(np.std(vals["bbpm"]))
                baseline_means.append(np.mean(vals["baseline"]))
                baseline_stds.append(np.std(vals["baseline"]))
                oracle_means.append(np.mean(vals["oracle"]))
                oracle_stds.append(np.std(vals["oracle"]))

        if len(N_over_D_vals) > 0:
            color = colors[color_idx % len(colors)]
            label = f"BBPM (D={D:,}, K={K}, H={H})"
            
            if num_seeds >= 3:
                ax1.errorbar(
                    N_over_D_vals, bbpm_means, yerr=bbpm_stds,
                    fmt=f"{color}-o", label=label, linewidth=2, capsize=3
                )
            else:
                ax1.plot(N_over_D_vals, bbpm_means, f"{color}-o", label=label, linewidth=2)

            # Baseline and oracle (plot once for first config)
            if color_idx == 0:
                if num_seeds >= 3:
                    ax1.errorbar(
                        N_over_D_vals, baseline_means, yerr=baseline_stds,
                        fmt="r--o", label="Baseline (Window)", linewidth=2, capsize=3
                    )
                    if len(oracle_means) > 0:
                        ax1.errorbar(
                            N_over_D_vals, oracle_means, yerr=oracle_stds,
                            fmt="g--o", label="Oracle (Upper Bound)", linewidth=2, capsize=3, alpha=0.7
                        )
                else:
                    ax1.plot(N_over_D_vals, baseline_means, "r--o", label="Baseline (Window)", linewidth=2)
                    if len(oracle_means) > 0:
                        ax1.plot(N_over_D_vals, oracle_means, "g--o", label="Oracle (Upper Bound)", linewidth=2, alpha=0.7)

            color_idx += 1

    ax1.set_xlabel("N/D (Load Ratio)")
    ax1.set_ylabel("Retrieval Accuracy")
    ax1.set_title("Exp 7: LLM Integration - Accuracy vs N/D")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tokens/sec vs N (optional)
    color_idx = 0
    for (D, K, H) in sorted(data_by_config.keys()):
        config_data = data_by_config[(D, K, H)]
        N_vals = []
        tokens_means = []
        tokens_stds = []

        for N in sorted(config_data.keys()):
            vals = config_data[N]
            if len(vals["tokens_per_sec"]) > 0 and np.mean(vals["tokens_per_sec"]) > 0:
                N_vals.append(N)
                tokens_means.append(np.mean(vals["tokens_per_sec"]))
                tokens_stds.append(np.std(vals["tokens_per_sec"]))

        if len(N_vals) > 0:
            color = colors[color_idx % len(colors)]
            label = f"BBPM (D={D:,}, K={K}, H={H})"
            
            if num_seeds >= 3:
                ax2.errorbar(
                    N_vals, tokens_means, yerr=tokens_stds,
                    fmt=f"{color}-o", label=label, linewidth=2, capsize=3
                )
            else:
                ax2.plot(N_vals, tokens_means, f"{color}-o", label=label, linewidth=2)

            color_idx += 1

    ax2.set_xlabel("Number of Streamed Facts (N)")
    ax2.set_ylabel("Tokens per Second")
    ax2.set_title("Exp 7: Generation Speed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

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
