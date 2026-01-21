"""Analyze experiment 4 results and generate figure."""

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

    ctx_lens = results["context_lengths"]
    kv_mem = results["kv_memory_gb"]
    bbpm_mem = results["bbpm_memory_gb"]
    oom_at = results.get("kv_oom")

    plt.figure(figsize=(10, 6))
    plt.plot(ctx_lens[:len(kv_mem)], kv_mem, "r-o", label="KV Cache (O(N))", linewidth=2)
    plt.plot(ctx_lens[:len(bbpm_mem)], bbpm_mem, "b--", label="BBPM (O(1) Memory)", linewidth=2)

    if oom_at:
        plt.axvline(oom_at, color="k", linestyle=":", label=f"OOM @ {oom_at} tokens")

    plt.xlabel("Context Length (T)")
    plt.ylabel("GPU Memory (GB)")
    plt.title("Memory Scaling: Transformer KV Cache vs BBPM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "kv_vs_bbpm_memory.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exp04 results")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "exp04_kv_memory_scaling" / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "figures" / "exp04_kv_memory_scaling")

    args = parser.parse_args()
    analyze(args.results, args.outdir)
