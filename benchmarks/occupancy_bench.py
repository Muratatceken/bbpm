"""Benchmark: occupancy distributions and skew."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bbpm import BBPMMemoryFloat, GlobalAffineHash, get_device, set_global_seed
from bbpm.addressing.bbpm_addressing import BBPMAddressing
from bbpm.hashing.diagnostics import gini_load, max_load, occupancy_summary
from bbpm.utils import get_logger


def benchmark_occupancy(D, N_values, K, H, device="cpu", seed=42):
    """Benchmark occupancy distributions."""
    results = []

    global_hash = GlobalAffineHash(D, seed=seed)
    # Use standard block_size=1024 (D must be divisible by 1024)
    block_size = 1024
    block_addressing = BBPMAddressing(D, block_size=block_size, seed=seed, num_hashes=H, K=K)

    for N in N_values:
        logger.info(f"Benchmarking N={N}")

        keys = torch.arange(N, device=device)

        # Global hash
        indices_global = global_hash.indices(keys, K, H).flatten()
        occ_global = occupancy_summary(indices_global, D)
        max_global = max_load(indices_global, D)
        gini_global = gini_load(indices_global, D)

        # BBPMAddressing (PRP-based)
        indices_block = block_addressing.indices(keys, K, H).flatten()
        occ_block = occupancy_summary(indices_block, D)
        max_block = max_load(indices_block, D)
        gini_block = gini_load(indices_block, D)

        results.append({
            "N": N,
            "global": {
                "max_load": max_global,
                "gini": gini_global,
                "mean_occupancy": occ_global["mean_load"],
                "std_occupancy": occ_global["std_load"],
                "occupancy_summary": occ_global,
            },
            "block": {
                "max_load": max_block,
                "gini": gini_block,
                "mean_occupancy": occ_block["mean_load"],
                "std_occupancy": occ_block["std_load"],
                "occupancy_summary": occ_block,
            },
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy benchmark")
    parser.add_argument("--out", type=Path, default=Path("../../results/benchmarks/occupancy.json"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--D", type=int, default=1000448)  # Divisible by 1024
    parser.add_argument("--N", type=int, nargs="+", default=[10000, 50000, 100000, 500000])
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--H", type=int, default=1)

    args = parser.parse_args()

    logger = get_logger("occupancy_bench")
    set_global_seed(42)

    results = benchmark_occupancy(
        D=args.D,
        N_values=args.N,
        K=args.K,
        H=args.H,
        device=args.device,
        seed=42,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.out}")
