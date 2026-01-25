"""Experiment 01: SNR scaling with capacity."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.retrieval import cosine_similarity, mse
from bbpm.metrics.stats import mean_ci95, summarize_groups
from bbpm.experiments.common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
)
from bbpm.experiments.plotting import save_pdf, add_footer, plot_line_with_ci
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp01"
EXP_SLUG = "snr_scaling"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
    parser.add_argument(
        "--N_values",
        type=int,
        nargs="+",
        default=[500, 1000, 3000, 6000, 10000, 15000, 20000],
        help="N values to sweep (number of stored items)",
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run SNR scaling experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    # Extract config
    device = ensure_device(args.device)
    dtype_str = args.dtype
    num_seeds = args.seeds
    N_values = args.N_values
    out_dir = args.out_dir
    
    # Fixed memory configuration
    D = 2**20  # 1M slots
    d = 64  # Vector dimension
    B = 2**14  # 16384 blocks
    L = 256  # Block size (power of 2)
    K = 32  # Slots per item
    H = 4  # Hash families
    
    # Use count_normalized mode to enable occupancy tracking via mem.stats()
    mem_cfg = MemoryConfig(
        num_blocks=B,
        block_size=L,
        key_dim=d,
        K=K,
        H=H,
        dtype=dtype_str,
        device=str(device),
        normalize_values="none",
        read_mode="count_normalized",  # Enables counts tensor for occupancy tracking
        master_seed=42,
    )
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    print(f"Running {len(seeds)} seeds, {len(N_values)} N values each...")
    
    # Run trials
    for seed_idx, seed in enumerate(seeds):
        print(f"  Seed {seed_idx + 1}/{num_seeds} (seed={seed})...")
        seed_everything(seed)
        mem = BBPMMemory(mem_cfg)
        
        for n_idx, N in enumerate(N_values):
            print(f"    N={N} ({n_idx + 1}/{len(N_values)})...", end=" ", flush=True)
            # Reset memory for this N
            mem.reset()
            
            # Generate N random keys and values
            random.seed(seed)
            torch.manual_seed(seed)
            hx_list = [random.randint(0, 2**64 - 1) for _ in range(N)]
            values = torch.randn(N, d, device=device)
            # Normalize values to unit length (signal power = 1)
            values = torch.nn.functional.normalize(values, p=2, dim=1)
            
            # Write all items
            print("writing...", end=" ", flush=True)
            for hx, v in zip(hx_list, values):
                mem.write(hx, v)
            
            # Evaluate retrieval on first min(N, 1000) items
            test_n = min(N, 1000)
            test_hx = hx_list[:test_n]
            test_values = values[:test_n]
            
            print("reading...", end=" ", flush=True)
            retrieved = []
            for hx in test_hx:
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            print("computing metrics...", end=" ", flush=True)
            
            # Compute metrics
            cosines = []
            mses = []
            for v, r in zip(test_values, retrieved_tensor):
                cos = cosine_similarity(v, r)
                mse_val = mse(v, r)
                cosines.append(cos)
                mses.append(mse_val)
            
            mean_cosine = np.mean(cosines)
            mean_mse = np.mean(mses)
            
            # Compute occupancy using mem.stats() (requires count_normalized mode)
            # This gives exact occupancy: fraction of slots with count > 0
            stats = mem.stats()
            if "occupied_slots" in stats:
                occupied_slots = stats["occupied_slots"]
                occupied_ratio = occupied_slots / D
            else:
                # Fallback: approximate occupancy if counts not available
                total_writes = N * K * H
                occupied_ratio = min(1.0, total_writes / D)
            
            # Record trial
            raw_trials.append({
                "seed": seed,
                "N": N,
                "cosine": mean_cosine,
                "mse": mean_mse,
                "occupancy": occupied_ratio,
            })
            print("done")
    
    # Summarize across seeds for each N
    print("Summarizing results...")
    summary = {}
    for N in N_values:
        n_trials = [t for t in raw_trials if t["N"] == N]
        
        cosine_vals = [t["cosine"] for t in n_trials]
        mse_vals = [t["mse"] for t in n_trials]
        occ_vals = [t["occupancy"] for t in n_trials]
        
        cosine_stats = mean_ci95(cosine_vals)
        mse_stats = mean_ci95(mse_vals)
        occ_stats = mean_ci95(occ_vals)
        
        cosine_mean = cosine_stats["mean"]
        cosine_lo = cosine_stats["ci95_low"]
        cosine_hi = cosine_stats["ci95_high"]
        cosine_std = cosine_stats["std"]
        
        mse_mean = mse_stats["mean"]
        mse_lo = mse_stats["ci95_low"]
        mse_hi = mse_stats["ci95_high"]
        mse_std = mse_stats["std"]
        
        occ_mean = occ_stats["mean"]
        occ_lo = occ_stats["ci95_low"]
        occ_hi = occ_stats["ci95_high"]
        occ_std = occ_stats["std"]
        
        summary[f"N_{N}"] = {
            "cosine": {
                "mean": cosine_mean,
                "ci95_low": cosine_lo,
                "ci95_high": cosine_hi,
                "std": cosine_std,
            },
            "mse": {
                "mean": mse_mean,
                "ci95_low": mse_lo,
                "ci95_high": mse_hi,
                "std": mse_std,
            },
            "occupancy": {
                "mean": occ_mean,
                "ci95_low": occ_lo,
                "ci95_high": occ_hi,
                "std": occ_std,
            },
        }
    
    # Generate figure
    print("Generating figure...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Panel 1: Cosine similarity vs N
    cosine_means = []
    cosine_lows = []
    cosine_highs = []
    for N in N_values:
        n_summary = summary[f"N_{N}"]
        cosine_means.append(n_summary["cosine"]["mean"])
        cosine_lows.append(n_summary["cosine"]["ci95_low"])
        cosine_highs.append(n_summary["cosine"]["ci95_high"])
    
    plot_line_with_ci(ax1, N_values, cosine_means, cosine_lows, cosine_highs,
                      label="Cosine Similarity", linestyle="-")
    ax1.set_xlabel("Number of Stored Items (N)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Exp01: Retrieval Quality vs Capacity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Occupancy vs N
    occ_means = []
    occ_lows = []
    occ_highs = []
    for N in N_values:
        n_summary = summary[f"N_{N}"]
        occ_means.append(n_summary["occupancy"]["mean"])
        occ_lows.append(n_summary["occupancy"]["ci95_low"])
        occ_highs.append(n_summary["occupancy"]["ci95_high"])
    
    plot_line_with_ci(ax2, N_values, occ_means, occ_lows, occ_highs,
                      label="Occupancy Ratio", linestyle="-")
    ax2.set_xlabel("Number of Stored Items (N)")
    ax2.set_ylabel("Occupancy Ratio")
    ax2.set_title("Memory Occupancy vs Capacity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    add_footer(fig, EXP_ID)
    
    # Save outputs
    print("Saving outputs...")
    metrics_path, figure_path = make_output_paths(out_dir, EXP_ID, EXP_SLUG)
    
    config_dict = {
        "D": D,
        "d": d,
        "B": B,
        "L": L,
        "K": K,
        "H": H,
        "N_values": N_values,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "SNR scaling",
        config_dict,
        seeds,
        raw_trials,
        summary,
    )
    
    save_pdf(fig, figure_path)
    
    return {
        "metrics_path": str(metrics_path),
        "figure_path": str(figure_path),
    }
