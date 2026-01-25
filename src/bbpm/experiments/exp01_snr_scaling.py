"""Experiment 01: SNR scaling with capacity."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bbpm.addressing.prp import u64_to_i64
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.stats import summarize_groups
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
        default=[500, 1000, 3000, 6000, 10000, 15000, 20000],  # Full sweep for paper runs
        # For quick CI/testing, use: --N_values 500 1000
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
            
            # Write all items (batch operation)
            print("writing...", end=" ", flush=True)
            hx_list_i64 = [u64_to_i64(hx) for hx in hx_list]
            hx_tensor = torch.tensor(hx_list_i64, dtype=torch.long, device=device)
            mem.write_batch(hx_tensor, values)
            
            # Evaluate retrieval on first min(N, 1000) items (batch operation)
            test_n = min(N, 1000)
            test_hx = hx_list[:test_n]
            test_hx_i64 = [u64_to_i64(hx) for hx in test_hx]
            test_hx_tensor = torch.tensor(test_hx_i64, dtype=torch.long, device=device)
            test_values = values[:test_n]
            
            print("reading...", end=" ", flush=True)
            retrieved_tensor = mem.read_batch(test_hx_tensor)  # [test_n, d]
            
            print("computing metrics...", end=" ", flush=True)
            
            # Compute error vector
            err = retrieved_tensor - test_values  # [test_n, d]
            
            # Compute noise variance: mean over items of variance across dimensions
            noise_var = err.var(dim=1, unbiased=False).mean().item()
            
            # Vectorize cosine similarity: batch computation
            cosines = F.cosine_similarity(test_values, retrieved_tensor, dim=1)  # [test_n]
            mean_cosine = cosines.mean().item()
            
            # Vectorize MSE: batch computation
            mses = F.mse_loss(test_values, retrieved_tensor, reduction='none').mean(dim=1)  # [test_n]
            mean_mse = mses.mean().item()
            
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
                "noise_var": noise_var,
                "occupancy": occupied_ratio,
            })
            print("done")
    
    # Summarize across seeds for each N using summarize_groups
    print("Summarizing results...")
    summary = summarize_groups(raw_trials, ["N"], ["cosine", "mse", "noise_var", "occupancy"])
    
    # Generate figure with 3 panels
    print("Generating figure...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    
    # Panel 1: Cosine similarity vs N
    cosine_means = []
    cosine_lows = []
    cosine_highs = []
    for N in N_values:
        n_key = f"N={N}"
        if n_key in summary and "cosine" in summary[n_key]:
            cosine_means.append(summary[n_key]["cosine"]["mean"])
            cosine_lows.append(summary[n_key]["cosine"]["ci95_low"])
            cosine_highs.append(summary[n_key]["cosine"]["ci95_high"])
        else:
            cosine_means.append(0)
            cosine_lows.append(0)
            cosine_highs.append(0)
    
    plot_line_with_ci(ax1, N_values, cosine_means, cosine_lows, cosine_highs,
                      label="Cosine Similarity", linestyle="-")
    ax1.set_xlabel("Number of Stored Items (N)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Exp01: Retrieval Quality vs Capacity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Noise variance vs N
    noise_var_means = []
    noise_var_lows = []
    noise_var_highs = []
    for N in N_values:
        n_key = f"N={N}"
        if n_key in summary and "noise_var" in summary[n_key]:
            noise_var_means.append(summary[n_key]["noise_var"]["mean"])
            noise_var_lows.append(summary[n_key]["noise_var"]["ci95_low"])
            noise_var_highs.append(summary[n_key]["noise_var"]["ci95_high"])
        else:
            noise_var_means.append(0)
            noise_var_lows.append(0)
            noise_var_highs.append(0)
    
    plot_line_with_ci(ax2, N_values, noise_var_means, noise_var_lows, noise_var_highs,
                      label="Noise Variance", linestyle="-")
    ax2.set_xlabel("Number of Stored Items (N)")
    ax2.set_ylabel("Noise Variance")
    ax2.set_title("Estimated Noise Variance vs Capacity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Occupancy vs N
    occ_means = []
    occ_lows = []
    occ_highs = []
    for N in N_values:
        n_key = f"N={N}"
        if n_key in summary and "occupancy" in summary[n_key]:
            occ_means.append(summary[n_key]["occupancy"]["mean"])
            occ_lows.append(summary[n_key]["occupancy"]["ci95_low"])
            occ_highs.append(summary[n_key]["occupancy"]["ci95_high"])
        else:
            occ_means.append(0)
            occ_lows.append(0)
            occ_highs.append(0)
    
    plot_line_with_ci(ax3, N_values, occ_means, occ_lows, occ_highs,
                      label="Occupancy Ratio", linestyle="-")
    ax3.set_xlabel("Number of Stored Items (N)")
    ax3.set_ylabel("Occupancy Ratio")
    ax3.set_title("Memory Occupancy vs Capacity")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
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
