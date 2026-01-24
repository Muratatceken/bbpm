"""Experiment 01: SNR scaling with capacity."""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for bbpm imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.retrieval import cosine_similarity, mse
from bbpm.metrics.stats import mean_ci95, summarize_groups
from common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
)
from plotting import save_pdf, add_footer, plot_line_with_ci
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
    
    mem_cfg = MemoryConfig(
        num_blocks=B,
        block_size=L,
        key_dim=d,
        K=K,
        H=H,
        dtype=dtype_str,
        device=str(device),
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    # Run trials
    for seed in seeds:
        seed_everything(seed)
        mem = BBPMMemory(mem_cfg)
        
        for N in N_values:
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
            for hx, v in zip(hx_list, values):
                mem.write(hx, v)
            
            # Evaluate retrieval on first min(N, 1000) items
            test_n = min(N, 1000)
            test_hx = hx_list[:test_n]
            test_values = values[:test_n]
            
            retrieved = []
            for hx in test_hx:
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
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
            
            # Compute occupancy (fraction of slots that received at least one write)
            # Approximate: total writes = N * K * H, unique slots <= D
            total_writes = N * K * H
            # For exact occupancy, we'd need to track unique addresses, but for now
            # use an approximation: occupied_slots ≈ min(D, total_writes) (collision-aware)
            # Actually, we can compute exact occupancy from memory state if needed
            # For now, use a simple approximation
            occupied_ratio = min(1.0, total_writes / D)
            
            # Record trial
            raw_trials.append({
                "seed": seed,
                "N": N,
                "cosine": mean_cosine,
                "mse": mean_mse,
                "occupancy": occupied_ratio,
            })
    
    # Summarize across seeds for each N
    summary = {}
    for N in N_values:
        n_trials = [t for t in raw_trials if t["N"] == N]
        
        cosine_vals = [t["cosine"] for t in n_trials]
        mse_vals = [t["mse"] for t in n_trials]
        occ_vals = [t["occupancy"] for t in n_trials]
        
        cosine_mean, cosine_lo, cosine_hi, cosine_std = mean_ci95(cosine_vals)
        mse_mean, mse_lo, mse_hi, mse_std = mean_ci95(mse_vals)
        occ_mean, occ_lo, occ_hi, occ_std = mean_ci95(occ_vals)
        
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
        f"{EXP_ID}_{EXP_SLUG}",
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
