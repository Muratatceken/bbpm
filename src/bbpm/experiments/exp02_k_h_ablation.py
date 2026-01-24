"""Experiment 02: K/H ablation study."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.retrieval import cosine_similarity
from bbpm.metrics.occupancy import block_occupancy
from bbpm.metrics.stats import mean_ci95, gini_coefficient
from bbpm.experiments.common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
)
from bbpm.experiments.plotting import save_pdf, add_footer, plot_line_with_ci
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp02"
EXP_SLUG = "k_h_ablation"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
    parser.add_argument(
        "--K_values",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help="K values to sweep",
    )
    parser.add_argument(
        "--H_values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="H values to sweep",
    )
    parser.add_argument(
        "--N_values",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="N values to sweep (x-axis)",
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run K/H ablation experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    device = ensure_device(args.device)
    dtype_str = args.dtype
    num_seeds = args.seeds
    K_values = args.K_values
    H_values = args.H_values
    N_values = args.N_values
    out_dir = args.out_dir
    
    # Fixed memory configuration
    B = 128  # Number of blocks
    L = 256  # Block size (must be >= max(K))
    d = 64  # Vector dimension
    
    # Filter K_values to respect K <= block_size
    K_values = [K for K in K_values if K <= L]
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    # Run trials
    for seed in seeds:
        seed_everything(seed)
        
        for K in K_values:
            for H in H_values:
                # Create memory config for this K, H
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
                
                # Create addresser for self-collision check
                addr_cfg = AddressConfig(
                    num_blocks=B,
                    block_size=L,
                    K=K,
                    H=H,
                    master_seed=42,
                )
                addresser = BlockAddress(addr_cfg)
                
                mem = BBPMMemory(mem_cfg)
                
                for N in N_values:
                    mem.reset()
                    
                    # Generate N random keys and values
                    random.seed(seed)
                    torch.manual_seed(seed)
                    hx_list = [random.randint(0, 2**64 - 1) for _ in range(N)]
                    values = torch.randn(N, d, device=device)
                    values = torch.nn.functional.normalize(values, p=2, dim=1)
                    
                    # Check self-collision (should be 0 with PRP)
                    self_collisions = 0
                    for hx in hx_list[:min(N, 100)]:  # Sample check
                        grouped = addresser.addresses_grouped(hx)
                        for addresses_h in grouped:
                            offsets = []
                            block_id = None
                            for addr in addresses_h:
                                if block_id is None:
                                    block_id = addr // L
                                offset = addr % L
                                offsets.append(offset)
                            # Check uniqueness
                            if len(offsets) != len(set(offsets)):
                                self_collisions += 1
                    
                    self_collision_rate = self_collisions / min(N, 100) if N > 0 else 0.0
                    
                    # Write all items and track addresses for collision analysis
                    all_addresses = []
                    block_ids_list = []
                    for hx, v in zip(hx_list, values):
                        mem.write(hx, v)
                        # Get addresses for collision analysis
                        addrs = addresser.addresses(hx)
                        all_addresses.extend(addrs)
                        # Extract block IDs
                        for addr in addrs:
                            block_ids_list.append(addr // L)
                    
                    # Compute cross-item collision rate
                    unique_addresses = len(set(all_addresses))
                    total_writes = len(all_addresses)
                    collision_rate = 1.0 - (unique_addresses / total_writes) if total_writes > 0 else 0.0
                    
                    # Compute block occupancy skew
                    occ_analysis = block_occupancy(block_ids_list, B)
                    max_mean_ratio = (
                        occ_analysis["max_occupancy"] / occ_analysis["mean_occupancy"]
                        if occ_analysis["mean_occupancy"] > 0
                        else 0.0
                    )
                    gini_skew = occ_analysis["gini_coefficient"]
                    
                    # Evaluate retrieval quality
                    test_n = min(N, 100)
                    test_hx = hx_list[:test_n]
                    test_values = values[:test_n]
                    
                    retrieved = []
                    for hx in test_hx:
                        r = mem.read(hx)
                        retrieved.append(r)
                    retrieved_tensor = torch.stack(retrieved)
                    
                    cosines = [cosine_similarity(v, r) for v, r in zip(test_values, retrieved_tensor)]
                    mean_cosine = np.mean(cosines)
                    
                    # Record trial
                    raw_trials.append({
                        "seed": seed,
                        "K": K,
                        "H": H,
                        "N": N,
                        "self_collision_rate": self_collision_rate,
                        "collision_rate": collision_rate,
                        "max_mean_ratio": max_mean_ratio,
                        "gini_skew": gini_skew,
                        "cosine": mean_cosine,
                    })
    
    # Summarize across seeds
    summary = {}
    for K in K_values:
        for H in H_values:
            for N in N_values:
                key = f"K_{K}_H_{H}_N_{N}"
                trials = [t for t in raw_trials if t["K"] == K and t["H"] == H and t["N"] == N]
                
                if not trials:
                    continue
                
                cosine_vals = [t["cosine"] for t in trials]
                collision_vals = [t["collision_rate"] for t in trials]
                skew_vals = [t["max_mean_ratio"] for t in trials]
                gini_vals = [t["gini_skew"] for t in trials]
                self_coll_vals = [t["self_collision_rate"] for t in trials]
                
                cosine_stats = mean_ci95(cosine_vals)
                collision_stats = mean_ci95(collision_vals)
                skew_stats = mean_ci95(skew_vals)
                gini_stats = mean_ci95(gini_vals)
                self_coll_stats = mean_ci95(self_coll_vals)
                
                cosine_mean = cosine_stats["mean"]
                cosine_lo = cosine_stats["ci95_low"]
                cosine_hi = cosine_stats["ci95_high"]
                cosine_std = cosine_stats["std"]
                
                collision_mean = collision_stats["mean"]
                collision_lo = collision_stats["ci95_low"]
                collision_hi = collision_stats["ci95_high"]
                collision_std = collision_stats["std"]
                
                skew_mean = skew_stats["mean"]
                skew_lo = skew_stats["ci95_low"]
                skew_hi = skew_stats["ci95_high"]
                skew_std = skew_stats["std"]
                
                gini_mean = gini_stats["mean"]
                gini_lo = gini_stats["ci95_low"]
                gini_hi = gini_stats["ci95_high"]
                gini_std = gini_stats["std"]
                
                self_coll_mean = self_coll_stats["mean"]
                self_coll_lo = self_coll_stats["ci95_low"]
                self_coll_hi = self_coll_stats["ci95_high"]
                self_coll_std = self_coll_stats["std"]
                
                summary[key] = {
                    "cosine": {
                        "mean": cosine_mean,
                        "ci95_low": cosine_lo,
                        "ci95_high": cosine_hi,
                        "std": cosine_std,
                    },
                    "collision_rate": {
                        "mean": collision_mean,
                        "ci95_low": collision_lo,
                        "ci95_high": collision_hi,
                        "std": collision_std,
                    },
                    "max_mean_ratio": {
                        "mean": skew_mean,
                        "ci95_low": skew_lo,
                        "ci95_high": skew_hi,
                        "std": skew_std,
                    },
                    "gini_skew": {
                        "mean": gini_mean,
                        "ci95_low": gini_lo,
                        "ci95_high": gini_hi,
                        "std": gini_std,
                    },
                    "self_collision_rate": {
                        "mean": self_coll_mean,
                        "ci95_low": self_coll_lo,
                        "ci95_high": self_coll_hi,
                        "std": self_coll_std,
                    },
                }
    
    # Generate figure with multiple panels
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: Accuracy (cosine) vs N for different K/H combinations
    ax1 = plt.subplot(2, 2, 1)
    # Plot a subset of K/H combinations for clarity
    plot_configs = [
        (K, H) for K in [8, 32, 64] for H in [1, 4] if (K, H) in [(k, h) for k in K_values for h in H_values]
    ]
    for K, H in plot_configs[:6]:  # Limit to 6 lines for readability
        cosine_means = []
        cosine_lows = []
        cosine_highs = []
        for N in N_values:
            key = f"K_{K}_H_{H}_N_{N}"
            if key in summary:
                cosine_means.append(summary[key]["cosine"]["mean"])
                cosine_lows.append(summary[key]["cosine"]["ci95_low"])
                cosine_highs.append(summary[key]["cosine"]["ci95_high"])
            else:
                cosine_means.append(0)
                cosine_lows.append(0)
                cosine_highs.append(0)
        plot_line_with_ci(ax1, N_values, cosine_means, cosine_lows, cosine_highs,
                          label=f"K={K}, H={H}", linestyle="-")
    ax1.set_xlabel("Number of Stored Items (N)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Retrieval Quality vs N (K/H Ablation)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Collision rate vs N
    ax2 = plt.subplot(2, 2, 2)
    # Plot for K=32, H=4 as example
    K_example, H_example = 32, 4
    if (K_example, H_example) in [(k, h) for k in K_values for h in H_values]:
        collision_means = []
        collision_lows = []
        collision_highs = []
        for N in N_values:
            key = f"K_{K_example}_H_{H_example}_N_{N}"
            if key in summary:
                collision_means.append(summary[key]["collision_rate"]["mean"])
                collision_lows.append(summary[key]["collision_rate"]["ci95_low"])
                collision_highs.append(summary[key]["collision_rate"]["ci95_high"])
        if collision_means:
            plot_line_with_ci(ax2, N_values, collision_means, collision_lows, collision_highs,
                            label=f"Collision Rate (K={K_example}, H={H_example})", linestyle="-")
    ax2.set_xlabel("Number of Stored Items (N)")
    ax2.set_ylabel("Collision Rate")
    ax2.set_title("Cross-Item Collision Rate vs N")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Block skew (max/mean) vs N
    ax3 = plt.subplot(2, 2, 3)
    if (K_example, H_example) in [(k, h) for k in K_values for h in H_values]:
        skew_means = []
        skew_lows = []
        skew_highs = []
        for N in N_values:
            key = f"K_{K_example}_H_{H_example}_N_{N}"
            if key in summary:
                skew_means.append(summary[key]["max_mean_ratio"]["mean"])
                skew_lows.append(summary[key]["max_mean_ratio"]["ci95_low"])
                skew_highs.append(summary[key]["max_mean_ratio"]["ci95_high"])
        if skew_means:
            plot_line_with_ci(ax3, N_values, skew_means, skew_lows, skew_highs,
                            label=f"Max/Mean Ratio (K={K_example}, H={H_example})", linestyle="-")
    ax3.set_xlabel("Number of Stored Items (N)")
    ax3.set_ylabel("Max/Mean Occupancy Ratio")
    ax3.set_title("Block Occupancy Skew vs N")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Self-collision rate (should be 0)
    ax4 = plt.subplot(2, 2, 4)
    if (K_example, H_example) in [(k, h) for k in K_values for h in H_values]:
        self_coll_means = []
        self_coll_lows = []
        self_coll_highs = []
        for N in N_values:
            key = f"K_{K_example}_H_{H_example}_N_{N}"
            if key in summary:
                self_coll_means.append(summary[key]["self_collision_rate"]["mean"])
                self_coll_lows.append(summary[key]["self_collision_rate"]["ci95_low"])
                self_coll_highs.append(summary[key]["self_collision_rate"]["ci95_high"])
        if self_coll_means:
            plot_line_with_ci(ax4, N_values, self_coll_means, self_coll_lows, self_coll_highs,
                            label=f"Self-Collision Rate (K={K_example}, H={H_example})", linestyle="-")
    ax4.set_xlabel("Number of Stored Items (N)")
    ax4.set_ylabel("Self-Collision Rate")
    ax4.set_title("Self-Collision Rate vs N (should be 0)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label="Expected (0)")
    
    add_footer(fig, EXP_ID)
    
    # Save outputs
    metrics_path, figure_path = make_output_paths(out_dir, EXP_ID, EXP_SLUG)
    
    config_dict = {
        "B": B,
        "L": L,
        "d": d,
        "K_values": K_values,
        "H_values": H_values,
        "N_values": N_values,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "K/H ablation",
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
