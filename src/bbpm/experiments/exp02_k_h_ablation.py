"""Experiment 02: K/H ablation study."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.addressing.hash_mix import mix64, u64
from bbpm.addressing.prp import u64_to_i64
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.retrieval import cosine_similarity
from bbpm.metrics.occupancy import block_occupancy
from bbpm.metrics.stats import summarize_groups
from bbpm.experiments.common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
    make_rng,
)
from bbpm.experiments.plotting import save_pdf, add_footer, plot_line_with_ci
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp02"
EXP_SLUG = "k_h_ablation"


def deterministic_u64_list(n: int, seed_u64: int) -> list[int]:
    """Generate deterministic list of uint64 values.
    
    Args:
        n: Number of values to generate
        seed_u64: Seed value (uint64)
        
    Returns:
        List of n uint64 values as Python ints
    """
    rng = make_rng(seed_u64)
    # Generate uint64 values using numpy RNG
    values = rng.integers(0, 2**64, size=n, dtype=np.uint64)
    return [int(v) for v in values]


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
                    
                    # Generate deterministic keys using (seed, K, H, N) combination
                    hx_seed = mix64(u64(seed) ^ u64(K) ^ u64(H) ^ u64(N))
                    hx_list = deterministic_u64_list(N, hx_seed)
                    
                    torch.manual_seed(seed)
                    values = torch.randn(N, d, device=device)
                    values = torch.nn.functional.normalize(values, p=2, dim=1)
                    
                    # Check self-collision exactly: check first 256 keys
                    # Self-collision = 1 if within any hash family, offsets repeat
                    check_n = min(N, 256)
                    self_collisions = 0
                    for hx in hx_list[:check_n]:
                        grouped = addresser.addresses_grouped(hx)
                        has_collision = False
                        for addresses_h in grouped:
                            offsets = []
                            for addr in addresses_h:
                                offset = addr % L
                                offsets.append(offset)
                            # Check if offsets repeat within this hash family
                            if len(offsets) != len(set(offsets)):
                                has_collision = True
                                break
                        if has_collision:
                            self_collisions += 1
                    
                    self_collision_rate = self_collisions / check_n if check_n > 0 else 0.0
                    
                    # Write all items (batch operation) and track addresses for collision analysis
                    hx_list_i64 = [u64_to_i64(hx) for hx in hx_list]
                    hx_tensor = torch.tensor(hx_list_i64, dtype=torch.long, device=device)
                    mem.write_batch(hx_tensor, values)
                    
                    # Get addresses for collision analysis (global addresses, not block IDs)
                    all_addresses = []
                    block_ids_list = []
                    for hx in hx_list:
                        addrs = addresser.addresses(hx)
                        all_addresses.extend(addrs)
                        # Extract block IDs for occupancy analysis
                        for addr in addrs:
                            block_ids_list.append(addr // L)
                    
                    # Compute cross-item collision rate from global addresses
                    unique_addresses = len(set(all_addresses))
                    total_writes = len(all_addresses)
                    collision_rate = (total_writes - unique_addresses) / total_writes if total_writes > 0 else 0.0
                    
                    # Compute block occupancy skew
                    occ_analysis = block_occupancy(block_ids_list, B)
                    max_mean_ratio = (
                        occ_analysis["max_occupancy"] / occ_analysis["mean_occupancy"]
                        if occ_analysis["mean_occupancy"] > 0
                        else 0.0
                    )
                    gini_skew = occ_analysis["gini_coefficient"]
                    
                    # Evaluate retrieval quality (batch operation)
                    test_n = min(N, 100)
                    test_hx = hx_list[:test_n]
                    test_hx_i64 = [u64_to_i64(hx) for hx in test_hx]
                    test_hx_tensor = torch.tensor(test_hx_i64, dtype=torch.long, device=device)
                    test_values = values[:test_n]
                    
                    retrieved_tensor = mem.read_batch(test_hx_tensor)  # [test_n, d]
                    
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
    
    # Summarize across seeds using summarize_groups
    summary = summarize_groups(
        raw_trials,
        ["K", "H", "N"],
        ["cosine", "collision_rate", "max_mean_ratio", "gini_skew", "self_collision_rate"]
    )
    
    # Generate figure with 2x2 grid showing multiple configs per panel
    fig = plt.figure(figsize=(14, 10))
    
    # Select configs to plot (at least 2-3 per panel)
    plot_configs = [
        (K, H) for K in [8, 32, 64] for H in [1, 4] 
        if (K, H) in [(k, h) for k in K_values for h in H_values]
    ][:6]  # Limit to 6 for readability
    
    # Panel 1: Accuracy (cosine) vs N for different K/H combinations
    ax1 = plt.subplot(2, 2, 1)
    for K, H in plot_configs:
        cosine_means = []
        cosine_lows = []
        cosine_highs = []
        for N in N_values:
            key = f"K={K}|H={H}|N={N}"
            if key in summary and "cosine" in summary[key]:
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
    
    # Panel 2: Collision rate vs N for multiple configs
    ax2 = plt.subplot(2, 2, 2)
    for K, H in plot_configs[:3]:  # Show 3 configs
        collision_means = []
        collision_lows = []
        collision_highs = []
        for N in N_values:
            key = f"K={K}|H={H}|N={N}"
            if key in summary and "collision_rate" in summary[key]:
                collision_means.append(summary[key]["collision_rate"]["mean"])
                collision_lows.append(summary[key]["collision_rate"]["ci95_low"])
                collision_highs.append(summary[key]["collision_rate"]["ci95_high"])
            else:
                collision_means.append(0)
                collision_lows.append(0)
                collision_highs.append(0)
        if any(collision_means):
            plot_line_with_ci(ax2, N_values, collision_means, collision_lows, collision_highs,
                            label=f"K={K}, H={H}", linestyle="-")
    ax2.set_xlabel("Number of Stored Items (N)")
    ax2.set_ylabel("Collision Rate")
    ax2.set_title("Cross-Item Collision Rate vs N")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Block skew (max/mean) vs N for multiple configs
    ax3 = plt.subplot(2, 2, 3)
    for K, H in plot_configs[:3]:  # Show 3 configs
        skew_means = []
        skew_lows = []
        skew_highs = []
        for N in N_values:
            key = f"K={K}|H={H}|N={N}"
            if key in summary and "max_mean_ratio" in summary[key]:
                skew_means.append(summary[key]["max_mean_ratio"]["mean"])
                skew_lows.append(summary[key]["max_mean_ratio"]["ci95_low"])
                skew_highs.append(summary[key]["max_mean_ratio"]["ci95_high"])
            else:
                skew_means.append(0)
                skew_lows.append(0)
                skew_highs.append(0)
        if any(skew_means):
            plot_line_with_ci(ax3, N_values, skew_means, skew_lows, skew_highs,
                            label=f"K={K}, H={H}", linestyle="-")
    ax3.set_xlabel("Number of Stored Items (N)")
    ax3.set_ylabel("Max/Mean Occupancy Ratio")
    ax3.set_title("Block Occupancy Skew vs N")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Self-collision rate (should be 0) for multiple configs
    ax4 = plt.subplot(2, 2, 4)
    for K, H in plot_configs[:3]:  # Show 3 configs
        self_coll_means = []
        self_coll_lows = []
        self_coll_highs = []
        for N in N_values:
            key = f"K={K}|H={H}|N={N}"
            if key in summary and "self_collision_rate" in summary[key]:
                self_coll_means.append(summary[key]["self_collision_rate"]["mean"])
                self_coll_lows.append(summary[key]["self_collision_rate"]["ci95_low"])
                self_coll_highs.append(summary[key]["self_collision_rate"]["ci95_high"])
            else:
                self_coll_means.append(0)
                self_coll_lows.append(0)
                self_coll_highs.append(0)
        if any(self_coll_means):
            plot_line_with_ci(ax4, N_values, self_coll_means, self_coll_lows, self_coll_highs,
                            label=f"K={K}, H={H}", linestyle="-")
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
