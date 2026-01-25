"""Experiment 06: Occupancy skew with non-uniform load."""

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
from bbpm.metrics.occupancy import block_occupancy
from bbpm.metrics.stats import mean_ci95, gini_coefficient
from bbpm.experiments.common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
    make_rng,
)
from bbpm.experiments.plotting import save_pdf, add_footer, plot_line_with_ci
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp06"
EXP_SLUG = "occupancy_skew"


def sample_zipf(vocab_size: int, num_samples: int, s: float, seed: int) -> list[int]:
    """Sample from Zipf distribution using local RNG.
    
    Args:
        vocab_size: Vocabulary size
        num_samples: Number of samples
        s: Zipf exponent (s > 1)
        seed: Random seed
        
    Returns:
        List of token IDs
    """
    rng = make_rng(seed)
    # Generate Zipf-distributed samples
    # Simple approximation: use power law
    probs = np.array([1.0 / (i + 1) ** s for i in range(vocab_size)])
    probs = probs / probs.sum()
    
    return rng.choice(vocab_size, size=num_samples, p=probs).tolist()


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments.
    
    Note: For paper runs, use --seeds 5 or more for statistical significance.
    """
    parser.add_argument(
        "--N",
        type=int,
        default=5000,
        help="Number of items to write",
    )
    parser.add_argument(
        "--s_values",
        type=float,
        nargs="+",
        default=[1.0, 1.2, 1.5, 2.0],
        help="Zipf exponent values to sweep",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size for Zipf distribution",
    )
    parser.add_argument(
        "--token_ids_path",
        type=str,
        default=None,
        help="Optional path to token IDs file (if absent, generate synthetic)",
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run occupancy skew experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    device = ensure_device(args.device)
    dtype_str = args.dtype
    num_seeds = args.seeds
    N = args.N
    s_values = args.s_values
    vocab_size = args.vocab_size
    token_ids_path = args.token_ids_path
    out_dir = args.out_dir
    
    # Fixed memory configuration
    B = 128  # Number of blocks
    L = 256  # Block size
    d = 64
    K = 32
    H = 4
    
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
    
    addr_cfg = AddressConfig(
        num_blocks=B,
        block_size=L,
        K=K,
        H=H,
        master_seed=42,
    )
    addresser = BlockAddress(addr_cfg)
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    # Run trials
    for seed in seeds:
        seed_everything(seed)
        mem = BBPMMemory(mem_cfg)
        
        # Mode 1: Uniform random keys
        mem.reset()
        random.seed(seed)
        uniform_hx_list = [random.randint(0, 2**64 - 1) for _ in range(N)]
        values = torch.randn(N, d, device=device)
        values = torch.nn.functional.normalize(values, p=2, dim=1)
        
        all_addresses_uniform = []
        all_block_ids_uniform = []
        # Write uniform keys (batch operation)
        uniform_hx_i64 = [u64_to_i64(hx) for hx in uniform_hx_list]
        uniform_hx_tensor = torch.tensor(uniform_hx_i64, dtype=torch.long, device=device)
        mem.write_batch(uniform_hx_tensor, values)
            addrs = addresser.addresses(hx)  # Global addresses
            all_addresses_uniform.extend(addrs)
            for addr in addrs:
                all_block_ids_uniform.append(addr // L)
        
        occ_uniform = block_occupancy(all_block_ids_uniform, B)
        
        # Collision rate from global addresses
        unique_addresses = len(set(all_addresses_uniform))
        total_writes = len(all_addresses_uniform)
        collision_rate = (total_writes - unique_addresses) / total_writes if total_writes > 0 else 0.0
        
        raw_trials.append({
            "seed": seed,
            "mode": "uniform",
            "s": None,
            "max_mean_ratio": occ_uniform["max_occupancy"] / occ_uniform["mean_occupancy"] if occ_uniform["mean_occupancy"] > 0 else 0.0,
            "gini": occ_uniform["gini_coefficient"],
            "collision_rate": collision_rate,
            "block_counts": occ_uniform["counts_per_block"],
        })
        
        # Mode 2: Zipf-distributed keys
        for s in s_values:
            mem.reset()
            zipf_token_ids = sample_zipf(vocab_size, N, s, seed)
            # Convert token IDs to hx (deterministic mapping using mix64)
            zipf_hx_list = []
            for token_id in zipf_token_ids:
                # Deterministic uint64 hash from token_id and seed
                hx = mix64(u64(token_id) ^ u64(seed))
                zipf_hx_list.append(hx)
            
            values = torch.randn(N, d, device=device)
            values = torch.nn.functional.normalize(values, p=2, dim=1)
            
            all_addresses_zipf = []
            all_block_ids_zipf = []
            # Write Zipf keys (batch operation)
            zipf_hx_i64 = [u64_to_i64(hx) for hx in zipf_hx_list]
            zipf_hx_tensor = torch.tensor(zipf_hx_i64, dtype=torch.long, device=device)
            mem.write_batch(zipf_hx_tensor, values)
                addrs = addresser.addresses(hx)  # Global addresses
                all_addresses_zipf.extend(addrs)
                for addr in addrs:
                    all_block_ids_zipf.append(addr // L)
            
            occ_zipf = block_occupancy(all_block_ids_zipf, B)
            
            # Collision rate from global addresses
            unique_addresses = len(set(all_addresses_zipf))
            total_writes = len(all_addresses_zipf)
            collision_rate = (total_writes - unique_addresses) / total_writes if total_writes > 0 else 0.0
            
            raw_trials.append({
                "seed": seed,
                "mode": "zipf",
                "s": s,
                "max_mean_ratio": occ_zipf["max_occupancy"] / occ_zipf["mean_occupancy"] if occ_zipf["mean_occupancy"] > 0 else 0.0,
                "gini": occ_zipf["gini_coefficient"],
                "collision_rate": collision_rate,
                "block_counts": occ_zipf["counts_per_block"],
            })
        
        # Mode 3: Token-ID mode (seed-independent keying)
        mem.reset()
        if token_ids_path and Path(token_ids_path).exists():
            # Load from file
            token_ids = [int(line.strip()) for line in open(token_ids_path)]
            token_ids = token_ids[:N]
        else:
            # Generate synthetic token IDs from unigram-like Zipf with fixed s=1.1
            token_ids = sample_zipf(vocab_size, N, s=1.1, seed=seed)
        
        # Map token_id -> hx = mix64(u64(token_id)) (seed independent)
        token_id_hx_list = [mix64(u64(tid)) for tid in token_ids]
        
        values = torch.randn(N, d, device=device)
        values = torch.nn.functional.normalize(values, p=2, dim=1)
        
        all_addresses_token = []
        all_block_ids_token = []
        # Write token-ID keys (batch operation)
        token_id_hx_i64 = [u64_to_i64(hx) for hx in token_id_hx_list]
        token_id_hx_tensor = torch.tensor(token_id_hx_i64, dtype=torch.long, device=device)
        mem.write_batch(token_id_hx_tensor, values)
            addrs = addresser.addresses(hx)  # Global addresses
            all_addresses_token.extend(addrs)
            for addr in addrs:
                all_block_ids_token.append(addr // L)
        
        occ_token = block_occupancy(all_block_ids_token, B)
        
        # Collision rate from global addresses
        unique_addresses = len(set(all_addresses_token))
        total_writes = len(all_addresses_token)
        collision_rate = (total_writes - unique_addresses) / total_writes if total_writes > 0 else 0.0
        
        raw_trials.append({
            "seed": seed,
            "mode": "token_id",
            "s": None,
            "max_mean_ratio": occ_token["max_occupancy"] / occ_token["mean_occupancy"] if occ_token["mean_occupancy"] > 0 else 0.0,
            "gini": occ_token["gini_coefficient"],
            "collision_rate": collision_rate,
            "block_counts": occ_token["counts_per_block"],
        })
    
    # Summarize
    summary = {}
    
    # Uniform mode
    uniform_trials = [t for t in raw_trials if t["mode"] == "uniform"]
    if uniform_trials:
        max_mean_vals = [t["max_mean_ratio"] for t in uniform_trials]
        gini_vals = [t["gini"] for t in uniform_trials]
        collision_vals = [t["collision_rate"] for t in uniform_trials]
        
        # Aggregate block_counts across seeds
        all_block_counts = [t["block_counts"] for t in uniform_trials]
        avg_block_counts = np.mean(all_block_counts, axis=0).tolist()
        
        summary["uniform"] = {
            "max_mean_ratio": {
                "mean": np.mean(max_mean_vals),
                "ci95_low": mean_ci95(max_mean_vals)["ci95_low"],
                "ci95_high": mean_ci95(max_mean_vals)["ci95_high"],
                "std": np.std(max_mean_vals),
            },
            "gini": {
                "mean": np.mean(gini_vals),
                "ci95_low": mean_ci95(gini_vals)["ci95_low"],
                "ci95_high": mean_ci95(gini_vals)["ci95_high"],
                "std": np.std(gini_vals),
            },
            "collision_rate": {
                "mean": np.mean(collision_vals),
                "ci95_low": mean_ci95(collision_vals)["ci95_low"],
                "ci95_high": mean_ci95(collision_vals)["ci95_high"],
                "std": np.std(collision_vals),
            },
            "block_counts": avg_block_counts,
        }
    
    # Zipf modes
    for s in s_values:
        zipf_trials = [t for t in raw_trials if t["mode"] == "zipf" and t["s"] == s]
        if zipf_trials:
            max_mean_vals = [t["max_mean_ratio"] for t in zipf_trials]
            gini_vals = [t["gini"] for t in zipf_trials]
            collision_vals = [t["collision_rate"] for t in zipf_trials]
            
            # Aggregate block_counts across seeds (average histogram)
            all_block_counts = [t["block_counts"] for t in zipf_trials]
            avg_block_counts = np.mean(all_block_counts, axis=0).tolist()
            
            summary[f"zipf_s_{s}"] = {
                "max_mean_ratio": {
                    "mean": np.mean(max_mean_vals),
                    "ci95_low": mean_ci95(max_mean_vals)["ci95_low"],
                    "ci95_high": mean_ci95(max_mean_vals)["ci95_high"],
                    "std": np.std(max_mean_vals),
                },
                "gini": {
                    "mean": np.mean(gini_vals),
                    "ci95_low": mean_ci95(gini_vals)["ci95_low"],
                    "ci95_high": mean_ci95(gini_vals)["ci95_high"],
                    "std": np.std(gini_vals),
                },
                "collision_rate": {
                    "mean": np.mean(collision_vals),
                    "ci95_low": mean_ci95(collision_vals)["ci95_low"],
                    "ci95_high": mean_ci95(collision_vals)["ci95_high"],
                    "std": np.std(collision_vals),
                },
                "block_counts": avg_block_counts,
            }
    
    # Token-ID mode
    token_id_trials = [t for t in raw_trials if t["mode"] == "token_id"]
    if token_id_trials:
        max_mean_vals = [t["max_mean_ratio"] for t in token_id_trials]
        gini_vals = [t["gini"] for t in token_id_trials]
        collision_vals = [t["collision_rate"] for t in token_id_trials]
        
        # Aggregate block_counts
        all_block_counts = [t["block_counts"] for t in token_id_trials]
        avg_block_counts = np.mean(all_block_counts, axis=0).tolist()
        
        summary["token_id"] = {
            "max_mean_ratio": {
                "mean": np.mean(max_mean_vals),
                "ci95_low": mean_ci95(max_mean_vals)["ci95_low"],
                "ci95_high": mean_ci95(max_mean_vals)["ci95_high"],
                "std": np.std(max_mean_vals),
            },
            "gini": {
                "mean": np.mean(gini_vals),
                "ci95_low": mean_ci95(gini_vals)["ci95_low"],
                "ci95_high": mean_ci95(gini_vals)["ci95_high"],
                "std": np.std(gini_vals),
            },
            "collision_rate": {
                "mean": np.mean(collision_vals),
                "ci95_low": mean_ci95(collision_vals)["ci95_low"],
                "ci95_high": mean_ci95(collision_vals)["ci95_high"],
                "std": np.std(collision_vals),
            },
            "block_counts": avg_block_counts,
        }
    
    # Generate figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Occupancy histogram for uniform, zipf(s=1.5), token-id
    if "uniform" in summary and "block_counts" in summary["uniform"]:
        uniform_counts = summary["uniform"]["block_counts"]
        ax1.hist(range(len(uniform_counts)), weights=uniform_counts, bins=len(uniform_counts),
                alpha=0.6, label="Uniform", density=False)
    
    if "zipf_s_1.5" in summary and "block_counts" in summary["zipf_s_1.5"]:
        zipf_counts = summary["zipf_s_1.5"]["block_counts"]
        ax1.hist(range(len(zipf_counts)), weights=zipf_counts, bins=len(zipf_counts),
                alpha=0.6, label="Zipf (s=1.5)", density=False)
    
    if "token_id" in summary and "block_counts" in summary["token_id"]:
        token_counts = summary["token_id"]["block_counts"]
        ax1.hist(range(len(token_counts)), weights=token_counts, bins=len(token_counts),
                alpha=0.6, label="Token-ID", density=False)
    
    ax1.set_xlabel("Block ID")
    ax1.set_ylabel("Occupancy Count")
    ax1.set_title("Block Occupancy Histogram")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Panel 2: Collision rate and Gini vs Zipf exponent (with uniform as separate point)
    # Fix s-axis bug: don't prepend 1.0 if already in list
    s_vals_plot = list(s_values)
    if 1.0 not in s_vals_plot:
        s_vals_plot = [1.0] + s_vals_plot
    
    collision_means = []
    collision_lows = []
    collision_highs = []
    gini_means = []
    gini_lows = []
    gini_highs = []
    
    # Uniform (as separate point, not in s sweep)
    uniform_s_pos = None
    if "uniform" in summary:
        uniform_s_pos = 0 if 1.0 not in s_values else None
        if uniform_s_pos is not None:
            collision_means.insert(0, summary["uniform"]["collision_rate"]["mean"])
            collision_lows.insert(0, summary["uniform"]["collision_rate"]["ci95_low"])
            collision_highs.insert(0, summary["uniform"]["collision_rate"]["ci95_high"])
            gini_means.insert(0, summary["uniform"]["gini"]["mean"])
            gini_lows.insert(0, summary["uniform"]["gini"]["ci95_low"])
            gini_highs.insert(0, summary["uniform"]["gini"]["ci95_high"])
    
    # Zipf values
    for s in s_values:
        key = f"zipf_s_{s}"
        if key in summary:
            collision_means.append(summary[key]["collision_rate"]["mean"])
            collision_lows.append(summary[key]["collision_rate"]["ci95_low"])
            collision_highs.append(summary[key]["collision_rate"]["ci95_high"])
            gini_means.append(summary[key]["gini"]["mean"])
            gini_lows.append(summary[key]["gini"]["ci95_low"])
            gini_highs.append(summary[key]["gini"]["ci95_high"])
        else:
            collision_means.append(0)
            collision_lows.append(0)
            collision_highs.append(0)
            gini_means.append(0)
            gini_lows.append(0)
            gini_highs.append(0)
    
    # Plot collision rate
    ax2_twin = ax2.twinx()
    plot_line_with_ci(ax2, s_vals_plot, collision_means, collision_lows, collision_highs,
                      label="Collision Rate", linestyle="-", color="blue")
    ax2.set_xlabel("Zipf Exponent (s)")
    ax2.set_ylabel("Collision Rate", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    
    # Plot Gini on secondary axis
    plot_line_with_ci(ax2_twin, s_vals_plot, gini_means, gini_lows, gini_highs,
                      label="Gini Coefficient", linestyle="--", color="red")
    ax2_twin.set_ylabel("Gini Coefficient", color="red")
    ax2_twin.tick_params(axis='y', labelcolor="red")
    
    # Mark uniform point if separate
    if uniform_s_pos is not None and uniform_s_pos < len(s_vals_plot):
        ax2.scatter([s_vals_plot[uniform_s_pos]], [collision_means[uniform_s_pos]],
                   color="blue", marker="o", s=100, zorder=5, label="Uniform (collision)")
        ax2_twin.scatter([s_vals_plot[uniform_s_pos]], [gini_means[uniform_s_pos]],
                        color="red", marker="s", s=100, zorder=5, label="Uniform (gini)")
    
    ax2.set_title("Collision Rate & Gini vs Skew")
    ax2.grid(True, alpha=0.3)
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    add_footer(fig, EXP_ID)
    
    # Save outputs
    metrics_path, figure_path = make_output_paths(out_dir, EXP_ID, EXP_SLUG)
    
    config_dict = {
        "N": N,
        "s_values": s_values,
        "vocab_size": vocab_size,
        "B": B,
        "L": L,
        "K": K,
        "H": H,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "Occupancy skew",
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
