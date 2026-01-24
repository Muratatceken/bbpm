"""Experiment 06: Occupancy skew with non-uniform load."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from bbpm.metrics.occupancy import block_occupancy
from bbpm.metrics.stats import mean_ci95, gini_coefficient
from common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
)
from plotting import save_pdf, add_footer
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp06"
EXP_SLUG = "occupancy_skew"


def sample_zipf(vocab_size: int, num_samples: int, s: float, seed: int) -> list[int]:
    """Sample from Zipf distribution.
    
    Args:
        vocab_size: Vocabulary size
        num_samples: Number of samples
        s: Zipf exponent (s > 1)
        seed: Random seed
        
    Returns:
        List of token IDs
    """
    random.seed(seed)
    # Generate Zipf-distributed samples
    # Simple approximation: use power law
    probs = np.array([1.0 / (i + 1) ** s for i in range(vocab_size)])
    probs = probs / probs.sum()
    
    return np.random.choice(vocab_size, size=num_samples, p=probs).tolist()


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
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
        
        all_block_ids_uniform = []
        for hx, v in zip(uniform_hx_list, values):
            mem.write(hx, v)
            addrs = addresser.addresses(hx)
            for addr in addrs:
                all_block_ids_uniform.append(addr // L)
        
        occ_uniform = block_occupancy(all_block_ids_uniform, B)
        
        raw_trials.append({
            "seed": seed,
            "mode": "uniform",
            "s": None,
            "max_mean_ratio": occ_uniform["max_occupancy"] / occ_uniform["mean_occupancy"] if occ_uniform["mean_occupancy"] > 0 else 0.0,
            "gini": occ_uniform["gini_coefficient"],
            "collision_rate": 1.0 - (len(set(all_block_ids_uniform)) / len(all_block_ids_uniform)) if all_block_ids_uniform else 0.0,
        })
        
        # Mode 2: Zipf-distributed keys
        for s in s_values:
            mem.reset()
            zipf_token_ids = sample_zipf(vocab_size, N, s, seed)
            # Convert token IDs to hx (deterministic mapping)
            zipf_hx_list = []
            for token_id in zipf_token_ids:
                # Deterministic hash from token ID
                hx = hash(f"token_{token_id}_{seed}") % (2**64)
                zipf_hx_list.append(hx)
            
            values = torch.randn(N, d, device=device)
            values = torch.nn.functional.normalize(values, p=2, dim=1)
            
            all_block_ids_zipf = []
            for hx, v in zip(zipf_hx_list, values):
                mem.write(hx, v)
                addrs = addresser.addresses(hx)
                for addr in addrs:
                    all_block_ids_zipf.append(addr // L)
            
            occ_zipf = block_occupancy(all_block_ids_zipf, B)
            
            raw_trials.append({
                "seed": seed,
                "mode": "zipf",
                "s": s,
                "max_mean_ratio": occ_zipf["max_occupancy"] / occ_zipf["mean_occupancy"] if occ_zipf["mean_occupancy"] > 0 else 0.0,
                "gini": occ_zipf["gini_coefficient"],
                "collision_rate": 1.0 - (len(set(all_block_ids_zipf)) / len(all_block_ids_zipf)) if all_block_ids_zipf else 0.0,
            })
    
    # Summarize
    summary = {}
    
    # Uniform mode
    uniform_trials = [t for t in raw_trials if t["mode"] == "uniform"]
    if uniform_trials:
        max_mean_vals = [t["max_mean_ratio"] for t in uniform_trials]
        gini_vals = [t["gini"] for t in uniform_trials]
        collision_vals = [t["collision_rate"] for t in uniform_trials]
        
        summary["uniform"] = {
            "max_mean_ratio": {
                "mean": np.mean(max_mean_vals),
                "ci95_low": mean_ci95(max_mean_vals)[1],
                "ci95_high": mean_ci95(max_mean_vals)[2],
                "std": np.std(max_mean_vals),
            },
            "gini": {
                "mean": np.mean(gini_vals),
                "ci95_low": mean_ci95(gini_vals)[1],
                "ci95_high": mean_ci95(gini_vals)[2],
                "std": np.std(gini_vals),
            },
            "collision_rate": {
                "mean": np.mean(collision_vals),
                "ci95_low": mean_ci95(collision_vals)[1],
                "ci95_high": mean_ci95(collision_vals)[2],
                "std": np.std(collision_vals),
            },
        }
    
    # Zipf modes
    for s in s_values:
        zipf_trials = [t for t in raw_trials if t["mode"] == "zipf" and t["s"] == s]
        if zipf_trials:
            max_mean_vals = [t["max_mean_ratio"] for t in zipf_trials]
            gini_vals = [t["gini"] for t in zipf_trials]
            collision_vals = [t["collision_rate"] for t in zipf_trials]
            
            summary[f"zipf_s_{s}"] = {
                "max_mean_ratio": {
                    "mean": np.mean(max_mean_vals),
                    "ci95_low": mean_ci95(max_mean_vals)[1],
                    "ci95_high": mean_ci95(max_mean_vals)[2],
                    "std": np.std(max_mean_vals),
                },
                "gini": {
                    "mean": np.mean(gini_vals),
                    "ci95_low": mean_ci95(gini_vals)[1],
                    "ci95_high": mean_ci95(gini_vals)[2],
                    "std": np.std(gini_vals),
                },
                "collision_rate": {
                    "mean": np.mean(collision_vals),
                    "ci95_low": mean_ci95(collision_vals)[1],
                    "ci95_high": mean_ci95(collision_vals)[2],
                    "std": np.std(collision_vals),
                },
            }
    
    # Generate figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Occupancy histogram (example for one seed, uniform vs zipf)
    seed_example = seeds[0]
    uniform_trial = [t for t in raw_trials if t["seed"] == seed_example and t["mode"] == "uniform"][0]
    zipf_trial_s15 = [t for t in raw_trials if t["seed"] == seed_example and t["mode"] == "zipf" and t["s"] == 1.5][0]
    
    # Recompute for visualization (simplified - use summary stats)
    ax1.bar([0, 1], [uniform_trial["gini"], zipf_trial_s15["gini"]],
            label=["Uniform", "Zipf (s=1.5)"], alpha=0.7)
    ax1.set_ylabel("Gini Coefficient")
    ax1.set_title("Occupancy Inequality (Gini)")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Uniform", "Zipf (s=1.5)"])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Panel 2: Collision rate vs Zipf exponent
    s_vals_plot = [1.0] + s_values  # Include uniform as s=1.0
    collision_means = []
    collision_lows = []
    collision_highs = []
    
    # Uniform
    if "uniform" in summary:
        collision_means.append(summary["uniform"]["collision_rate"]["mean"])
        collision_lows.append(summary["uniform"]["collision_rate"]["ci95_low"])
        collision_highs.append(summary["uniform"]["collision_rate"]["ci95_high"])
    else:
        collision_means.append(0)
        collision_lows.append(0)
        collision_highs.append(0)
    
    # Zipf
    for s in s_values:
        key = f"zipf_s_{s}"
        if key in summary:
            collision_means.append(summary[key]["collision_rate"]["mean"])
            collision_lows.append(summary[key]["collision_rate"]["ci95_low"])
            collision_highs.append(summary[key]["collision_rate"]["ci95_high"])
        else:
            collision_means.append(0)
            collision_lows.append(0)
            collision_highs.append(0)
    
    plot_line_with_ci(ax2, s_vals_plot, collision_means, collision_lows, collision_highs,
                      label="Collision Rate", linestyle="-")
    ax2.set_xlabel("Zipf Exponent (s)")
    ax2.set_ylabel("Collision Rate")
    ax2.set_title("Collision Rate vs Skew (Zipf Exponent)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
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
