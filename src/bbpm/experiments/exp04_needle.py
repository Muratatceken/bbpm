"""Experiment 04: Needle-in-haystack retrieval."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.retrieval import cosine_similarity
from bbpm.metrics.stats import mean_ci95
from bbpm.experiments.common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
)
from bbpm.experiments.plotting import save_pdf, add_footer, plot_line_with_ci
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp04"
EXP_SLUG = "needle"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
    parser.add_argument(
        "--N_values",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="Load values (number of distractors)",
    )
    parser.add_argument(
        "--distance_values",
        type=int,
        nargs="+",
        default=[0, 10, 50, 100, 500, 1000],
        help="Distance values (number of items between needle and query)",
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run needle-in-haystack experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    device = ensure_device(args.device)
    dtype_str = args.dtype
    num_seeds = args.seeds
    N_values = args.N_values
    distance_values = args.distance_values
    out_dir = args.out_dir
    
    # Fixed memory configuration
    D = 2**20  # 1M slots
    d = 64
    B = 2**14  # 16384 blocks
    L = 256
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
        read_mode="count_normalized",  # Enable occupancy tracking
        master_seed=42,
    )
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    # Run trials
    for seed in seeds:
        seed_everything(seed)
        mem = BBPMMemory(mem_cfg)
        
        # === Experiment 1: Retrieval vs Distance (fixed N) ===
        fixed_N = 2000
        for distance in distance_values:
            mem.reset()
            
            # Generate needle item
            random.seed(seed)
            torch.manual_seed(seed)
            needle_hx = random.randint(0, 2**64 - 1)
            needle_value = torch.randn(d, device=device)
            needle_value = torch.nn.functional.normalize(needle_value, p=2, dim=0)
            
            # Generate all distractors
            distractor_hx_list = [random.randint(0, 2**64 - 1) for _ in range(fixed_N)]
            distractor_values = torch.randn(fixed_N, d, device=device)
            distractor_values = torch.nn.functional.normalize(distractor_values, p=2, dim=1)
            
            # Fix distance protocol: keep total load fixed
            # Write (fixed_N - distance) distractors first
            num_before = fixed_N - distance
            for hx, v in zip(distractor_hx_list[:num_before], distractor_values[:num_before]):
                mem.write(hx, v)
            
            # Write needle
            mem.write(needle_hx, needle_value)
            
            # Write remaining distance distractors
            for hx, v in zip(distractor_hx_list[num_before:], distractor_values[num_before:]):
                mem.write(hx, v)
            
            # Compute measured occupancy
            stats = mem.stats()
            if "occupied_slots" in stats:
                occupied_ratio = stats["occupied_slots"] / D
            else:
                total_writes = (fixed_N + 1) * K * H
                occupied_ratio = min(1.0, total_writes / D)
            
            # Compute empirical lambda
            total_writes = (fixed_N + 1) * K * H
            empirical_lambda = total_writes / D
            
            # Retrieve needle
            retrieved = mem.read(needle_hx)
            cosine = cosine_similarity(needle_value, retrieved)
            
            raw_trials.append({
                "seed": seed,
                "mode": "distance",
                "N": fixed_N,
                "distance": distance,
                "cosine": cosine,
                "empirical_lambda": empirical_lambda,
                "occupancy": occupied_ratio,
            })
        
        # === Experiment 2: Retrieval vs Load (fixed distances) ===
        fixed_distances = [0, 500]  # Two distances for stronger evidence
        for fixed_distance in fixed_distances:
            for N in N_values:
                mem.reset()
                
                # Generate needle
                random.seed(seed)
                torch.manual_seed(seed)
                needle_hx = random.randint(0, 2**64 - 1)
                needle_value = torch.randn(d, device=device)
                needle_value = torch.nn.functional.normalize(needle_value, p=2, dim=0)
                
                # Generate all distractors
                distractor_hx_list = [random.randint(0, 2**64 - 1) for _ in range(N)]
                distractor_values = torch.randn(N, d, device=device)
                distractor_values = torch.nn.functional.normalize(distractor_values, p=2, dim=1)
                
                # Fix distance protocol: keep total load fixed
                # Write (N - distance) distractors first
                num_before = N - fixed_distance
                for hx, v in zip(distractor_hx_list[:num_before], distractor_values[:num_before]):
                    mem.write(hx, v)
                
                # Write needle
                mem.write(needle_hx, needle_value)
                
                # Write remaining distance distractors
                for hx, v in zip(distractor_hx_list[num_before:], distractor_values[num_before:]):
                    mem.write(hx, v)
                
                # Compute measured occupancy
                stats = mem.stats()
                if "occupied_slots" in stats:
                    occupied_ratio = stats["occupied_slots"] / D
                else:
                    total_writes = (N + 1) * K * H
                    occupied_ratio = min(1.0, total_writes / D)
                
                # Compute empirical lambda
                total_writes = (N + 1) * K * H
                empirical_lambda = total_writes / D
                
                # Retrieve needle
                retrieved = mem.read(needle_hx)
                cosine = cosine_similarity(needle_value, retrieved)
                
                raw_trials.append({
                    "seed": seed,
                    "mode": "load",
                    "N": N,
                    "distance": fixed_distance,
                    "cosine": cosine,
                    "empirical_lambda": empirical_lambda,
                    "occupancy": occupied_ratio,
                })
    
    # Summarize
    summary = {}
    
    # Distance curves (fixed N)
    distance_cosines = {d: [] for d in distance_values}
    for trial in raw_trials:
        if trial["mode"] == "distance":
            distance_cosines[trial["distance"]].append(trial["cosine"])
    
    for distance in distance_values:
        if distance_cosines[distance]:
            stats = mean_ci95(distance_cosines[distance])
            mean = stats["mean"]
            lo = stats["ci95_low"]
            hi = stats["ci95_high"]
            std = stats["std"]
            summary[f"distance_{distance}"] = {
                "cosine": {
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "std": std,
                }
            }
    
    # Load curves (fixed distances)
    fixed_distances = [0, 500]
    for fixed_distance in fixed_distances:
        load_cosines = {N: [] for N in N_values}
        for trial in raw_trials:
            if trial["mode"] == "load" and trial["distance"] == fixed_distance:
                load_cosines[trial["N"]].append(trial["cosine"])
        
        for N in N_values:
            if load_cosines[N]:
                stats = mean_ci95(load_cosines[N])
                mean = stats["mean"]
                lo = stats["ci95_low"]
                hi = stats["ci95_high"]
                std = stats["std"]
                summary[f"load_d{fixed_distance}_N{N}"] = {
                    "cosine": {
                        "mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "std": std,
                    }
                }
    
    # Generate figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Retrieval vs Distance (fixed N)
    distance_means = []
    distance_lows = []
    distance_highs = []
    for distance in distance_values:
        key = f"distance_{distance}"
        if key in summary:
            distance_means.append(summary[key]["cosine"]["mean"])
            distance_lows.append(summary[key]["cosine"]["ci95_low"])
            distance_highs.append(summary[key]["cosine"]["ci95_high"])
        else:
            distance_means.append(0)
            distance_lows.append(0)
            distance_highs.append(0)
    
    plot_line_with_ci(ax1, distance_values, distance_means, distance_lows, distance_highs,
                      label=f"Cosine (N={fixed_N})", linestyle="-")
    ax1.set_xlabel("Distance (items between needle and query)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Retrieval vs Distance (Fixed Load)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Retrieval vs Load (fixed distances: 0 and 500)
    fixed_distances = [0, 500]
    for fixed_distance in fixed_distances:
        load_means = []
        load_lows = []
        load_highs = []
        for N in N_values:
            key = f"load_d{fixed_distance}_N{N}"
            if key in summary:
                load_means.append(summary[key]["cosine"]["mean"])
                load_lows.append(summary[key]["cosine"]["ci95_low"])
                load_highs.append(summary[key]["cosine"]["ci95_high"])
            else:
                load_means.append(0)
                load_lows.append(0)
                load_highs.append(0)
        
        plot_line_with_ci(ax2, N_values, load_means, load_lows, load_highs,
                          label=f"Distance={fixed_distance}", linestyle="-")
    ax2.set_xlabel("Load (Number of Distractors)")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Retrieval vs Load (Fixed Distances)")
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
        "distance_values": distance_values,
        "fixed_N": fixed_N,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "Needle-in-haystack",
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
