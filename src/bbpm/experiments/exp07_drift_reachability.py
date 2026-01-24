"""Experiment 07: Drift and reachability under keying modes."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from bbpm.addressing.hash_mix import mix64, u64
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

EXP_ID = "exp07"
EXP_SLUG = "drift_reachability"


def token_id_keying(token_id: int, seed: int) -> int:
    """Stable token-ID keying: hx derived from token_id.
    
    Args:
        token_id: Token identifier
        seed: Random seed (for determinism)
        
    Returns:
        hx (uint64)
    """
    # Deterministic hash from token_id
    hx = mix64(u64(token_id) ^ u64(seed))
    return hx


def frozen_projection_keying(embedding: torch.Tensor, projection: torch.Tensor, seed: int) -> int:
    """Frozen random projection keying.
    
    Args:
        embedding: Token embedding [d]
        projection: Frozen projection matrix [d]
        seed: Random seed (unused, projection is frozen)
        
    Returns:
        hx (uint64)
    """
    # Project to scalar
    scalar = torch.sum(embedding * projection).item()
    # Hash to uint64
    hx = mix64(u64(int(scalar * 1000) % (2**64)))
    return hx


def trainable_projection_keying(embedding: torch.Tensor, projection: nn.Parameter, seed: int) -> int:
    """Trainable projection keying (drifts during training).
    
    Args:
        embedding: Token embedding [d]
        projection: Trainable projection parameter [d]
        seed: Random seed (unused)
        
    Returns:
        hx (uint64)
    """
    # Project to scalar
    scalar = torch.sum(embedding * projection).item()
    # Hash to uint64
    hx = mix64(u64(int(scalar * 1000) % (2**64)))
    return hx


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of training steps to simulate",
    )
    parser.add_argument(
        "--num_early_items",
        type=int,
        default=10,
        help="Number of early items to track",
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run drift and reachability experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    device = ensure_device(args.device)
    dtype_str = args.dtype
    num_seeds = args.seeds
    num_steps = args.num_steps
    num_early_items = args.num_early_items
    out_dir = args.out_dir
    
    # Fixed memory configuration
    D = 2**18  # 256K slots
    d = 64
    B = 2**12  # 4096 blocks
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
        read_mode="raw_mean",
        master_seed=42,
    )
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    # Run trials
    for seed in seeds:
        seed_everything(seed)
        mem = BBPMMemory(mem_cfg)
        
        # Generate early items (written at step 0)
        random.seed(seed)
        torch.manual_seed(seed)
        early_token_ids = list(range(num_early_items))
        early_embeddings = torch.randn(num_early_items, d, device=device)
        early_embeddings = torch.nn.functional.normalize(early_embeddings, p=2, dim=1)
        early_values = early_embeddings.clone()
        
        # Mode 1: Token-ID keying (stable)
        mem.reset()
        token_hx_list = [token_id_keying(tid, seed) for tid in early_token_ids]
        for hx, v in zip(token_hx_list, early_values):
            mem.write(hx, v)
        
        # Simulate training steps (add more items, embeddings change)
        reachability_token = []
        for step in range(num_steps):
            # Generate new embeddings (simulating training drift)
            torch.manual_seed(seed + step)
            new_embeddings = torch.randn(num_early_items, d, device=device)
            new_embeddings = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
            
            # Retrieve early items using token-ID keying (stable)
            retrieved = []
            for tid in early_token_ids:
                hx = token_id_keying(tid, seed)  # Same key always
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            # Compute reachability
            cosines = [cosine_similarity(v, r) for v, r in zip(early_values, retrieved_tensor)]
            mean_cosine = np.mean(cosines)
            reachability_token.append(mean_cosine)
        
        # Mode 2: Frozen projection keying
        mem.reset()
        # Create frozen projection
        torch.manual_seed(seed)
        frozen_proj = torch.randn(d, device=device)
        frozen_proj = frozen_proj / torch.norm(frozen_proj)
        
        frozen_hx_list = [frozen_projection_keying(emb, frozen_proj, seed) for emb in early_embeddings]
        for hx, v in zip(frozen_hx_list, early_values):
            mem.write(hx, v)
        
        reachability_frozen = []
        for step in range(num_steps):
            # New embeddings
            torch.manual_seed(seed + step)
            new_embeddings = torch.randn(num_early_items, d, device=device)
            new_embeddings = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
            
            # Retrieve using frozen projection (stable)
            retrieved = []
            for emb in new_embeddings:
                hx = frozen_projection_keying(emb, frozen_proj, seed)
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            cosines = [cosine_similarity(v, r) for v, r in zip(early_values, retrieved_tensor)]
            mean_cosine = np.mean(cosines)
            reachability_frozen.append(mean_cosine)
        
        # Mode 3: Trainable projection keying (drifts)
        mem.reset()
        # Create trainable projection
        trainable_proj = nn.Parameter(torch.randn(d, device=device))
        trainable_proj.data = trainable_proj.data / torch.norm(trainable_proj.data)
        
        initial_hx_list = [trainable_projection_keying(emb, trainable_proj, seed) for emb in early_embeddings]
        for hx, v in zip(initial_hx_list, early_values):
            mem.write(hx, v)
        
        reachability_trainable = []
        for step in range(num_steps):
            # Simulate training: update projection slightly
            with torch.no_grad():
                noise = torch.randn(d, device=device) * 0.01
                trainable_proj.data = trainable_proj.data + noise
                trainable_proj.data = trainable_proj.data / torch.norm(trainable_proj.data)
            
            # New embeddings
            torch.manual_seed(seed + step)
            new_embeddings = torch.randn(num_early_items, d, device=device)
            new_embeddings = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
            
            # Retrieve using current projection (may have drifted)
            retrieved = []
            for emb in new_embeddings:
                hx = trainable_projection_keying(emb, trainable_proj, seed)
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            cosines = [cosine_similarity(v, r) for v, r in zip(early_values, retrieved_tensor)]
            mean_cosine = np.mean(cosines)
            reachability_trainable.append(mean_cosine)
        
        # Record trial
        raw_trials.append({
            "seed": seed,
            "steps": list(range(num_steps)),
            "token_id_reachability": reachability_token,
            "frozen_proj_reachability": reachability_frozen,
            "trainable_proj_reachability": reachability_trainable,
        })
    
    # Summarize across seeds
    summary = {}
    steps = list(range(num_steps))
    
    for mode in ["token_id", "frozen_proj", "trainable_proj"]:
        mode_key = f"{mode}_reachability"
        reachability_by_step = {step: [] for step in steps}
        
        for trial in raw_trials:
            for step, reach in zip(trial["steps"], trial[mode_key]):
                reachability_by_step[step].append(reach)
        
        step_means = []
        step_lows = []
        step_highs = []
        for step in steps:
            if reachability_by_step[step]:
                stats = mean_ci95(reachability_by_step[step])
                mean = stats["mean"]
                lo = stats["ci95_low"]
                hi = stats["ci95_high"]
                std = stats["std"]
                step_means.append(mean)
                step_lows.append(lo)
                step_highs.append(hi)
            else:
                step_means.append(0)
                step_lows.append(0)
                step_highs.append(0)
        
        summary[mode] = {
            "reachability": {
                "mean": step_means,
                "ci95_low": step_lows,
                "ci95_high": step_highs,
            }
        }
    
    # Generate figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot reachability curves for three modes
    for mode, label in [
        ("token_id", "Token-ID Keying (Stable)"),
        ("frozen_proj", "Frozen Projection Keying"),
        ("trainable_proj", "Trainable Projection Keying (Drifts)"),
    ]:
        if mode in summary:
            means = summary[mode]["reachability"]["mean"]
            lows = summary[mode]["reachability"]["ci95_low"]
            highs = summary[mode]["reachability"]["ci95_high"]
            plot_line_with_ci(ax, steps, means, lows, highs, label=label, linestyle="-")
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reachability (Cosine Similarity)")
    ax.set_title("Exp07: Early-Item Reachability vs Training Step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
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
        "num_steps": num_steps,
        "num_early_items": num_early_items,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "Drift and reachability",
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
