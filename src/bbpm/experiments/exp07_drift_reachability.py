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
    # Deterministic hash from token_id (no seed XOR for stability)
    hx = mix64(u64(token_id))
    return hx


def packbits_to_u64(bits_bool: torch.Tensor) -> int:
    """Pack boolean tensor to uint64.
    
    Args:
        bits_bool: Boolean tensor of shape [bits] where bits <= 64
        
    Returns:
        Packed uint64 integer
    """
    bits = bits_bool.cpu().numpy().astype(np.uint64)
    result = 0
    for i, bit in enumerate(bits):
        if bit:
            result |= (1 << i)
    return int(result)


def frozen_projection_keying(embedding: torch.Tensor, projection: torch.Tensor, seed: int) -> int:
    """Frozen random projection keying with multi-bit sign-hash.
    
    Args:
        embedding: Token embedding [d]
        projection: Frozen projection matrix [bits, d] where bits=64
        seed: Random seed (unused, projection is frozen)
        
    Returns:
        hx (uint64)
    """
    # Project to bits: [bits] = (W @ e > 0)
    bits_bool = (projection @ embedding > 0)  # [bits]
    # Pack to uint64
    packed_u64 = packbits_to_u64(bits_bool)
    # Hash
    hx = mix64(u64(packed_u64))
    return hx


def trainable_projection_keying(embedding: torch.Tensor, projection: torch.Tensor, seed: int) -> int:
    """Trainable projection keying (drifts during training) with multi-bit sign-hash.
    
    Args:
        embedding: Token embedding [d]
        projection: Projection tensor [bits, d] where bits=64 (can be Parameter or Tensor)
        seed: Random seed (unused)
        
    Returns:
        hx (uint64)
    """
    # Project to bits: [bits] = (W @ e > 0)
    bits_bool = (projection @ embedding > 0)  # [bits]
    # Pack to uint64
    packed_u64 = packbits_to_u64(bits_bool)
    # Hash
    hx = mix64(u64(packed_u64))
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
        default=512,
        help="Number of early items to track",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="Drift noise scale for embeddings",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1e-3,
        help="Drift noise scale for trainable projection",
    )
    parser.add_argument(
        "--reachability_threshold",
        type=float,
        default=0.9,
        help="Cosine threshold for reachability success",
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
    sigma = args.sigma
    eta = args.eta
    reachability_threshold = args.reachability_threshold
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
        # Initialize e0 ~ N(0,1) normalized
        torch.manual_seed(seed)
        early_embeddings = torch.randn(num_early_items, d, device=device)
        early_embeddings = torch.nn.functional.normalize(early_embeddings, p=2, dim=1)
        early_values = early_embeddings.clone()
        early_token_ids = list(range(num_early_items))
        
        # Mode 1: Token-ID keying (stable)
        mem.reset()
        token_hx_list_initial = [token_id_keying(tid, seed) for tid in early_token_ids]
        for hx, v in zip(token_hx_list_initial, early_values):
            mem.write(hx, v)
        
        # Simulate gradual drift: e = normalize(e + sigma * noise)
        current_embeddings = early_embeddings.clone()
        reachability_token = []
        key_change_rate_token = []
        for step in range(num_steps):
            # Gradual drift: deterministic noise from (seed, step)
            torch.manual_seed(seed + step)
            noise = torch.randn(num_early_items, d, device=device)
            current_embeddings = current_embeddings + sigma * noise
            current_embeddings = torch.nn.functional.normalize(current_embeddings, p=2, dim=1)
            
            # Retrieve early items using token-ID keying (stable)
            retrieved = []
            for tid in early_token_ids:
                hx = token_id_keying(tid, seed)  # Same key always
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            # Compute reachability (success rate)
            cosines = torch.tensor([cosine_similarity(v, r) for v, r in zip(early_values, retrieved_tensor)])
            success = (cosines >= reachability_threshold).float()
            reachability = success.mean().item()
            reachability_token.append(reachability)
            
            # Key change rate (always 0 for token-ID, keys don't change)
            key_change_rate_token.append(0.0)
        
        # Mode 2: Frozen projection keying (multi-bit sign-hash)
        mem.reset()
        # Create frozen projection W: [bits, d] where bits=64
        bits = 64
        torch.manual_seed(seed)
        frozen_proj = torch.randn(bits, d, device=device)
        frozen_proj = frozen_proj / torch.norm(frozen_proj, dim=1, keepdim=True)  # Normalize each row
        
        frozen_hx_list_initial = [frozen_projection_keying(emb, frozen_proj, seed) for emb in early_embeddings]
        for hx, v in zip(frozen_hx_list_initial, early_values):
            mem.write(hx, v)
        
        # Simulate gradual drift: e = normalize(e + sigma * noise)
        current_embeddings = early_embeddings.clone()
        reachability_frozen = []
        key_change_rate_frozen = []
        hamming_frozen = []
        for step in range(num_steps):
            # Gradual drift: deterministic noise from (seed, step)
            torch.manual_seed(seed + step)
            noise = torch.randn(num_early_items, d, device=device)
            current_embeddings = current_embeddings + sigma * noise
            current_embeddings = torch.nn.functional.normalize(current_embeddings, p=2, dim=1)
            
            # Compute current keys
            current_hx_list = [frozen_projection_keying(emb, frozen_proj, seed) for emb in current_embeddings]
            
            # Key change rate
            key_changes = sum(1 for hx0, hx in zip(frozen_hx_list_initial, current_hx_list) if hx0 != hx)
            key_change_rate = key_changes / num_early_items
            key_change_rate_frozen.append(key_change_rate)
            
            # Hamming distance (bitcount XOR)
            hamming_dists = []
            for hx0, hx in zip(frozen_hx_list_initial, current_hx_list):
                hamming = bin(hx0 ^ hx).count('1')
                hamming_dists.append(hamming)
            mean_hamming = np.mean(hamming_dists)
            hamming_frozen.append(mean_hamming)
            
            # Retrieve using frozen projection (stable projection, but embeddings drift)
            retrieved = []
            for hx in current_hx_list:
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            # Compute reachability (success rate)
            cosines = torch.tensor([cosine_similarity(v, r) for v, r in zip(early_values, retrieved_tensor)])
            success = (cosines >= reachability_threshold).float()
            reachability = success.mean().item()
            reachability_frozen.append(reachability)
        
        # Mode 3: Trainable projection keying (drifts)
        mem.reset()
        # Create trainable projection W: [bits, d] where bits=64
        bits = 64
        trainable_proj = nn.Parameter(torch.randn(bits, d, device=device))
        trainable_proj.data = trainable_proj.data / torch.norm(trainable_proj.data, dim=1, keepdim=True)
        
        initial_hx_list = [trainable_projection_keying(emb, trainable_proj, seed) for emb in early_embeddings]
        for hx, v in zip(initial_hx_list, early_values):
            mem.write(hx, v)
        
        # Simulate gradual drift: e = normalize(e + sigma * noise), W = normalize(W + eta * noise)
        current_embeddings = early_embeddings.clone()
        current_proj = trainable_proj.data.clone()
        reachability_trainable = []
        key_change_rate_trainable = []
        hamming_trainable = []
        for step in range(num_steps):
            # Drift embeddings: deterministic noise from (seed, step)
            torch.manual_seed(seed + step)
            emb_noise = torch.randn(num_early_items, d, device=device)
            current_embeddings = current_embeddings + sigma * emb_noise
            current_embeddings = torch.nn.functional.normalize(current_embeddings, p=2, dim=1)
            
            # Drift projection: W = normalize(W + eta * noise)
            torch.manual_seed(seed + step + 10000)  # Different seed for projection noise
            proj_noise = torch.randn(bits, d, device=device)
            current_proj = current_proj + eta * proj_noise
            current_proj = current_proj / torch.norm(current_proj, dim=1, keepdim=True)
            
            # Compute current keys with drifted projection
            current_hx_list = []
            for emb in current_embeddings:
                # Use current_proj tensor directly
                hx = trainable_projection_keying(emb, current_proj, seed)
                current_hx_list.append(hx)
            
            # Key change rate
            key_changes = sum(1 for hx0, hx in zip(initial_hx_list, current_hx_list) if hx0 != hx)
            key_change_rate = key_changes / num_early_items
            key_change_rate_trainable.append(key_change_rate)
            
            # Hamming distance
            hamming_dists = []
            for hx0, hx in zip(initial_hx_list, current_hx_list):
                hamming = bin(hx0 ^ hx).count('1')
                hamming_dists.append(hamming)
            mean_hamming = np.mean(hamming_dists)
            hamming_trainable.append(mean_hamming)
            
            # Retrieve using current projection (may have drifted)
            retrieved = []
            for hx in current_hx_list:
                r = mem.read(hx)
                retrieved.append(r)
            retrieved_tensor = torch.stack(retrieved)
            
            # Compute reachability (success rate)
            cosines = torch.tensor([cosine_similarity(v, r) for v, r in zip(early_values, retrieved_tensor)])
            success = (cosines >= reachability_threshold).float()
            reachability = success.mean().item()
            reachability_trainable.append(reachability)
        
        # Record trial
        raw_trials.append({
            "seed": seed,
            "steps": list(range(num_steps)),
            "token_id_reachability": reachability_token,
            "token_id_key_change_rate": key_change_rate_token,
            "frozen_proj_reachability": reachability_frozen,
            "frozen_proj_key_change_rate": key_change_rate_frozen,
            "frozen_proj_hamming": hamming_frozen,
            "trainable_proj_reachability": reachability_trainable,
            "trainable_proj_key_change_rate": key_change_rate_trainable,
            "trainable_proj_hamming": hamming_trainable,
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
    
    # Generate figure with 3 panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Panel 1: Reachability vs step (3 modes)
    for mode, label in [
        ("token_id", "Token-ID Keying (Stable)"),
        ("frozen_proj", "Frozen Projection Keying"),
        ("trainable_proj", "Trainable Projection Keying (Drifts)"),
    ]:
        if mode in summary:
            means = summary[mode]["reachability"]["mean"]
            lows = summary[mode]["reachability"]["ci95_low"]
            highs = summary[mode]["reachability"]["ci95_high"]
            plot_line_with_ci(ax1, steps, means, lows, highs, label=label, linestyle="-")
    
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Reachability (Success Rate)")
    ax1.set_title("Exp07: Early-Item Reachability vs Training Step")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Key change rate vs step
    for mode, label in [
        ("token_id", "Token-ID Keying (Stable)"),
        ("frozen_proj", "Frozen Projection Keying"),
        ("trainable_proj", "Trainable Projection Keying (Drifts)"),
    ]:
        if mode in summary and "key_change_rate" in summary[mode]:
            means = summary[mode]["key_change_rate"]["mean"]
            lows = summary[mode]["key_change_rate"]["ci95_low"]
            highs = summary[mode]["key_change_rate"]["ci95_high"]
            plot_line_with_ci(ax2, steps, means, lows, highs, label=label, linestyle="-")
    
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Key Change Rate")
    ax2.set_title("Key Change Rate vs Training Step")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Hamming distance vs step (for sign-hash modes)
    for mode, label in [
        ("frozen_proj", "Frozen Projection Keying"),
        ("trainable_proj", "Trainable Projection Keying (Drifts)"),
    ]:
        if mode in summary and "hamming_distance" in summary[mode]:
            means = summary[mode]["hamming_distance"]["mean"]
            lows = summary[mode]["hamming_distance"]["ci95_low"]
            highs = summary[mode]["hamming_distance"]["ci95_high"]
            plot_line_with_ci(ax3, steps, means, lows, highs, label=label, linestyle="-")
    
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Mean Hamming Distance")
    ax3.set_title("Mean Hamming Distance vs Training Step")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
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
        "sigma": sigma,
        "eta": eta,
        "reachability_threshold": reachability_threshold,
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
