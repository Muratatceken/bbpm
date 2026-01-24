"""Experiment 03: Runtime benchmark vs attention."""

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.experiments.common import (
    make_output_paths,
    ensure_device,
    write_metrics_json,
)
from bbpm.experiments.plotting import save_pdf, add_footer, plot_line_with_ci
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp03"
EXP_SLUG = "runtime_vs_attention"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
    parser.add_argument(
        "--T_values",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Sequence length T values to sweep",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model dimension",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )


class ScaledDotProductAttention(nn.Module):
    """Minimal scaled dot-product attention for baseline."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention: x -> Q, K, V -> attention -> output.
        
        Args:
            x: Input tensor [T, d_model]
            
        Returns:
            Output tensor [T, d_model]
        """
        T = x.shape[0]
        # Simple projection: use x as Q, K, V (for baseline)
        q = k = v = x.unsqueeze(0)  # [1, T, d_model]
        
        # Reshape for multi-head
        q = q.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)  # [1, H, T, d_h]
        k = k.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(1, T, self.d_model)
        return out.squeeze(0)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run runtime benchmark experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    device = ensure_device(args.device)
    dtype_str = args.dtype
    T_values = args.T_values
    d_model = args.d_model
    num_heads = args.num_heads
    out_dir = args.out_dir
    
    # BBPM configuration
    B = 2**14  # 16384 blocks
    L = 256
    K = 32
    H = 4
    d = d_model
    
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
    
    # Attention baseline
    attention = ScaledDotProductAttention(d_model, num_heads).to(device)
    if dtype_str == "bfloat16":
        attention = attention.to(torch.bfloat16)
    
    raw_trials = []
    num_warmup = 50
    num_timed = 200
    
    seed_everything(42)
    
    for T in T_values:
        # Generate test data
        torch.manual_seed(42)
        x = torch.randn(T, d_model, device=device)
        if dtype_str == "bfloat16":
            x = x.to(torch.bfloat16)
        
        # Generate random keys for BBPM
        import random
        random.seed(42)
        hx_list = [random.randint(0, 2**64 - 1) for _ in range(T)]
        values = torch.randn(T, d_model, device=device)
        if dtype_str == "bfloat16":
            values = values.to(torch.bfloat16)
        
        # === Timing: Attention baseline ===
        # Warmup
        for _ in range(num_warmup):
            _ = attention(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_timed):
            _ = attention(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        attention_time = (time.perf_counter() - start) / num_timed
        
        # Peak memory for attention
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = attention(x)
            torch.cuda.synchronize()
            attention_peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            attention_peak_mem = 0.0
        
        # === Timing: BBPM addressing only ===
        mem = BBPMMemory(mem_cfg)
        
        # Warmup
        for _ in range(num_warmup):
            for hx in hx_list:
                _ = addresser.addresses_tensor(hx, device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_timed):
            for hx in hx_list:
                _ = addresser.addresses_tensor(hx, device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        addressing_time = (time.perf_counter() - start) / num_timed
        
        # === Timing: BBPM gather throughput ===
        # Write once, then measure gather
        mem.reset()
        for hx, v in zip(hx_list, values):
            mem.write(hx, v)
        
        # Warmup
        for _ in range(num_warmup):
            for hx in hx_list:
                _ = mem.read(hx)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_timed):
            for hx in hx_list:
                _ = mem.read(hx)
        
        if device.type == "cuda":
                torch.cuda.synchronize()
        
        gather_time = (time.perf_counter() - start) / num_timed
        
        # Compute gather bandwidth (GB/s)
        # Each read gathers K*H*d floats = K*H*d*4 bytes (float32) or K*H*d*2 bytes (bfloat16)
        bytes_per_read = K * H * d * (4 if dtype_str == "float32" else 2)
        total_bytes = bytes_per_read * T * num_timed
        gather_bandwidth = (total_bytes / (1024**3)) / gather_time  # GB/s
        
        # === Timing: BBPM end-to-end (addressing + gather) ===
        mem.reset()
        
        # Warmup
        for _ in range(num_warmup):
            for hx, v in zip(hx_list, values):
                mem.write(hx, v)
            for hx in hx_list:
                _ = mem.read(hx)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_timed):
            for hx, v in zip(hx_list, values):
                mem.write(hx, v)
            for hx in hx_list:
                _ = mem.read(hx)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        bbpm_e2e_time = (time.perf_counter() - start) / num_timed
        
        # Peak memory for BBPM
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem.reset()
            for hx, v in zip(hx_list, values):
                mem.write(hx, v)
            torch.cuda.synchronize()
            bbpm_peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            bbpm_peak_mem = 0.0
        
        raw_trials.append({
            "T": T,
            "attention_time_ms": attention_time * 1000,
            "addressing_time_ms": addressing_time * 1000,
            "gather_time_ms": gather_time * 1000,
            "gather_bandwidth_gbps": gather_bandwidth,
            "bbpm_e2e_time_ms": bbpm_e2e_time * 1000,
            "attention_peak_mem_mb": attention_peak_mem,
            "bbpm_peak_mem_mb": bbpm_peak_mem,
        })
    
    # Summarize (for this experiment, we just use raw values since T is the sweep variable)
    summary = {}
    for trial in raw_trials:
        T = trial["T"]
        summary[f"T_{T}"] = {
            "attention_time_ms": trial["attention_time_ms"],
            "addressing_time_ms": trial["addressing_time_ms"],
            "gather_time_ms": trial["gather_time_ms"],
            "gather_bandwidth_gbps": trial["gather_bandwidth_gbps"],
            "bbpm_e2e_time_ms": trial["bbpm_e2e_time_ms"],
            "attention_peak_mem_mb": trial["attention_peak_mem_mb"],
            "bbpm_peak_mem_mb": trial["bbpm_peak_mem_mb"],
        }
    
    # Generate figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Panel 1: Runtime vs T
    T_vals = [t["T"] for t in raw_trials]
    attention_times = [t["attention_time_ms"] for t in raw_trials]
    addressing_times = [t["addressing_time_ms"] for t in raw_trials]
    gather_times = [t["gather_time_ms"] for t in raw_trials]
    bbpm_e2e_times = [t["bbpm_e2e_time_ms"] for t in raw_trials]
    
    ax1.plot(T_vals, attention_times, "o-", label="Attention", linewidth=2)
    ax1.plot(T_vals, addressing_times, "s-", label="BBPM Addressing", linewidth=2)
    ax1.plot(T_vals, gather_times, "^-", label="BBPM Gather", linewidth=2)
    ax1.plot(T_vals, bbpm_e2e_times, "d-", label="BBPM End-to-End", linewidth=2)
    ax1.set_xlabel("Sequence Length (T)")
    ax1.set_ylabel("Time per Forward Pass (ms)")
    ax1.set_title("Exp03: Runtime vs Sequence Length")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    
    # Panel 2: Peak Memory vs T
    if device.type == "cuda":
        attention_mems = [t["attention_peak_mem_mb"] for t in raw_trials]
        bbpm_mems = [t["bbpm_peak_mem_mb"] for t in raw_trials]
        
        ax2.plot(T_vals, attention_mems, "o-", label="Attention", linewidth=2)
        ax2.plot(T_vals, bbpm_mems, "s-", label="BBPM", linewidth=2)
        ax2.set_xlabel("Sequence Length (T)")
        ax2.set_ylabel("Peak Memory (MB)")
        ax2.set_title("Peak Memory vs Sequence Length")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xscale("log", base=2)
    else:
        ax2.text(0.5, 0.5, "Peak memory tracking\navailable on CUDA only",
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Peak Memory vs Sequence Length (CUDA only)")
    
    add_footer(fig, EXP_ID)
    
    # Save outputs
    metrics_path, figure_path = make_output_paths(out_dir, EXP_ID, EXP_SLUG)
    
    config_dict = {
        "B": B,
        "L": L,
        "K": K,
        "H": H,
        "d": d,
        "d_model": d_model,
        "num_heads": num_heads,
        "T_values": T_values,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "Runtime vs Attention",
        config_dict,
        [42],  # Single seed for timing
        raw_trials,
        summary,
    )
    
    save_pdf(fig, figure_path)
    
    return {
        "metrics_path": str(metrics_path),
        "figure_path": str(figure_path),
    }
