#!/usr/bin/env python3
"""CUDA benchmark for vectorized batch addressing operations.

Measures performance of addresses_batch, write_batch, and read_batch
with proper CUDA synchronization. Compares batch vs scalar (loop) performance.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.addressing.prp import u64_to_i64
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.utils.seeds import seed_everything


def benchmark_addresses_batch(
    addresser: BlockAddress,
    hx_tensor: "torch.LongTensor",
    device: torch.device,
    num_warmup: int = 50,
    num_repeats: int = 200,
) -> Dict[str, Any]:
    """Benchmark addresses_batch (vectorized)."""
    # Warmup
    for _ in range(num_warmup):
        _ = addresser.addresses_batch(hx_tensor, device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Timed repeats
    times = []
    for _ in range(num_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = addresser.addresses_batch(hx_tensor, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    ci95 = 1.96 * std_time / (len(times) ** 0.5)
    
    return {
        "method": "addresses_batch (vectorized)",
        "device": str(device),
        "T": hx_tensor.shape[0],
        "mean_sec": mean_time,
        "std_sec": std_time,
        "ci95_low": mean_time - ci95,
        "ci95_high": mean_time + ci95,
        "ops_per_sec": hx_tensor.shape[0] / mean_time,
    }


def benchmark_addresses_scalar_loop(
    addresser: BlockAddress,
    hx_list: list[int],
    device: torch.device,
    num_warmup: int = 50,
    num_repeats: int = 200,
) -> Dict[str, Any]:
    """Benchmark addresses_tensor in a loop (scalar, for comparison)."""
    # Warmup
    for hx in hx_list[:num_warmup]:
        _ = addresser.addresses_tensor(hx, device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Timed repeats
    times = []
    for _ in range(num_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for hx in hx_list:
            _ = addresser.addresses_tensor(hx, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    ci95 = 1.96 * std_time / (len(times) ** 0.5)
    
    return {
        "method": "addresses_tensor (loop)",
        "device": str(device),
        "T": len(hx_list),
        "mean_sec": mean_time,
        "std_sec": std_time,
        "ci95_low": mean_time - ci95,
        "ci95_high": mean_time + ci95,
        "ops_per_sec": len(hx_list) / mean_time,
    }


def benchmark_write_batch(
    mem: BBPMMemory,
    hx_tensor: "torch.LongTensor",
    values: "torch.Tensor",
    device: torch.device,
    num_warmup: int = 50,
    num_repeats: int = 200,
) -> Dict[str, Any]:
    """Benchmark write_batch (vectorized)."""
    # Warmup
    for _ in range(num_warmup):
        mem.reset()
        mem.write_batch(hx_tensor, values)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Timed repeats
    times = []
    for _ in range(num_repeats):
        mem.reset()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        mem.write_batch(hx_tensor, values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    ci95 = 1.96 * std_time / (len(times) ** 0.5)
    
    # Compute bandwidth
    T = hx_tensor.shape[0]
    H = mem.cfg.H
    K = mem.cfg.K
    d = mem.cfg.key_dim
    bytes_per_elem = 4 if mem.dtype == torch.float32 else 2  # float32=4, bfloat16=2
    bytes_touched = T * H * K * d * bytes_per_elem
    bandwidth_gb_s = bytes_touched / mean_time / (1024**3)
    
    result = {
        "method": "write_batch (vectorized)",
        "device": str(device),
        "T": T,
        "mean_sec": mean_time,
        "std_sec": std_time,
        "ci95_low": mean_time - ci95,
        "ci95_high": mean_time + ci95,
        "bandwidth_gb_s": bandwidth_gb_s,
        "bytes_touched": bytes_touched,
    }
    
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    
    return result


def benchmark_read_batch(
    mem: BBPMMemory,
    hx_tensor: "torch.LongTensor",
    device: torch.device,
    num_warmup: int = 50,
    num_repeats: int = 200,
) -> Dict[str, Any]:
    """Benchmark read_batch (vectorized)."""
    # Warmup
    for _ in range(num_warmup):
        _ = mem.read_batch(hx_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Timed repeats
    times = []
    for _ in range(num_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = mem.read_batch(hx_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    ci95 = 1.96 * std_time / (len(times) ** 0.5)
    
    # Compute bandwidth
    T = hx_tensor.shape[0]
    H = mem.cfg.H
    K = mem.cfg.K
    d = mem.cfg.key_dim
    bytes_per_elem = 4 if mem.dtype == torch.float32 else 2
    bytes_touched = T * H * K * d * bytes_per_elem
    bandwidth_gb_s = bytes_touched / mean_time / (1024**3)
    
    return {
        "method": "read_batch (vectorized)",
        "device": str(device),
        "T": T,
        "mean_sec": mean_time,
        "std_sec": std_time,
        "ci95_low": mean_time - ci95,
        "ci95_high": mean_time + ci95,
        "bandwidth_gb_s": bandwidth_gb_s,
        "bytes_touched": bytes_touched,
    }


def main():
    """Run CUDA benchmarks."""
    seed_everything(42)
    random.seed(42)
    
    # Test configurations
    configs = [
        {"T": 1000, "num_blocks": 2**14, "block_size": 256, "H": 4, "K": 32, "d": 64},
        {"T": 10000, "num_blocks": 2**14, "block_size": 256, "H": 4, "K": 32, "d": 64},
        {"T": 100000, "num_blocks": 2**14, "block_size": 256, "H": 4, "K": 32, "d": 64},
    ]
    
    num_warmup = 50
    num_repeats = 200
    
    print("=" * 80)
    print("CUDA Batch Addressing Performance Benchmark")
    print("=" * 80)
    print(f"Warmup iterations: {num_warmup}")
    print(f"Timed repeats: {num_repeats}")
    print()
    
    for cfg_dict in configs:
        T = cfg_dict["T"]
        B = cfg_dict["num_blocks"]
        L = cfg_dict["block_size"]
        H = cfg_dict["H"]
        K = cfg_dict["K"]
        d = cfg_dict["d"]
        
        print(f"Config: T={T}, B={B}, L={L}, H={H}, K={K}, d={d}")
        print("-" * 80)
        
        # Setup
        addr_cfg = AddressConfig(
            num_blocks=B,
            block_size=L,
            K=K,
            H=H,
            master_seed=42,
        )
        addresser = BlockAddress(addr_cfg)
        
        mem_cfg = MemoryConfig(
            num_blocks=B,
            block_size=L,
            key_dim=d,
            K=K,
            H=H,
            dtype="float32",
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_values="none",
            read_mode="raw_mean",
            master_seed=42,
        )
        mem = BBPMMemory(mem_cfg)
        device = torch.device(mem_cfg.device)
        
        # Generate test data
        hx_list = [random.randint(0, 2**64 - 1) for _ in range(T)]
        # Convert uint64 to int64 representation for PyTorch (handles values >= 2^63)
        hx_list_i64 = [u64_to_i64(hx) for hx in hx_list]
        hx_tensor = torch.tensor(hx_list_i64, dtype=torch.long, device=device)
        values = torch.randn(T, d, device=device)
        
        # Benchmark addresses_batch
        print("\nAddressing:")
        batch_addr = benchmark_addresses_batch(addresser, hx_tensor, device, num_warmup, num_repeats)
        print(f"  {batch_addr['method']:30s} {batch_addr['mean_sec']*1000:8.3f} ms  "
              f"({batch_addr['ci95_low']*1000:.3f}-{batch_addr['ci95_high']*1000:.3f} ms CI95)")
        print(f"    Throughput: {batch_addr['ops_per_sec']:,.0f} keys/sec")
        
        # Compare with scalar loop (only for very small T to avoid timeout)
        # Note: Scalar loop is extremely slow for large T, so we skip it for T > 100
        if T <= 100:
            print("  Computing scalar loop comparison (this may take a while)...")
            scalar_addr = benchmark_addresses_scalar_loop(addresser, hx_list, device, num_warmup, num_repeats)
            print(f"  {scalar_addr['method']:30s} {scalar_addr['mean_sec']*1000:8.3f} ms  "
                  f"({scalar_addr['ci95_low']*1000:.3f}-{scalar_addr['ci95_high']*1000:.3f} ms CI95)")
            speedup = scalar_addr['mean_sec'] / batch_addr['mean_sec']
            print(f"    Speedup: {speedup:.2f}x")
        else:
            print(f"  (Skipping scalar loop comparison for T={T} - too slow, use T<=100 to compare)")
        
        # Benchmark write_batch
        print("\nWrite:")
        write_result = benchmark_write_batch(mem, hx_tensor, values, device, num_warmup, num_repeats)
        print(f"  {write_result['method']:30s} {write_result['mean_sec']*1000:8.3f} ms  "
              f"({write_result['ci95_low']*1000:.3f}-{write_result['ci95_high']*1000:.3f} ms CI95)")
        print(f"    Bandwidth: {write_result['bandwidth_gb_s']:.2f} GB/s")
        if device.type == "cuda":
            print(f"    Peak memory: {write_result['peak_memory_mb']:.2f} MB")
        
        # Benchmark read_batch
        print("\nRead:")
        read_result = benchmark_read_batch(mem, hx_tensor, device, num_warmup, num_repeats)
        print(f"  {read_result['method']:30s} {read_result['mean_sec']*1000:8.3f} ms  "
              f"({read_result['ci95_low']*1000:.3f}-{read_result['ci95_high']*1000:.3f} ms CI95)")
        print(f"    Bandwidth: {read_result['bandwidth_gb_s']:.2f} GB/s")
        
        print()
    
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
