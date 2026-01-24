#!/usr/bin/env python3
"""Micro-benchmark for BlockAddress addressing performance.

Measures CPU/CUDA performance and memory allocation for addresses_tensor()
vs addresses() reference implementation. Not part of unit test suite.
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
from bbpm.utils.seeds import seed_everything


def benchmark_addresses(
    addresser: BlockAddress,
    hx_list: list[int],
    device: torch.device,
    num_warmup: int = 10,
) -> Dict[str, Any]:
    """Benchmark addresses() list-based implementation."""
    # Warmup
    for hx in hx_list[:num_warmup]:
        _ = addresser.addresses(hx)
    
    # Time measurement
    start = time.perf_counter()
    for i, hx in enumerate(hx_list):
        _ = addresser.addresses(hx)
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i + 1}/{len(hx_list)} calls", end="\r", flush=True)
    if len(hx_list) >= 1000:
        print()  # New line after progress
    elapsed = time.perf_counter() - start
    
    return {
        "method": "addresses()",
        "device": str(device),
        "calls": len(hx_list),
        "total_sec": elapsed,
        "avg_usec": (elapsed / len(hx_list)) * 1e6,
        "ops_per_sec": len(hx_list) / elapsed,
    }


def benchmark_addresses_tensor(
    addresser: BlockAddress,
    hx_list: list[int],
    device: torch.device,
    num_warmup: int = 10,
) -> Dict[str, Any]:
    """Benchmark addresses_tensor() implementation."""
    # Warmup
    for hx in hx_list[:num_warmup]:
        _ = addresser.addresses_tensor(hx, device)
    
    # CUDA memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    
    # Time measurement
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i, hx in enumerate(hx_list):
        _ = addresser.addresses_tensor(hx, device)
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i + 1}/{len(hx_list)} calls", end="\r", flush=True)
    if len(hx_list) >= 1000:
        print()  # New line after progress
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    result = {
        "method": "addresses_tensor()",
        "device": str(device),
        "calls": len(hx_list),
        "total_sec": elapsed,
        "avg_usec": (elapsed / len(hx_list)) * 1e6,
        "ops_per_sec": len(hx_list) / elapsed,
    }
    
    if device.type == "cuda":
        mem_after = torch.cuda.memory_allocated()
        mem_delta = mem_after - mem_before
        mem_peak = torch.cuda.max_memory_allocated()
        result["mem_delta_bytes"] = mem_delta
        result["mem_peak_bytes"] = mem_peak
        result["mem_delta_mb"] = mem_delta / (1024**2)
        result["mem_peak_mb"] = mem_peak / (1024**2)
    
    return result


def main():
    """Run addressing benchmarks."""
    seed_everything(42)
    random.seed(42)
    
    # Test configurations: large realistic scenarios
    configs = [
        {"num_blocks": 2**14, "block_size": 256, "H": 4, "K": 32},
        {"num_blocks": 2**14, "block_size": 256, "H": 4, "K": 64},
        {"num_blocks": 2**14, "block_size": 256, "H": 8, "K": 64},
        {"num_blocks": 2**14, "block_size": 256, "H": 8, "K": 128},
    ]
    
    # Reduced iterations for faster testing (can be increased for more accurate results)
    num_iterations = 1000
    num_warmup = 10
    
    print("=" * 80)
    print("BlockAddress Addressing Performance Benchmark")
    print("=" * 80)
    print(f"Iterations per config: {num_iterations}")
    print(f"Warmup iterations: {num_warmup}")
    print()
    print("Note: For small workloads, addresses() may be faster due to tensor overhead.")
    print("      addresses_tensor() benefits increase with larger K and batch processing.")
    print("      CUDA has launch overhead; benefits appear with larger workloads.")
    print()
    
    for cfg_dict in configs:
        cfg = AddressConfig(
            num_blocks=cfg_dict["num_blocks"],
            block_size=cfg_dict["block_size"],
            K=cfg_dict["K"],
            H=cfg_dict["H"],
            master_seed=42,
        )
        addresser = BlockAddress(cfg)
        
        # Generate random hx values
        hx_list = [random.randint(0, 2**64 - 1) for _ in range(num_iterations)]
        
        print(f"Config: B={cfg.num_blocks}, L={cfg.block_size}, H={cfg.H}, K={cfg.K}")
        print("-" * 80)
        
        # CPU benchmarks
        print("\nCPU Benchmarks:")
        ref_cpu = benchmark_addresses(addresser, hx_list, torch.device("cpu"), num_warmup)
        vec_cpu = benchmark_addresses_tensor(addresser, hx_list, torch.device("cpu"), num_warmup)
        
        print(f"  {ref_cpu['method']:25s} {ref_cpu['avg_usec']:8.2f} μs/call  {ref_cpu['ops_per_sec']:10.0f} ops/sec")
        print(f"  {vec_cpu['method']:25s} {vec_cpu['avg_usec']:8.2f} μs/call  {vec_cpu['ops_per_sec']:10.0f} ops/sec")
        speedup = ref_cpu['avg_usec'] / vec_cpu['avg_usec']
        if speedup >= 1.0:
            print(f"  Speedup: {speedup:.2f}x (tensor faster)")
        else:
            print(f"  Speedup: {speedup:.2f}x (list faster, tensor overhead for small K)")
        
        # CUDA benchmarks (if available)
        if torch.cuda.is_available():
            print("\nCUDA Benchmarks:")
            vec_cuda = benchmark_addresses_tensor(
                addresser, hx_list, torch.device("cuda"), num_warmup
            )
            print(f"  {vec_cuda['method']:25s} {vec_cuda['avg_usec']:8.2f} μs/call  {vec_cuda['ops_per_sec']:10.0f} ops/sec")
            print(f"  Memory delta: {vec_cuda['mem_delta_mb']:.2f} MB")
            print(f"  Peak memory:  {vec_cuda['mem_peak_mb']:.2f} MB")
            
            if vec_cuda['mem_delta_bytes'] < 1024:  # < 1KB per 10k calls
                print("  ✓ Low allocation churn (masks cached)")
            else:
                print(f"  ⚠ Allocation churn detected: {vec_cuda['mem_delta_bytes']} bytes")
        
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
