"""Benchmark: compare torch ops (and triton if available)."""

import argparse
import json
import time
from pathlib import Path

import torch

from bbpm.utils import get_logger

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def benchmark_kernels(D, d, K, N, device="cpu", num_trials=10):
    """Benchmark different kernel implementations."""
    results = {}

    # Generate test data
    keys = torch.arange(N, device=device)
    values = torch.randn(N, d, device=device)
    indices = torch.randint(0, D, (N * K,), device=device)

    # PyTorch index_add_
    memory_torch = torch.zeros(D, d, device=device)
    write_times = []
    for _ in range(num_trials):
        memory_torch.zero_()
        start = time.perf_counter()
        memory_torch.index_add_(0, indices, values.repeat_interleave(K, dim=0))
        if device == "cuda":
            torch.cuda.synchronize()
        write_times.append(time.perf_counter() - start)

    results["torch_index_add"] = {
        "avg_time": sum(write_times) / len(write_times),
        "throughput": N / (sum(write_times) / len(write_times)),
    }

    # PyTorch gather
    read_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        _ = memory_torch[indices[:N]]
        if device == "cuda":
            torch.cuda.synchronize()
        read_times.append(time.perf_counter() - start)

    results["torch_gather"] = {
        "avg_time": sum(read_times) / len(read_times),
        "throughput": N / (sum(read_times) / len(read_times)),
    }

    if TRITON_AVAILABLE and device == "cuda":
        logger.info("Triton available, but custom kernels not implemented yet")
        # Placeholder for triton kernels

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel benchmark")
    parser.add_argument("--out", type=Path, default=Path("../../results/benchmarks/kernel.json"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--D", type=int, default=1000000)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--N", type=int, default=10000)

    args = parser.parse_args()

    logger = get_logger("kernel_bench")
    logger.info(f"Triton available: {TRITON_AVAILABLE}")

    results = benchmark_kernels(
        D=args.D,
        d=args.d,
        K=args.K,
        N=args.N,
        device=args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.out}")
