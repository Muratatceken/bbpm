"""Microbenchmark: timing vs K, D, dtype."""

import argparse
import json
import time
from pathlib import Path

import torch

from bbpm import BBPMMemoryFloat, get_device, set_global_seed
from bbpm.utils import get_logger

logger = get_logger("microbench")


def benchmark_write_read(K_values, D_values, dtypes, d=64, H=1, device="cpu", num_trials=10):
    """Benchmark write/read performance."""
    results = []

    for D in D_values:
        for K in K_values:
            for dtype in dtypes:
                logger.info(f"Benchmarking D={D}, K={K}, dtype={dtype}")

                try:
                    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, dtype=dtype, device=device)
                    memory.clear()

                    # Generate test data
                    N = min(10000, D // 10)  # Reasonable load
                    keys = torch.arange(N, device=device)
                    values = torch.randn(N, d, device=device, dtype=dtype)

                    # Warmup
                    memory.write(keys[:100], values[:100])
                    memory.read(keys[:100])

                    # Benchmark write
                    write_times = []
                    for _ in range(num_trials):
                        memory.clear()
                        start = time.perf_counter()
                        memory.write(keys, values)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        write_times.append(time.perf_counter() - start)

                    # Benchmark read
                    read_times = []
                    for _ in range(num_trials):
                        start = time.perf_counter()
                        _ = memory.read(keys)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        read_times.append(time.perf_counter() - start)

                    avg_write = sum(write_times) / len(write_times)
                    avg_read = sum(read_times) / len(read_times)

                    results.append({
                        "D": D,
                        "K": K,
                        "dtype": str(dtype),
                        "avg_write_time": avg_write,
                        "avg_read_time": avg_read,
                        "write_throughput": N / avg_write,
                        "read_throughput": N / avg_read,
                    })

                except Exception as e:
                    logger.warning(f"Failed D={D}, K={K}, dtype={dtype}: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microbenchmark write/read")
    parser.add_argument("--out", type=Path, default=Path("../../results/benchmarks/microbench.json"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--K", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--D", type=int, nargs="+", default=[10000, 100000, 1000000])
    parser.add_argument("--dtypes", type=str, nargs="+", default=["float32"])

    args = parser.parse_args()

    logger = get_logger("microbench")

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    dtypes = [dtype_map[d] for d in args.dtypes]

    results = benchmark_write_read(
        K_values=args.K,
        D_values=args.D,
        dtypes=dtypes,
        device=args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.out}")
