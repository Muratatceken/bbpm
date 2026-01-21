"""Run experiment 4: KV cache vs BBPM memory scaling."""

import argparse
import json
from pathlib import Path

import torch

from bbpm import BBPMMemoryFloat, get_device, set_global_seed
from bbpm.config import load_config
from bbpm.utils import get_logger

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run KV vs BBPM memory scaling experiment."""
    config = load_config(config_path)
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    batch_size = config["batch_size"]
    dtype_bytes = config["dtype_bytes"]
    D = config["D"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    context_lengths = config["context_lengths"]
    seed = config["seed"]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp04", log_file=outdir / "log.txt")

    logger.info(f"Starting exp04_kv_memory_scaling")

    results = {
        "config": config,
        "context_lengths": [],
        "kv_memory_gb": [],
        "bbpm_memory_gb": [],
        "kv_oom": None,
    }

    # BBPM memory size (constant)
    bbpm_memory_bytes = D * d * 4  # float32 = 4 bytes
    bbpm_memory_gb = bbpm_memory_bytes / (1024 ** 3)

    logger.info(f"BBPM memory: {bbpm_memory_gb:.4f} GB (constant)")

    # Initialize BBPM (for measurement if CUDA)
    if device_str == "cuda" and torch.cuda.is_available():
        bbpm = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / (1024 ** 3)
    else:
        initial_mem = 0

    for T in context_lengths:
        logger.info(f"Testing context length T={T}")

        # KV cache memory (analytic)
        # KV cache: [layers, batch, heads, seq_len, head_dim] for K and V
        kv_elements = num_layers * batch_size * num_heads * T * head_dim * 2  # *2 for K and V
        kv_memory_bytes = kv_elements * dtype_bytes
        kv_memory_gb = kv_memory_bytes / (1024 ** 3)

        results["context_lengths"].append(T)
        results["kv_memory_gb"].append(kv_memory_gb)

        # BBPM memory (constant)
        if device_str == "cuda" and torch.cuda.is_available():
            # Measure actual GPU memory
            current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            bbpm_measured = current_mem - initial_mem
            results["bbpm_memory_gb"].append(bbpm_measured)
        else:
            results["bbpm_memory_gb"].append(bbpm_memory_gb)

        logger.info(f"T={T}: KV={kv_memory_gb:.2f} GB, BBPM={results['bbpm_memory_gb'][-1]:.4f} GB")

        # Check for OOM (simulated)
        if device_str == "cuda" and torch.cuda.is_available():
            try:
                # Try to allocate KV cache
                test_k = torch.randn(
                    num_layers, batch_size, num_heads, T, head_dim,
                    dtype=torch.float16, device=device_str
                )
                test_v = torch.randn(
                    num_layers, batch_size, num_heads, T, head_dim,
                    dtype=torch.float16, device=device_str
                )
                del test_k, test_v
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results["kv_oom"] = T
                    logger.warning(f"OOM at T={T}")
                    break

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp04: KV vs BBPM scaling")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp04_kv_memory_scaling")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
