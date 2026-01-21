"""Run experiment 1: Capacity scaling."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, get_device, set_global_seed
from bbpm.config import load_config
from bbpm.utils import get_logger, Timer

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run capacity scaling experiment."""
    # Load config
    config = load_config(config_path)
    D = config["D"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    item_counts = config["item_counts"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    seed = config["seed"]

    # Setup
    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp01", log_file=outdir / "log.txt")

    logger.info(f"Starting exp01_capacity_scaling")
    logger.info(f"Config: D={D}, d={d}, K={K}, H={H}")
    logger.info(f"Device: {device_str}")

    # Initialize memory
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)

    results = {
        "config": config,
        "item_counts": [],
        "cosine_similarities": [],
        "metrics": [],
    }

    for N in item_counts:
        logger.info(f"Testing N={N}")
        memory.clear()

        # Generate keys and values
        keys = torch.arange(N, device=device_str)
        values = torch.randn(N, d, device=device_str)
        values = F.normalize(values, p=2, dim=1)  # Normalize to unit vectors

        # Batch write
        with Timer(f"Write {N} items"):
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                memory.write(keys[i:end], values[i:end])

        # Test retrieval
        test_n = min(test_size, N)
        retrieved = []
        for i in range(0, test_n, batch_size):
            end = min(i + batch_size, test_n)
            retrieved.append(memory.read(keys[i:end]))

        retrieved = torch.cat(retrieved, dim=0)
        true_vals = values[:test_n]

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(retrieved, true_vals, dim=1).mean().item()

        results["item_counts"].append(N)
        results["cosine_similarities"].append(cos_sim)

        logger.info(f"N={N}: Cosine similarity = {cos_sim:.4f}")

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp01: Capacity scaling")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "results" / "exp01_capacity_scaling",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use",
    )

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
