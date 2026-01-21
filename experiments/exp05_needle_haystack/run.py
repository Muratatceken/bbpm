"""Run experiment 5: Needle in a haystack."""

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
    """Run needle in haystack experiment."""
    config = load_config(config_path)
    D = config["D"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    N_values = config["N_values"]
    test_queries = config["test_queries"]
    window_size = config["window_size"]
    seed = config["seed"]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp05", log_file=outdir / "log.txt")

    logger.info(f"Starting exp05_needle_haystack")

    results = {
        "config": config,
        "N_values": [],
        "bbpm_success": [],
        "window_success": [],
    }

    for N in N_values:
        logger.info(f"Testing N={N}")

        # Initialize memory
        memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)
        memory.clear()

        # Generate N key-value pairs
        keys = torch.arange(N, device=device_str)
        values = torch.randn(N, d, device=device_str)
        values = F.normalize(values, p=2, dim=1)

        # Write all pairs
        batch_size = 10000
        with Timer(f"Write {N} pairs"):
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                memory.write(keys[i:end], values[i:end])

        # Query random subset
        query_indices = torch.randint(0, N, (test_queries,), device=device_str)
        query_keys = keys[query_indices]
        query_values = values[query_indices]

        # BBPM retrieval
        retrieved = []
        for i in range(0, test_queries, batch_size):
            end = min(i + batch_size, test_queries)
            retrieved.append(memory.read(query_keys[i:end]))

        retrieved = torch.cat(retrieved, dim=0)
        cos_sim = F.cosine_similarity(retrieved, query_values, dim=1)
        bbpm_success = (cos_sim > 0.9).float().mean().item()

        # Window baseline (only remembers last W items)
        window_success = (query_indices >= (N - window_size)).float().mean().item()

        results["N_values"].append(N)
        results["bbpm_success"].append(bbpm_success)
        results["window_success"].append(window_success)

        logger.info(f"N={N}: BBPM success={bbpm_success:.4f}, Window success={window_success:.4f}")

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp05: Needle in haystack")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp05_needle_haystack")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
