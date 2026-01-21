"""Run experiment 3: Block vs Global hash comparison."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, BlockHash, GlobalAffineHash, get_device, self_collision_prob, set_global_seed
from bbpm.config import load_config
from bbpm.hashing.diagnostics import collision_rate, gini_load, max_load
from bbpm.utils import get_logger, Timer

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run block vs global hash comparison."""
    config = load_config(config_path)
    D = config["D"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    block_size = config["block_size"]
    N_values = config["N_values"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    seed = config["seed"]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp03", log_file=outdir / "log.txt")

    logger.info(f"Starting exp03_block_vs_global")

    results = {
        "config": config,
        "N_values": N_values,
        "global": {"cosines": [], "collision_rates": [], "max_loads": [], "gini_loads": [], "self_collision_probs": []},
        "block": {"cosines": [], "collision_rates": [], "max_loads": [], "gini_loads": [], "self_collision_probs": []},
    }

    # Initialize hash functions
    global_hash = GlobalAffineHash(D, seed=seed)
    block_hash = BlockHash(D, block_size, seed=seed)

    for N in N_values:
        logger.info(f"Testing N={N}")

        # Generate data
        keys = torch.arange(N, device=device_str)
        values = torch.randn(N, d, device=device_str)
        values = F.normalize(values, p=2, dim=1)

        # Test Global Hash
        global_mem = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=global_hash, device=device_str)
        global_mem.clear()

        with Timer(f"Global write N={N}"):
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                global_mem.write(keys[i:end], values[i:end])

        # Test retrieval
        test_n = min(test_size, N)
        retrieved_global = []
        for i in range(0, test_n, batch_size):
            end = min(i + batch_size, test_n)
            retrieved_global.append(global_mem.read(keys[i:end]))
        retrieved_global = torch.cat(retrieved_global, dim=0)
        cos_global = F.cosine_similarity(retrieved_global, values[:test_n], dim=1).mean().item()

        # Diagnostics
        indices_global = global_hash.indices(keys[:test_n], K, H)
        results["global"]["cosines"].append(cos_global)
        results["global"]["collision_rates"].append(collision_rate(indices_global))
        results["global"]["max_loads"].append(max_load(indices_global.flatten(), D))
        results["global"]["gini_loads"].append(gini_load(indices_global.flatten(), D))
        # For global hash, L = D (total memory size)
        results["global"]["self_collision_probs"].append(self_collision_prob(K, D))

        # Test Block Hash
        block_mem = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=block_hash, device=device_str)
        block_mem.clear()

        with Timer(f"Block write N={N}"):
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                block_mem.write(keys[i:end], values[i:end])

        retrieved_block = []
        for i in range(0, test_n, batch_size):
            end = min(i + batch_size, test_n)
            retrieved_block.append(block_mem.read(keys[i:end]))
        retrieved_block = torch.cat(retrieved_block, dim=0)
        cos_block = F.cosine_similarity(retrieved_block, values[:test_n], dim=1).mean().item()

        indices_block = block_hash.indices(keys[:test_n], K, H)
        results["block"]["cosines"].append(cos_block)
        results["block"]["collision_rates"].append(collision_rate(indices_block))
        results["block"]["max_loads"].append(max_load(indices_block.flatten(), D))
        results["block"]["gini_loads"].append(gini_load(indices_block.flatten(), D))
        # For block hash, L = block_size
        results["block"]["self_collision_probs"].append(self_collision_prob(K, block_size))

        logger.info(f"N={N}: Global cos={cos_global:.4f}, Block cos={cos_block:.4f}")

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp03: Block vs Global")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp03_block_vs_global")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
