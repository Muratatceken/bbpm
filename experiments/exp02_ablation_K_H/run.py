"""Run experiment 2: Ablation study on K and H."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, get_device, self_collision_prob, set_global_seed
from bbpm.config import load_config
from bbpm.utils import get_logger, Timer

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run K and H ablation experiment."""
    config = load_config(config_path)
    D = config["D"]
    d = config["d"]
    N = config["N"]
    K_values = config["K_values"]
    H_values = config["H_values"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    threshold = config["cosine_threshold"]
    seed = config["seed"]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp02", log_file=outdir / "log.txt")

    logger.info(f"Starting exp02_ablation_K_H")
    logger.info(f"Config: D={D}, d={d}, N={N}")

    results = {
        "config": config,
        "results": {},
    }

    for H in H_values:
        accuracies = []
        avg_cosines = []
        self_collision_probs = []

        for K in K_values:
            logger.info(f"Testing H={H}, K={K}")

            memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)
            memory.clear()

            # Generate data
            keys = torch.arange(N, device=device_str)
            values = torch.randn(N, d, device=device_str)
            values = F.normalize(values, p=2, dim=1)

            # Write
            with Timer(f"Write N={N} with K={K}, H={H}"):
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    memory.write(keys[i:end], values[i:end])

            # Test
            test_n = min(test_size, N)
            retrieved = []
            for i in range(0, test_n, batch_size):
                end = min(i + batch_size, test_n)
                retrieved.append(memory.read(keys[i:end]))

            retrieved = torch.cat(retrieved, dim=0)
            true_vals = values[:test_n]

            # Compute metrics
            cos_sim = F.cosine_similarity(retrieved, true_vals, dim=1)
            avg_cosine = cos_sim.mean().item()
            accuracy = (cos_sim > threshold).float().mean().item()

            accuracies.append(accuracy)
            avg_cosines.append(avg_cosine)
            
            # Compute self-collision probability (for global hash, L = D)
            self_coll_prob = self_collision_prob(K, D)
            self_collision_probs.append(self_coll_prob)

            logger.info(f"H={H}, K={K}: Accuracy={accuracy:.4f}, AvgCos={avg_cosine:.4f}, SelfCollProb={self_coll_prob:.6f}")

        results["results"][f"H={H}"] = {
            "accuracies": accuracies,
            "avg_cosines": avg_cosines,
            "self_collision_probs": self_collision_probs,
        }

    results["K_values"] = K_values
    results["H_values"] = H_values

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp02: Ablation K and H")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp02_ablation_K_H")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
