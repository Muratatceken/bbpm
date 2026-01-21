"""Run experiment 6: Drift stability."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, get_device, set_global_seed
from bbpm.config import load_config
from bbpm.utils import get_logger, Timer

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run drift stability experiment."""
    config = load_config(config_path)
    D = config["D"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    num_steps = config["num_steps"]
    drift_noise_scale = config["drift_noise_scale"]
    stable_key_id = config["stable_key_id"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    seed = config["seed"]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp06", log_file=outdir / "log.txt")

    logger.info(f"Starting exp06_drift_stability")

    # Projection for key generation
    proj = nn.Linear(d, d).to(device_str)

    results = {
        "config": config,
        "steps": [],
        "stable_accuracy": [],
        "drifting_accuracy": [],
    }

    # Initialize memory
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)

    # Stable key: fixed embedding
    stable_x = torch.randn(1, d, device=device_str)
    stable_x = F.normalize(stable_x, p=2, dim=1)
    stable_key = (proj(stable_x.detach()).sum() * 1000).long() % (2 ** 31)
    stable_value = torch.randn(1, d, device=device_str)
    stable_value = F.normalize(stable_value, p=2, dim=1)

    # Drifting key: starts same but updates
    drifting_x = stable_x.clone()
    drifting_value = torch.randn(1, d, device=device_str)
    drifting_value = F.normalize(drifting_value, p=2, dim=1)

    for step in range(num_steps):
        memory.clear()

        # Write stable key
        memory.write(stable_key.unsqueeze(0), stable_value)

        # Write drifting key (update x with noise)
        drifting_x = drifting_x + drift_noise_scale * torch.randn_like(drifting_x)
        drifting_x = F.normalize(drifting_x, p=2, dim=1)
        drifting_key = (proj(drifting_x.detach()).sum() * 1000).long() % (2 ** 31)

        # Write some other keys for noise
        noise_keys = torch.randint(0, D, (test_size,), device=device_str)
        noise_values = torch.randn(test_size, d, device=device_str)
        noise_values = F.normalize(noise_values, p=2, dim=1)

        for i in range(0, test_size, batch_size):
            end = min(i + batch_size, test_size)
            memory.write(noise_keys[i:end], noise_values[i:end])

        # Test retrieval
        # Stable key
        retrieved_stable = memory.read(stable_key.unsqueeze(0))
        stable_acc = F.cosine_similarity(retrieved_stable, stable_value, dim=1).item()

        # Drifting key (use original value for comparison)
        retrieved_drifting = memory.read(drifting_key.unsqueeze(0))
        # Compare to original drifting value (before drift)
        drifting_acc = F.cosine_similarity(retrieved_drifting, drifting_value, dim=1).item()

        results["steps"].append(step)
        results["stable_accuracy"].append(stable_acc)
        results["drifting_accuracy"].append(drifting_acc)

        if step % 10 == 0:
            logger.info(f"Step {step}: Stable={stable_acc:.4f}, Drifting={drifting_acc:.4f}")

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp06: Drift stability")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp06_drift_stability")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
