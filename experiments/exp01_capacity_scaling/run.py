"""Run experiment 1: Capacity scaling."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, compute_capacity_metrics, get_device, occupancy_summary, query_hit_analysis, set_global_seed
from bbpm.hashing.diagnostics import slot_loads
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
    block_size = config.get("block_size", 1024)
    item_counts = config["item_counts"]
    test_size = config["test_size"]
    batch_size = config["batch_size"]
    seeds = config.get("seeds", [0, 1, 2])

    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp01", log_file=outdir / "log.txt")

    logger.info(f"Starting exp01_capacity_scaling")
    logger.info(f"Config: D={D}, d={d}, K={K}, H={H}, seeds={seeds}")
    logger.info(f"Device: {device_str}")

    results = {
        "config": config,
        "seeds": {},
    }

    for seed in seeds:
        set_global_seed(seed)
        logger.info(f"Running seed={seed}")

        results["seeds"][f"seed_{seed}"] = {
            "item_counts": [],
            "cosine_similarities": [],
            "N_over_D": [],
            "diagnostics": [],
        }

        # Initialize memory with PRP-based addressing
        memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, block_size=block_size, device=device_str, seed=seed)

        # Warmup GPU (only once per seed, before first operation)
        if device_str == "cuda" and seed == seeds[0]:
            logger.info("  Warming up GPU...")
            warmup_keys = torch.arange(100, device=device_str)
            warmup_values = torch.randn(100, d, device=device_str)
            warmup_values = F.normalize(warmup_values, p=2, dim=1)
            memory.write(warmup_keys, warmup_values)
            _ = memory.read(warmup_keys)
            memory.clear()
            if device_str == "cuda":
                torch.cuda.synchronize()

        for N in item_counts:
            logger.info(f"  Testing N={N}")
            memory.clear()

            # Generate keys and values
            keys = torch.arange(N, device=device_str)
            values = torch.randn(N, d, device=device_str)
            values = F.normalize(values, p=2, dim=1)  # Normalize to unit vectors

            # Batch write
            with Timer(f"Write {N} items", device=device_str):
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    memory.write(keys[i:end], values[i:end])

            # Get diagnostics
            all_indices = memory.hash_fn.indices(keys, K, H)
            occ_summary = occupancy_summary(all_indices.flatten(), D)

            # Compute capacity metrics
            cap_metrics = compute_capacity_metrics(N, D, K, H)

            # Test retrieval
            test_n = min(test_size, N)
            retrieved = []
            for i in range(0, test_n, batch_size):
                end = min(i + batch_size, test_n)
                retrieved.append(memory.read(keys[i:end]))

            retrieved = torch.cat(retrieved, dim=0)
            true_vals = values[:test_n]

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(retrieved, true_vals, dim=1)
            cos_sim_mean = cos_sim.mean().item()

            # Failure diagnostics (when degradation detected)
            # Skip for very large D to avoid slowdown
            failure_diagnostics = None
            if cos_sim_mean < 0.7 and D <= 2000000:  # Skip for D > 2M
                # Get slot loads
                slot_loads_array = slot_loads(all_indices.flatten(), D)
                
                # Top-10 most loaded slots
                top_slots = occ_summary.get("top_slots", [])[:10]
                
                # Query hit analysis (use test keys, only for reasonable test sizes)
                if test_n <= 10000:
                    test_keys = keys[:test_n]
                    test_indices_tensor = memory.hash_fn.indices(test_keys, K, H)
                    hit_analysis = query_hit_analysis(test_indices_tensor, slot_loads_array)
                else:
                    hit_analysis = {}
                
                # SNR proxy
                snr_proxy = 1.0 / np.sqrt(max(1e-9, cap_metrics["load_ratio"]))
                
                failure_diagnostics = {
                    "top_10_slots": top_slots,
                    "query_hit_analysis": hit_analysis,
                    "snr_proxy": float(snr_proxy),
                }
                
                logger.info(
                    f"  Degradation detected: cosine={cos_sim_mean:.4f}, "
                    f"max_load={occ_summary['max_load']}, snr_proxy={snr_proxy:.4f}"
                )

            diagnostics_dict = {
                "q2_estimate": occ_summary["q2_estimate"],
                "max_load": occ_summary["max_load"],
                "collision_rate": occ_summary["collision_rate"],
                "load_ratio": cap_metrics["load_ratio"],
                "capacity_units": cap_metrics["capacity_units"],
                "effective_capacity": cap_metrics["effective_capacity"],
            }
            if failure_diagnostics is not None:
                diagnostics_dict["failure_diagnostics"] = failure_diagnostics

            results["seeds"][f"seed_{seed}"]["item_counts"].append(N)
            results["seeds"][f"seed_{seed}"]["cosine_similarities"].append(cos_sim_mean)
            results["seeds"][f"seed_{seed}"]["N_over_D"].append(N / D)
            results["seeds"][f"seed_{seed}"]["diagnostics"].append(diagnostics_dict)

            logger.info(
                f"  N={N}: Cosine similarity = {cos_sim_mean:.4f}, "
                f"capacity_units={cap_metrics['capacity_units']:.4f}, "
                f"q2={occ_summary['q2_estimate']:.6f}"
            )

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
