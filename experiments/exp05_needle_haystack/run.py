"""Run experiment 5: Needle in a haystack with degradation regimes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, get_device, occupancy_summary, set_global_seed
from bbpm.config import load_config
from bbpm.utils import get_logger, Timer

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run needle in haystack experiment with D×N sweep."""
    config = load_config(config_path)
    D_list = config["D_list"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    N_list = config["N_list"]
    max_N_over_D = config["max_N_over_D"]
    test_queries = config["test_queries"]
    window_size = config["window_size"]
    seeds = config["seeds"]

    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp05", log_file=outdir / "log.txt")

    logger.info(f"Starting exp05_needle_haystack")
    logger.info(f"Config: D_list={D_list}, N_list={N_list}, seeds={seeds}")

    results = {
        "config": config,
        "seeds": {},
    }

    for seed in seeds:
        set_global_seed(seed)
        logger.info(f"Running seed={seed}")

        results["seeds"][f"seed_{seed}"] = {}

        for D in D_list:
            logger.info(f"  Testing D={D}")

            results["seeds"][f"seed_{seed}"][f"D_{D}"] = {
                "D": D,
                "runs": [],
            }

            # Filter N_list by max_N_over_D
            valid_N_list = [N for N in N_list if N <= max_N_over_D * D]

            for N in valid_N_list:
                logger.info(f"    Testing N={N} (N/D={N/D:.4f})")

                # Create fresh memory with seed
                memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str, seed=seed)
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

                # Get all write indices for diagnostics
                all_indices = memory.hash_fn.indices(keys, K, H)  # [N, K*H]
                indices_flat = all_indices.flatten()

                # Compute diagnostics
                occ_summary = occupancy_summary(indices_flat, D)
                load_ratio = (N * K * H) / D
                N_over_D = N / D

                # Query uniformly sampled IDs from [0, N)
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
                cosine_mean = cos_sim.mean().item()

                # Window baseline (only remembers last W items)
                # Success = 1 if queried ID in [N-W, N) else 0
                window_success = (query_indices >= (N - window_size)).float().mean().item()

                # Store results
                run_data = {
                    "N": N,
                    "N_over_D": N_over_D,
                    "load_ratio": load_ratio,
                    "bbpm_success": bbpm_success,
                    "cosine_mean": cosine_mean,
                    "window_success": window_success,
                    "q2_estimate": occ_summary["q2_estimate"],
                    "max_load": occ_summary["max_load"],
                    "collision_rate": occ_summary["collision_rate"],
                    "unique_slots_touched": occ_summary["unique_slots_touched"],
                }

                results["seeds"][f"seed_{seed}"][f"D_{D}"]["runs"].append(run_data)

                logger.info(
                    f"      N={N}: BBPM={bbpm_success:.4f}, Window={window_success:.4f}, "
                    f"q2={occ_summary['q2_estimate']:.6f}, max_load={occ_summary['max_load']}"
                )

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
