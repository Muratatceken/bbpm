"""Run experiment 5: Needle in a haystack with degradation regimes."""

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
    """Run needle in haystack experiment with D×N sweep."""
    config = load_config(config_path)
    D_list = config["D_list"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    N_list = config["N_list"]
    max_N_over_D = config["max_N_over_D"]
    test_queries = config["test_queries"]
    window_size = config.get("window_size", 1000)  # Default for backward compatibility
    window_sizes = config.get("window_sizes", [256, 1000, 4000, 16000])
    cosine_threshold = config.get("cosine_threshold", 0.7)
    seeds = config["seeds"]

    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp05", log_file=outdir / "log.txt")

    logger.info(f"Starting exp05_needle_haystack")
    logger.info(f"Config: D_list={D_list}, N_list={N_list}, seeds={seeds}")

    results = {
        "config": config,
        "seeds": {},
    }

    # Warmup GPU (only once, before first operation)
    if device_str == "cuda":
        logger.info("Warming up GPU...")
        warmup_memory = BBPMMemoryFloat(D=10000, d=d, K=K, H=H, device=device_str, seed=seeds[0])
        warmup_keys = torch.arange(100, device=device_str)
        warmup_values = torch.randn(100, d, device=device_str)
        warmup_values = F.normalize(warmup_values, p=2, dim=1)
        warmup_memory.write(warmup_keys, warmup_values)
        _ = warmup_memory.read(warmup_keys)
        del warmup_memory
        torch.cuda.synchronize()

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
                with Timer(f"Write {N} pairs", device=device_str):
                    for i in range(0, N, batch_size):
                        end = min(i + batch_size, N)
                        memory.write(keys[i:end], values[i:end])

                # Get all write indices for diagnostics
                all_indices = memory.hash_fn.indices(keys, K, H)  # [N, K*H]
                indices_flat = all_indices.flatten()

                # Compute diagnostics
                occ_summary = occupancy_summary(indices_flat, D)
                cap_metrics = compute_capacity_metrics(N, D, K, H)
                load_ratio = cap_metrics["load_ratio"]
                capacity_units = cap_metrics["capacity_units"]
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
                cos_sim_np = cos_sim.cpu().numpy()
                bbpm_success = (cos_sim > cosine_threshold).float().mean().item()
                cosine_mean = cos_sim.mean().item()

                # Multiple window baselines (only remembers last W items)
                # Success = 1 if queried ID in [N-W, N) else 0
                window_successes = {}
                for W in window_sizes:
                    if W < N:  # Skip if W >= N (would be trivial 1.0)
                        window_successes[f"window_success_W_{W}"] = (
                            (query_indices >= (N - W)).float().mean().item()
                        )

                # Oracle baseline (hash-table perfect map)
                oracle_map = {int(key.item()): value.cpu() for key, value in zip(keys, values)}
                oracle_correct = 0
                for q_idx, q_key in enumerate(query_keys):
                    q_key_int = int(q_key.item())
                    if q_key_int in oracle_map:
                        oracle_correct += 1
                oracle_success = oracle_correct / test_queries if test_queries > 0 else 0.0

                # Cosine similarity histogram for representative load ratios
                # Save histogram for capacity_units near {0.1, 0.5, 1.0, 2.0}
                cosine_hist = None
                representative_cap_units = [0.1, 0.5, 1.0, 2.0]
                save_hist = False
                for rep_cap in representative_cap_units:
                    if abs(capacity_units - rep_cap) < 0.05:  # Within 5% of representative point
                        save_hist = True
                        break

                if save_hist:
                    hist_counts, hist_bins = np.histogram(cos_sim_np, bins=30, range=(0.0, 1.0))
                    cosine_hist = {
                        "bins": hist_bins.tolist(),
                        "counts": hist_counts.tolist(),
                    }

                # Failure diagnostics (when degradation detected)
                failure_diagnostics = None
                if bbpm_success < 0.5 or cosine_mean < 0.7:
                    # Get slot loads
                    slot_loads_array = slot_loads(indices_flat, D)
                    
                    # Top-10 most loaded slots
                    top_slots = occ_summary.get("top_slots", [])[:10]
                    
                    # Query hit analysis
                    query_indices_tensor = memory.hash_fn.indices(query_keys, K, H)
                    hit_analysis = query_hit_analysis(query_indices_tensor, slot_loads_array)
                    
                    # SNR proxy
                    snr_proxy = 1.0 / np.sqrt(max(1e-9, load_ratio))
                    
                    failure_diagnostics = {
                        "top_10_slots": top_slots,
                        "query_hit_analysis": hit_analysis,
                        "snr_proxy": float(snr_proxy),
                    }
                    
                    logger.info(
                        f"      Degradation detected: bbpm_success={bbpm_success:.4f}, "
                        f"cosine_mean={cosine_mean:.4f}, max_load={occ_summary['max_load']}, "
                        f"snr_proxy={snr_proxy:.4f}"
                    )
                    if top_slots:
                        logger.info(f"      Top loaded slots: {top_slots[:5]}")

                # Store results
                run_data = {
                    "N": N,
                    "N_over_D": N_over_D,
                    "load_ratio": load_ratio,
                    "capacity_units": capacity_units,
                    "effective_capacity": cap_metrics["effective_capacity"],
                    "bbpm_success": bbpm_success,
                    "cosine_mean": cosine_mean,
                    "oracle_success": oracle_success,
                    "q2_estimate": occ_summary["q2_estimate"],
                    "max_load": occ_summary["max_load"],
                    "collision_rate": occ_summary["collision_rate"],
                    "unique_slots_touched": occ_summary["unique_slots_touched"],
                }
                # Add window successes
                run_data.update(window_successes)
                # Add cosine histogram if computed
                if cosine_hist is not None:
                    run_data["cosine_hist"] = cosine_hist
                # Add failure diagnostics if computed
                if failure_diagnostics is not None:
                    run_data["failure_diagnostics"] = failure_diagnostics

                results["seeds"][f"seed_{seed}"][f"D_{D}"]["runs"].append(run_data)

                # Log summary (use default window_size if available, otherwise show oracle)
                window_summary = ""
                if window_size < N and f"window_success_W_{window_size}" in window_successes:
                    window_summary = f"Window(W={window_size})={window_successes[f'window_success_W_{window_size}']:.4f}"
                elif window_successes:
                    # Show first available window size
                    first_w_key = list(window_successes.keys())[0]
                    window_summary = f"Window({first_w_key.replace('window_success_W_', 'W=')})={window_successes[first_w_key]:.4f}"
                else:
                    window_summary = f"Oracle={oracle_success:.4f}"

                logger.info(
                    f"      N={N}: BBPM={bbpm_success:.4f}, {window_summary}, "
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
