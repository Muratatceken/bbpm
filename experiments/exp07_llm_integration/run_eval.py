"""Run experiment 7: LLM integration evaluation with controller-based retrieval injection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, compute_capacity_metrics, get_device, occupancy_summary, query_hit_analysis, set_global_seed
from bbpm.hashing.diagnostics import slot_loads
from bbpm.config import load_config
from bbpm.utils import get_logger

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def run_experiment(config_path: Path, outdir: Path, device: str = "auto"):
    """Run LLM integration experiment with stress sweep and proper timing."""
    config = load_config(config_path)
    D_list = config.get("D_list", [50000, 100000, 200000])
    d_config = config.get("d", 64)  # Will be overridden by model.config.hidden_size if available
    K_list = config.get("K_list", [4, 16, 64])
    H_list = config.get("H_list", [1])
    model_name = config.get("model_name", "gpt2")
    window_size = config.get("window_size", 50)
    N_list = config.get("N_list", [1000, 2000, 5000, 10000, 20000])
    num_queries = config.get("num_queries", 30)
    seeds = config.get("seeds", [0, 1, 2])
    max_new_tokens = config.get("max_new_tokens", 16)

    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp07", log_file=outdir / "log.txt")

    logger.info(f"Starting exp07_llm_integration")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    logger.info(f"Config: D_list={D_list}, N_list={N_list}, K_list={K_list}, seeds={seeds}")

    results = {
        "config": config,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "seeds": {},
    }

    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers not available. Running in synthetic scoring mode.")
        logger.warning("This mode only tests retrieval correctness, not LLM generation.")
        
        # Synthetic mode: test retrieval correctness only
        for seed in seeds:
            set_global_seed(seed)
            results["seeds"][f"seed_{seed}"] = {}
            
            for D in D_list:
                for K in K_list:
                    for H in H_list:
                        for N in N_list:
                            if N > 5 * D:  # Skip if too large
                                continue
                                
                            logger.info(f"Testing seed={seed}, D={D}, K={K}, H={H}, N={N} (synthetic)")
                            
                            d = d_config
                            memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str, seed=seed)
                            memory.clear()
                            
                            # Generate deterministic codebook
                            set_global_seed(seed)
                            codebook = torch.randn(N, d, device=device_str)
                            codebook = F.normalize(codebook, p=2, dim=1)
                            
                            # Write facts to BBPM
                            batch_size = 100
                            fact_ids = torch.arange(N, device=device_str)
                            for i in range(0, N, batch_size):
                                end = min(i + batch_size, N)
                                memory.write(fact_ids[i:end], codebook[i:end])
                            
                            # Get diagnostics
                            all_indices = memory.hash_fn.indices(fact_ids, K, H)
                            occ_summary = occupancy_summary(all_indices.flatten(), D)
                            
                            # Test queries
                            query_ids = torch.randint(0, N, (num_queries,), device=device_str)
                            correct_bbpm = 0
                            
                            for q_id in query_ids:
                                retrieved_emb = memory.read(q_id.unsqueeze(0))
                                retrieved_emb = F.normalize(retrieved_emb, p=2, dim=1)
                                similarities = torch.matmul(retrieved_emb, codebook.t())
                                decoded_id = similarities.argmax().item()
                                if decoded_id == q_id.item():
                                    correct_bbpm += 1
                            
                            bbpm_acc = correct_bbpm / num_queries
                            key = f"D_{D}_K_{K}_H_{H}_N_{N}"
                            results["seeds"][f"seed_{seed}"][key] = {
                                "D": D, "K": K, "H": H, "N": N,
                                "N_over_D": N / D,
                                "load_ratio": (N * K * H) / D,
                                "bbpm_accuracy": bbpm_acc,
                                "q2_estimate": occ_summary["q2_estimate"],
                                "max_load": occ_summary["max_load"],
                                "collision_rate": occ_summary["collision_rate"],
                            }
        
        # Save results
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {outdir / 'metrics.json'}")
        logger.info("Note: Synthetic mode completed. Install transformers for full LLM evaluation.")
        return

    # Real LLM mode
    logger.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device_str)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Falling back to synthetic mode")
        run_experiment(config_path, outdir, device)
        return

    # Override d with model's hidden size
    d = model.config.hidden_size
    logger.info(f"Using d={d} (model.config.hidden_size)")

    # Stress sweep: Fix K=16, H=1, sweep N at D=50k and D=100k; add K=64 run
    stress_configs = [
        (50000, 16, 1),  # D=50k, K=16, H=1
        (100000, 16, 1),  # D=100k, K=16, H=1
        (100000, 64, 1),  # D=100k, K=64, H=1 (collapse case)
    ]

    # Warmup GPU (only once, before first operation)
    if device_str == "cuda":
        logger.info("Warming up GPU...")
        warmup_memory = BBPMMemoryFloat(D=10000, d=d, K=16, H=1, device=device_str, seed=seeds[0])
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

        for D, K, H in stress_configs:
            if D not in D_list or K not in K_list or H not in H_list:
                continue

            logger.info(f"  Testing D={D}, K={K}, H={H}")

            for N in N_list:
                if N > 5 * D:  # Skip if too large
                    continue

                logger.info(f"    Testing N={N} (N/D={N/D:.4f})")

                # Create fresh memory
                memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str, seed=seed)
                memory.clear()

                # Generate deterministic codebook
                set_global_seed(seed)
                codebook = torch.randn(N, d, device=device_str)
                codebook = F.normalize(codebook, p=2, dim=1)

                # Generate fact values
                fact_values = [f"val_{i:05d}" for i in range(N)]
                fact_texts = [f"ID={i} VALUE={fact_values[i]}" for i in range(N)]

                # Write facts to BBPM
                batch_size = 100
                fact_ids = torch.arange(N, device=device_str)
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    memory.write(fact_ids[i:end], codebook[i:end])

                # Get diagnostics
                all_indices = memory.hash_fn.indices(fact_ids, K, H)
                occ_summary = occupancy_summary(all_indices.flatten(), D)
                cap_metrics = compute_capacity_metrics(N, D, K, H)
                load_ratio = cap_metrics["load_ratio"]
                capacity_units = cap_metrics["capacity_units"]
                N_over_D = N / D

                # Test queries with proper timing
                query_ids = torch.randint(0, N, (num_queries,), device=device_str)
                correct_bbpm = 0
                correct_baseline = 0
                correct_oracle = 0
                latencies = []
                total_generated_tokens = 0
                total_time = 0.0

                for q_id in query_ids:
                    q_id_item = q_id.item()
                    target_value = fact_values[q_id_item]

                    # Build prompts
                    window_facts = fact_texts[max(0, N - window_size):]
                    baseline_prompt = "\n".join(window_facts)
                    baseline_prompt += f"\nQuestion: What is VALUE for ID={q_id_item}? Answer:"

                    # BBPM retrieval
                    retrieved_emb = memory.read(q_id.unsqueeze(0))
                    retrieved_emb = F.normalize(retrieved_emb, p=2, dim=1)
                    similarities = torch.matmul(retrieved_emb, codebook.t())
                    decoded_id = similarities.argmax().item()
                    retrieved_fact_text = fact_texts[decoded_id]

                    bbpm_prompt = "\n".join(window_facts)
                    bbpm_prompt += f"\nRetrieved: {retrieved_fact_text}"
                    bbpm_prompt += f"\nQuestion: What is VALUE for ID={q_id_item}? Answer:"

                    oracle_prompt = "\n".join(window_facts)
                    oracle_prompt += f"\nRetrieved: {fact_texts[q_id_item]}"
                    oracle_prompt += f"\nQuestion: What is VALUE for ID={q_id_item}? Answer:"

                    # Generate with proper timing (ONLY generation time)
                    with torch.no_grad():
                        # BBPM (main timing measurement)
                        bbpm_tokens = tokenizer(bbpm_prompt, return_tensors="pt").to(device_str)
                        input_ids = bbpm_tokens["input_ids"]
                        
                        # Synchronize CUDA before timing
                        if device_str == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        bbpm_output = model.generate(
                            **bbpm_tokens,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        # Synchronize CUDA after generation
                        if device_str == "cuda":
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        
                        gen_len = bbpm_output.shape[1] - input_ids.shape[1]
                        latencies.append((t1 - t0) * 1000)  # ms
                        total_generated_tokens += gen_len
                        total_time += (t1 - t0)
                        
                        bbpm_text = tokenizer.decode(bbpm_output[0], skip_special_tokens=True)

                        # Baseline
                        baseline_tokens = tokenizer(baseline_prompt, return_tensors="pt").to(device_str)
                        baseline_output = model.generate(
                            **baseline_tokens,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

                        # Oracle
                        oracle_tokens = tokenizer(oracle_prompt, return_tensors="pt").to(device_str)
                        oracle_output = model.generate(
                            **oracle_tokens,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        oracle_text = tokenizer.decode(oracle_output[0], skip_special_tokens=True)

                    # Score success
                    if target_value in baseline_text:
                        correct_baseline += 1
                    if target_value in bbpm_text:
                        correct_bbpm += 1
                    if target_value in oracle_text:
                        correct_oracle += 1

                bbpm_acc = correct_bbpm / num_queries
                baseline_acc = correct_baseline / num_queries
                oracle_acc = correct_oracle / num_queries

                # Compute timing stats
                tokens_per_sec = total_generated_tokens / total_time if total_time > 0 else 0.0
                median_latency_ms = np.median(latencies) if latencies else 0.0
                p90_latency_ms = np.percentile(latencies, 90) if latencies else 0.0

                # Failure diagnostics (when degradation detected)
                failure_diagnostics = None
                if bbpm_acc < 0.5:
                    # Get slot loads
                    slot_loads_array = slot_loads(all_indices.flatten(), D)
                    
                    # Top-10 most loaded slots
                    top_slots = occ_summary.get("top_slots", [])[:10]
                    
                    # Query hit analysis
                    query_indices_tensor = memory.hash_fn.indices(query_ids, K, H)
                    hit_analysis = query_hit_analysis(query_indices_tensor, slot_loads_array)
                    
                    # SNR proxy
                    snr_proxy = 1.0 / np.sqrt(max(1e-9, load_ratio))
                    
                    failure_diagnostics = {
                        "top_10_slots": top_slots,
                        "query_hit_analysis": hit_analysis,
                        "snr_proxy": float(snr_proxy),
                    }
                    
                    logger.info(
                        f"      Degradation detected: bbpm_acc={bbpm_acc:.4f}, "
                        f"max_load={occ_summary['max_load']}, snr_proxy={snr_proxy:.4f}"
                    )

                # Store results
                key = f"D_{D}_K_{K}_H_{H}_N_{N}"
                result_dict = {
                    "D": D,
                    "K": K,
                    "H": H,
                    "N": N,
                    "N_over_D": N_over_D,
                    "load_ratio": load_ratio,
                    "capacity_units": capacity_units,
                    "effective_capacity": cap_metrics["effective_capacity"],
                    "bbpm_accuracy": bbpm_acc,
                    "baseline_accuracy": baseline_acc,
                    "oracle_accuracy": oracle_acc,
                    "tokens_per_sec": tokens_per_sec,
                    "median_latency_ms": median_latency_ms,
                    "p90_latency_ms": p90_latency_ms,
                    "q2_estimate": occ_summary["q2_estimate"],
                    "max_load": occ_summary["max_load"],
                    "collision_rate": occ_summary["collision_rate"],
                }
                if failure_diagnostics is not None:
                    result_dict["failure_diagnostics"] = failure_diagnostics
                results["seeds"][f"seed_{seed}"][key] = result_dict

                logger.info(
                    f"      N={N}: BBPM={bbpm_acc:.4f}, Baseline={baseline_acc:.4f}, "
                    f"Oracle={oracle_acc:.4f}, Tokens/sec={tokens_per_sec:.2f}, "
                    f"q2={occ_summary['q2_estimate']:.6f}"
                )

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp07: LLM integration")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp07_llm_integration")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device)
