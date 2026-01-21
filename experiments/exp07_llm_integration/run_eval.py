"""Run experiment 7: LLM integration evaluation with controller-based retrieval injection."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, get_device, set_global_seed
from bbpm.config import load_config
from bbpm.utils import get_logger

# Compute project root for robust paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def run_experiment(config_path: Path, outdir: Path, device: str = "auto", N_override: int = None):
    """Run LLM integration experiment with controller-based retrieval injection."""
    config = load_config(config_path)
    D = config["D"]
    d_config = config.get("d", 64)  # Will be overridden by model.config.hidden_size if available
    K = config["K"]
    H = config["H"]
    model_name = config.get("model_name", "gpt2")
    window_size = config.get("window_size", 50)
    N_values = config.get("N_values", [100, 500, 1000, 2000, 5000])
    num_queries = config.get("num_queries", 20)
    seed = config.get("seed", 42)
    max_new_tokens = config.get("max_new_tokens", 20)

    if N_override is not None:
        N_values = [N_override]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp07", log_file=outdir / "log.txt")

    logger.info(f"Starting exp07_llm_integration")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")

    results = {
        "config": config,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "N_values": [],
        "bbpm_accuracy": [],
        "baseline_accuracy": [],
        "oracle_accuracy": [],  # Optional upper bound
        "tokens_per_sec": [],
        "memory_scaling": [],
    }

    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers not available. Running in synthetic scoring mode.")
        logger.warning("This mode only tests retrieval correctness, not LLM generation.")
        
        # Synthetic mode: test retrieval correctness only
        for N in N_values:
            logger.info(f"Testing N={N} (synthetic mode - retrieval only)")
            
            # Use d from config for synthetic mode
            d = d_config
            memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)
            memory.clear()
            
            # Generate deterministic codebook
            set_global_seed(seed)
            codebook = torch.randn(N, d, device=device_str)
            codebook = F.normalize(codebook, p=2, dim=1)
            
            # Generate fact values
            fact_values = [f"val_{i:05d}" for i in range(N)]
            
            # Write facts to BBPM: key=ID, value=codebook[ID]
            batch_size = 100
            fact_ids = torch.arange(N, device=device_str)
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                keys = fact_ids[i:end]
                values = codebook[i:end]
                memory.write(keys, values)
            
            # Test queries
            query_ids = torch.randint(0, N, (num_queries,), device=device_str)
            correct_bbpm = 0
            
            for q_id in query_ids:
                q_id_item = q_id.item()
                
                # BBPM retrieval
                retrieved_emb = memory.read(q_id.unsqueeze(0))  # [1, d]
                retrieved_emb = F.normalize(retrieved_emb, p=2, dim=1)
                
                # Decode: argmax(cos(v_hat, codebook))
                similarities = torch.matmul(retrieved_emb, codebook.t())  # [1, N]
                decoded_id = similarities.argmax().item()
                
                if decoded_id == q_id_item:
                    correct_bbpm += 1
            
            bbpm_acc = correct_bbpm / num_queries
            baseline_acc = 0.0  # No baseline in synthetic mode
            oracle_acc = 1.0  # Perfect retrieval in synthetic mode
            
            results["N_values"].append(N)
            results["bbpm_accuracy"].append(bbpm_acc)
            results["baseline_accuracy"].append(baseline_acc)
            results["oracle_accuracy"].append(oracle_acc)
            results["tokens_per_sec"].append(0.0)
            results["memory_scaling"].append({"bbpm": "O(1)", "prompt": "O(W)", "model": "O(1)"})
            
            logger.info(f"N={N}: BBPM retrieval accuracy={bbpm_acc:.4f}")
        
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
        # Re-run in synthetic mode
        run_experiment(config_path, outdir, device, N_override)
        return

    # Override d with model's hidden size
    d = model.config.hidden_size
    logger.info(f"Using d={d} (model.config.hidden_size)")

    # Initialize BBPM
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)

    for N in N_values:
        logger.info(f"Testing N={N}")

        memory.clear()

        # Generate deterministic codebook with fixed seed
        set_global_seed(seed)
        codebook = torch.randn(N, d, device=device_str)
        codebook = F.normalize(codebook, p=2, dim=1)

        # Generate fact values: ID=<i> VALUE=<val_xxxxx>
        fact_values = [f"val_{i:05d}" for i in range(N)]
        fact_texts = [f"ID={i} VALUE={fact_values[i]}" for i in range(N)]

        # Stream N facts to BBPM
        start_time = time.time()
        batch_size = 100
        fact_ids = torch.arange(N, device=device_str)
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            keys = fact_ids[i:end]
            # Write codebook vectors as values
            values = codebook[i:end]
            memory.write(keys, values)

        elapsed = time.time() - start_time
        tokens_per_sec = N / elapsed if elapsed > 0 else 0

        # Test queries
        query_ids = torch.randint(0, N, (num_queries,), device=device_str)
        correct_bbpm = 0
        correct_baseline = 0
        correct_oracle = 0

        for q_id in query_ids:
            q_id_item = q_id.item()
            target_value = fact_values[q_id_item]

            # Build baseline prompt: last W facts + question
            window_facts = fact_texts[max(0, N - window_size):]
            baseline_prompt = "\n".join(window_facts)
            baseline_prompt += f"\nQuestion: What is VALUE for ID={q_id_item}? Answer:"

            # BBPM retrieval
            retrieved_emb = memory.read(q_id.unsqueeze(0))  # [1, d]
            retrieved_emb = F.normalize(retrieved_emb, p=2, dim=1)

            # Decode: argmax(cos(v_hat, codebook))
            similarities = torch.matmul(retrieved_emb, codebook.t())  # [1, N]
            decoded_id = similarities.argmax().item()
            retrieved_fact_text = fact_texts[decoded_id]

            # Build BBPM prompt: last W facts + Retrieved fact + question
            bbpm_prompt = "\n".join(window_facts)
            bbpm_prompt += f"\nRetrieved: {retrieved_fact_text}"
            bbpm_prompt += f"\nQuestion: What is VALUE for ID={q_id_item}? Answer:"

            # Oracle prompt: inject true fact (upper bound)
            oracle_prompt = "\n".join(window_facts)
            oracle_prompt += f"\nRetrieved: {fact_texts[q_id_item]}"
            oracle_prompt += f"\nQuestion: What is VALUE for ID={q_id_item}? Answer:"

            # Generate with model
            with torch.no_grad():
                # Baseline
                baseline_tokens = tokenizer(baseline_prompt, return_tensors="pt").to(device_str)
                baseline_output = model.generate(
                    **baseline_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

                # BBPM
                bbpm_tokens = tokenizer(bbpm_prompt, return_tensors="pt").to(device_str)
                bbpm_output = model.generate(
                    **bbpm_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                bbpm_text = tokenizer.decode(bbpm_output[0], skip_special_tokens=True)

                # Oracle
                oracle_tokens = tokenizer(oracle_prompt, return_tensors="pt").to(device_str)
                oracle_output = model.generate(
                    **oracle_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                oracle_text = tokenizer.decode(oracle_output[0], skip_special_tokens=True)

            # Score success: correct if output contains target_value as substring
            if target_value in baseline_text:
                correct_baseline += 1
            if target_value in bbpm_text:
                correct_bbpm += 1
            if target_value in oracle_text:
                correct_oracle += 1

        bbpm_acc = correct_bbpm / num_queries
        baseline_acc = correct_baseline / num_queries
        oracle_acc = correct_oracle / num_queries

        # Memory scaling: analytic estimate
        memory_scaling = {
            "bbpm": "O(1) constant",
            "prompt": f"O(W={window_size}) window",
            "model": "O(1) constant",
        }

        results["N_values"].append(N)
        results["bbpm_accuracy"].append(bbpm_acc)
        results["baseline_accuracy"].append(baseline_acc)
        results["oracle_accuracy"].append(oracle_acc)
        results["tokens_per_sec"].append(tokens_per_sec)
        results["memory_scaling"].append(memory_scaling)

        logger.info(
            f"N={N}: BBPM={bbpm_acc:.4f}, Baseline={baseline_acc:.4f}, "
            f"Oracle={oracle_acc:.4f}, Tokens/sec={tokens_per_sec:.2f}"
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
    parser.add_argument("--N", type=int, default=None, help="Override N_values with single value")

    args = parser.parse_args()
    run_experiment(args.config, args.outdir, args.device, args.N)
