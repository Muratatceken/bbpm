"""Run experiment 7 speed test: tokens/sec vs context length."""

import argparse
import json
from pathlib import Path
import time

import torch

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


def run_speed_test(config_path: Path, outdir: Path, device: str = "auto"):
    """Run speed test for LLM integration."""
    config = load_config(config_path)
    D = config["D"]
    d = config["d"]
    K = config["K"]
    H = config["H"]
    model_name = config.get("model_name", "sshleifer/tiny-gpt2")
    context_lengths = config.get("context_lengths", [100, 500, 1000, 2000, 5000])
    seed = config["seed"]

    set_global_seed(seed)
    device_str = get_device(device if device != "auto" else config.get("device", "auto"))
    logger = get_logger("exp07_speed", log_file=outdir / "log.txt")

    logger.info(f"Starting exp07 speed test")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")

    results = {
        "config": config,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "context_lengths": [],
        "tokens_per_sec": [],
    }

    if TRANSFORMERS_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device_str)
        model.eval()

        memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device_str)

        for T in context_lengths:
            logger.info(f"Testing context length T={T}")

            memory.clear()

            start_time = time.time()

            for i in range(T):
                fact_text = f"ID={i} VALUE=fact_{i}"
                tokens = tokenizer(fact_text, return_tensors="pt").to(device_str)
                with torch.no_grad():
                    outputs = model.transformer(**tokens)
                    fact_emb = outputs.last_hidden_state.mean(dim=1)
                    fact_emb = torch.nn.functional.normalize(fact_emb, p=2, dim=1)

                key = torch.tensor([i], device=device_str)
                memory.write(key, fact_emb)

            elapsed = time.time() - start_time
            tokens_per_sec = T / elapsed if elapsed > 0 else 0

            results["context_lengths"].append(T)
            results["tokens_per_sec"].append(tokens_per_sec)

            logger.info(f"T={T}: {tokens_per_sec:.2f} tokens/sec")

    # Save results
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "speed_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {outdir / 'speed_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exp07 speed test")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "exp07_llm_integration")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()
    run_speed_test(args.config, args.outdir, args.device)
