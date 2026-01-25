.PHONY: lint test experiments experiments-paper experiments-fast all clean

lint:
	@echo "Running ruff..."
	ruff check src/ tests/
	@echo "Running black --check..."
	black --check src/ tests/
	@echo "Running mypy..."
	mypy src/bbpm/

test:
	@echo "Running pytest..."
	pytest -q tests/

# Paper-grade configurations (for ICML submission)
# Canonical memory config: B=16384, L=256, H=4, K=32, d=64, seeds=10
experiments-paper:
	@echo "Running all experiments with paper-grade configurations..."
	@echo "Exp01: SNR scaling (seeds=10, N=[2k,4k,8k,16k,32k,48k,64k,80k])"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp01 --device cpu --seeds 10 --N_values 2000 4000 8000 16000 32000 48000 64000 80000 || exit 1
	@echo "Exp02: K/H ablation (seeds=10, N=[2k,8k,16k,32k,48k])"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp02 --device cpu --seeds 10 --N_values 2000 8000 16000 32000 48000 || exit 1
	@echo "Exp03: Runtime vs Attention (CUDA, seeds=1, T=[256,512,1024,2048,4096])"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp03 --device cuda --seeds 1 --T_values 256 512 1024 2048 4096 || exit 1
	@echo "Exp04: Needle-in-haystack (seeds=10, fixed_N=32k, distance=[0,128,512,2048,8192,16384,32768], N=[2k,8k,16k,32k,48k,64k,80k])"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp04 --device cpu --seeds 10 --N_values 2000 8000 16000 32000 48000 64000 80000 --distance_values 0 128 512 2048 8192 16384 32768 || exit 1
	@echo "Exp05: End-to-end associative recall (seeds=5, vocab=50k, T=256, epochs=10)"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp05 --device cpu --seeds 5 --vocab_size 50000 --T 256 --num_epochs 10 --batch_size 64 || exit 1
	@echo "Exp06: Occupancy skew (seeds=10, N=64k, vocab=100k, s=[0.0,0.5,1.0,1.2,1.5,2.0])"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp06 --device cpu --seeds 10 --N 64000 --vocab_size 100000 --s_values 0.0 0.5 1.0 1.2 1.5 2.0 || exit 1
	@echo "Exp07: Drift and reachability (seeds=10, num_early_items=2048, num_steps=200)"
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp07 --device cpu --seeds 10 --num_early_items 2048 --num_steps 200 || exit 1
	@echo "All paper-grade experiments completed!"

# Fast "camera-ready sanity" configurations (for quick reruns/debugging)
experiments-fast:
	@echo "Running all experiments with fast configurations (seeds=3)..."
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp01 --device cpu --seeds 3 --N_values 2000 8000 16000 32000 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp02 --device cpu --seeds 3 --N_values 2000 8000 16000 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp03 --device cpu --seeds 1 --T_values 256 512 1024 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp04 --device cpu --seeds 3 --N_values 2000 8000 16000 --distance_values 0 128 512 2048 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp05 --device cpu --seeds 2 --vocab_size 1000 --T 64 --num_epochs 5 --batch_size 32 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp06 --device cpu --seeds 3 --N 32000 --vocab_size 10000 --s_values 0.0 1.0 1.5 2.0 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp07 --device cpu --seeds 3 --num_early_items 512 --num_steps 50 || exit 1
	@echo "All fast experiments completed!"

# Default: use fast configs for CI/testing
experiments: experiments-fast

all: lint test experiments-fast
	@echo "All checks passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
