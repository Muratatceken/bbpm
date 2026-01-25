# BBPM Experiments Guide

This document describes how to run the ICML-grade experiment suite for BBPM.

## Running Experiments

All experiments are run via the canonical CLI:

```bash
python -m bbpm.experiments.run --exp <exp_id> [options]
```

### Common Arguments

All experiments support these common arguments:

- `--exp`: Experiment ID (required): `exp01`, `exp02`, `exp03`, `exp04`, `exp05`, `exp06`, or `exp07`
- `--out_dir`: Output directory (default: `artifacts`)
- `--device`: Device to use: `cpu` or `cuda` (default: `cpu`)
- `--seeds`: Number of random seeds for trials (default: `10`)
- `--dtype`: Data type: `float32` or `bfloat16` (default: `float32`)

### Experiment-Specific Arguments

Each experiment has additional arguments. Use `--help` to see them:

```bash
python -m bbpm.experiments.run --exp exp01 --help
```

## Experiments

### Exp01: SNR Scaling

Measures signal-to-noise ratio as a function of memory capacity.

```bash
python -m bbpm.experiments.run --exp exp01 --device cpu --seeds 10 --N_values 500 1000 3000 6000 10000
```

**Outputs:**
- `artifacts/metrics/exp01.json`
- `artifacts/figures/exp01_snr_scaling.pdf`

**Metrics:** Cosine similarity, MSE, estimated noise variance, measured occupancy vs N

### Exp02: K/H Ablation

Studies retrieval quality and collision behavior across K and H parameters.

```bash
python -m bbpm.experiments.run --exp exp02 --device cpu --seeds 10
```

**Outputs:**
- `artifacts/metrics/exp02.json`
- `artifacts/figures/exp02_k_h_ablation.pdf`

**Metrics:** Retrieval quality, exact self-collision rate (per hash family), cross-item collision rate (from global addresses), block occupancy skew

### Exp03: Runtime vs Attention

Benchmarks BBPM addressing, gather/scatter, and end-to-end performance vs attention.

```bash
python -m bbpm.experiments.run --exp exp03 --device cuda --seeds 1
```

**Outputs:**
- `artifacts/metrics/exp03.json`
- `artifacts/figures/exp03_runtime_vs_attention.pdf`

**Metrics:** Runtime vs sequence length (with CI95), peak memory vs sequence length. Separates addressing-only, gather bandwidth, and end-to-end costs. Uses batch APIs for vectorized GPU operations.

### Exp04: Needle-in-Haystack

Tests retrieval quality as a function of distance and load.

```bash
python -m bbpm.experiments.run --exp exp04 --device cpu --seeds 10
```

**Outputs:**
- `artifacts/metrics/exp04.json`
- `artifacts/figures/exp04_needle.pdf`

**Metrics:** Retrieval vs distance (fixed load), retrieval vs load for two fixed distances (0 and 500). Distance protocol keeps total load constant while varying temporal distance.

### Exp05: End-to-End Associative Recall

Full training experiment comparing BBPM to transformer baselines.

```bash
python -m bbpm.experiments.run --exp exp05 --device cpu --seeds 5
```

**Outputs:**
- `artifacts/metrics/exp05.json`
- `artifacts/figures/exp05_end2end_assoc.pdf`

**Metrics:** Training loss curves, test accuracy curves, parameter counts (with ±5% control). Uses token IDs as inputs with shared embedding layer and classification task (exact match accuracy).

### Exp06: Occupancy Skew

Studies memory behavior under non-uniform key distributions (uniform, Zipf).

```bash
python -m bbpm.experiments.run --exp exp06 --device cpu --seeds 10
```

**Outputs:**
- `artifacts/metrics/exp06.json`
- `artifacts/figures/exp06_occupancy_skew.pdf`

**Metrics:** Block occupancy distribution (histograms), Gini coefficient, collision rate vs skew (from global addresses). Supports uniform, Zipf, and token-ID modes. Deterministic with local NumPy RNG.

### Exp07: Drift and Reachability

Tests reachability under different keying modes as embeddings drift.

```bash
python -m bbpm.experiments.run --exp exp07 --device cpu --seeds 10
```

**Outputs:**
- `artifacts/metrics/exp07.json`
- `artifacts/figures/exp07_drift_reachability.pdf`

**Metrics:** Reachability (success rate) vs training step for three keying modes (token-ID, frozen projection, trainable projection), key change rate vs step, mean Hamming distance vs step. Gradual drift model with deterministic noise.

## Output Format

### Metrics JSON Schema

All experiments produce JSON files with the following structure:

```json
{
  "experiment_id": "exp01",
  "experiment_name": "SNR scaling",
  "timestamp": "2026-01-25T01:00:00",
  "git_commit": "abc123...",
  "hardware": {
    "device": "cpu",
    "torch_version": "2.0.0",
    "cuda_version": null,
    "gpu_name": null
  },
  "config": {...},
  "seeds": [0, 1, 2, ...],
  "raw_trials": [...],
  "summary": {...}
}
```

**Schema Requirements:**
- All experiments must include all top-level keys
- `summary` uses string keys (e.g., `"N=5000"`, `"K=32|H=4|N=1000"`) for JSON compatibility
- `raw_trials` contains per-seed, per-configuration results
- `summary` contains aggregated statistics with mean ± 95% CI where applicable

### Figures

All figures are saved as PDFs with:
- Vector-friendly format (PDF)
- Non-interactive backend (Agg) for reproducibility
- Hardware/version footer (torch version, CUDA info, git commit)
- Confidence intervals (CI95) where multi-seed data exists
- Clean legends, axis labels, and titles
- ICML-grade formatting standards

## Reproducibility

### Deterministic Execution

All experiments use deterministic seeding:

- Same seed → bit-identical numeric outputs (timestamps may differ)
- Uses `bbpm.utils.seeds.seed_everything(seed)` for each trial
- NumPy RNG uses `np.random.default_rng(seed)` for local, deterministic generation
- Never uses Python `hash()` or global `np.random` state
- Verified: Exp06 and Exp07 produce identical numeric arrays across runs with same seeds

### Running with Specific Seeds

The `--seeds` argument controls the number of seeds used. Seeds are `[0, 1, 2, ..., seeds-1]`.

To reproduce a specific result:
1. Note the seed from the JSON output (`raw_trials[].seed`)
2. Modify the experiment code to use only that seed, or
3. Filter the JSON output by seed

## CPU vs CUDA

### CPU Execution

All experiments can run on CPU:

```bash
python -m bbpm.experiments.run --exp exp01 --device cpu
```

- Slower but universally available
- No CUDA dependencies required
- Memory tracking limited (no peak memory stats)

### CUDA Execution

For GPU-accelerated execution:

```bash
python -m bbpm.experiments.run --exp exp03 --device cuda
```

**Requirements:**
- CUDA-capable GPU
- PyTorch with CUDA support
- CUDA drivers installed

**Benefits:**
- Faster execution (especially Exp03)
- Peak memory tracking available
- CUDA synchronization for accurate timing

**Note:** Exp03 (runtime benchmark) is most relevant for CUDA, as it measures performance.

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'bbpm'`:

1. Ensure you're running from the project root
2. Install the package: `pip install -e .`
3. Or set PYTHONPATH: `PYTHONPATH=src python -m bbpm.experiments.run ...`

### CUDA Errors

If CUDA is requested but unavailable:

- Use `--device cpu` instead
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Errors

If experiments run out of memory:

- Reduce `--seeds` (fewer trials)
- Reduce experiment-specific parameters (e.g., `--N_values` for exp01)
- Use `--dtype bfloat16` to reduce memory usage

## Running All Experiments

Use the Makefile:

```bash
make experiments
```

This runs all 7 experiments sequentially with default settings.
