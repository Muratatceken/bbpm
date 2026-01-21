# Experiments

This directory contains all experiments for the BBPM paper.

## Running Experiments

Each experiment follows a consistent structure:
- `run.py`: Executes the experiment and saves metrics to `results/expXX_*/metrics.json`
- `analyze.py`: Loads metrics and generates figures in `figures/expXX_*/`
- `config.yaml`: Configuration parameters

### Individual Experiment

```bash
cd experiments/exp01_capacity_scaling
python run.py --config config.yaml --outdir ../../results/exp01_capacity_scaling --device cpu
python analyze.py --results ../../results/exp01_capacity_scaling --outdir ../../figures/exp01_capacity_scaling
```

### All Experiments

```bash
# From project root
bash scripts/run_all_experiments.sh
```

## Experiment Descriptions

### exp01_capacity_scaling
Replicates the "cos sim fidelity vs N" trend. Sweeps N over increasing values using normalized random unit vectors. Reports mean cosine similarity over test subset.

**Output**: `capacity_vs_fidelity.png`

### exp02_ablation_K_H
Reproduces K and multi-hash ablation. Uses small D to force collisions. Sweeps K in [4, 16, 64], H in [1, 3]. Scores cosine sim threshold accuracy AND average cosine similarity.

**Output**: `ablation_K_H.png`

### exp03_block_vs_global
Compares GlobalAffineHash vs BlockPermutationHash at same D, N, K, H. Reports fidelity, collision stats, occupancy skew.

**Output**: `block_vs_global.png`

### exp04_kv_memory_scaling
Produces plot showing memory scaling: KV cache memory ∝ layers * heads * T * head_dim (analytic estimate) vs BBPM memory = D*d (constant in T). Optionally measures actual CUDA memory if available.

**Output**: `kv_vs_bbpm_memory.png`

### exp05_needle_haystack
Simulates long context retrieval. Writes N key/value pairs, queries random subset. Reports success vs N, compares to "window baseline" that only remembers last W items.

**Output**: `needle_accuracy.png`

### exp06_drift_stability
Simulates key drift. Defines keys as hash(proj(x)) where x changes over time steps. Two modes: stable keys (fixed IDs) and drifting keys (x updated with noise). Reports retrieval accuracy over time.

**Output**: `drift_stability.png`

### exp07_llm_integration
Controller-based retrieval injection for streaming associative recall beyond context window. Uses Hugging Face transformers with small causal LM (default: sshleifer/tiny-gpt2). Can skip if transformers unavailable.

**Output**: `llm_accuracy_vs_N.png`
