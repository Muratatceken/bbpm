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

## Experiment Features

All experiments now include:
- **Multiple seeds** (`seeds: [0, 1, 2]` by default) for error bars when `num_seeds >= 3`
- **Compact diagnostics**: `q2_estimate`, `max_load`, `collision_rate`, `occupancy_summary` (no full-length arrays)
- **N/D axes**: Where applicable, plots use N/D (load ratio) instead of just N
- **Error bars**: Mean ± std across seeds when available

## Experiment Descriptions

### exp01_capacity_scaling
Replicates the "cos sim fidelity vs N" trend. Sweeps N over increasing values using normalized random unit vectors. Reports mean cosine similarity over test subset. Now includes multiple seeds for error bars and plots cosine similarity vs N/D (load ratio).

**Sweep axes**: `item_counts` (N values), `seeds` (for error bars)
**Diagnostics**: `q2_estimate`, `max_load`, `collision_rate` via `occupancy_summary`
**Output**: `capacity_vs_fidelity.png` (with error bars and N/D axis)

### exp02_ablation_K_H
Reproduces K and multi-hash ablation. Uses small D to force collisions. Sweeps K in [4, 16, 64], H in [1, 3]. Scores cosine sim threshold accuracy AND average cosine similarity.

**Output**: `ablation_K_H.png`

### exp03_block_vs_global
Compares BBPMAddressing (PRP-based) vs GlobalAffineHash at same D, N, K, H. Reports fidelity, collision stats, occupancy skew. Now includes full `occupancy_summary` diagnostics and a second plot showing diagnostics (q2_estimate or max_load) vs N.

**Sweep axes**: `N_values`
**Diagnostics**: Full `occupancy_summary` (q2_estimate, max_load, collision_rate, gini_load, self_collision_prob)
**Output**: `block_vs_global.png` (two-panel: fidelity and diagnostics)

### exp04_kv_memory_scaling
Produces plot showing memory scaling: KV cache memory ∝ layers * heads * T * head_dim (analytic estimate) vs BBPM memory = D*d (constant in T). Optionally measures actual CUDA memory if available.

**Output**: `kv_vs_bbpm_memory.png`

### exp05_needle_haystack
Simulates long context retrieval with degradation regimes. Sweeps both D (memory size) and N (items) to show performance degradation at high load ratios. Writes N key/value pairs, queries uniformly sampled IDs. Reports success vs N/D (load ratio) with error bars, compares to "window baseline" that only remembers last W items.

**Sweep axes**: `D_list` (memory sizes), `N_list` (item counts, filtered by `max_N_over_D`), `seeds` (for error bars)
**Diagnostics**: `q2_estimate`, `max_load`, `collision_rate`, `load_ratio = (N*K*H)/D`, `N_over_D = N/D`
**Output**: `needle_accuracy.png` (two-panel: success vs N/D and success vs q2_estimate, with error bars)

### exp06_drift_stability
Simulates key drift. Defines keys as hash(proj(x)) where x changes over time steps. Two modes: stable keys (fixed IDs) and drifting keys (x updated with noise). Reports retrieval accuracy over time.

**Output**: `drift_stability.png`

### exp07_llm_integration
Controller-based retrieval injection for streaming associative recall beyond context window. Uses stress sweep (D_list, N_list, K_list) to show degradation regimes. Uses Hugging Face transformers with small causal LM (default: gpt2). Properly measures tokens/sec (generation time only) and reports median/p90 latency. Can skip if transformers unavailable (runs in synthetic mode).

**Sweep axes**: `D_list` (memory sizes), `N_list` (fact counts), `K_list` (sparsity factors), `H_list` (hash counts), `seeds` (for error bars)
**Stress configs**: Fix K=16, H=1, sweep N at D=50k and D=100k; add K=64 run to show collapse
**Diagnostics**: `q2_estimate`, `max_load`, `collision_rate`, `load_ratio`, `N_over_D`
**Timing**: `tokens_per_sec` (generation only), `median_latency_ms`, `p90_latency_ms`
**Output**: `llm_accuracy_vs_N.png` (two-panel: accuracy vs N/D and tokens/sec vs N, with error bars)
