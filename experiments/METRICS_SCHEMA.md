# Metrics Schema Documentation

This document describes the metrics fields stored in experiment results JSON files.

## Core Capacity Metrics

### `load_ratio`
- **Type**: `float`
- **Definition**: `(N * K * H) / D`
- **Description**: Total number of hash writes normalized by memory capacity. When `load_ratio > 1`, collisions dominate.
- **Units**: Dimensionless

### `capacity_units`
- **Type**: `float`
- **Definition**: `N / (D / (K * H)) = (N * K * H) / D = load_ratio`
- **Description**: Number of items stored normalized by theoretical capacity. The theoretical transition occurs at `capacity_units = 1.0`.
- **Units**: Dimensionless

### `effective_capacity`
- **Type**: `float`
- **Definition**: `D / (K * H)`
- **Description**: Theoretical capacity of the memory system. This is the expected number of items that can be stored before significant degradation.
- **Units**: Items

### `N_over_D`
- **Type**: `float`
- **Definition**: `N / D`
- **Description**: Simple load ratio without accounting for K and H. Less informative than `capacity_units` for capacity analysis.
- **Units**: Dimensionless

## Occupancy Diagnostics

### `q2_estimate`
- **Type**: `float`
- **Definition**: Sum of squared slot probabilities, computed from nonzero loads only: `∑ (load_i / total_writes)²` for `load_i > 0`
- **Description**: Proxy for collision probability. Higher values indicate more collisions.
- **Range**: `[0, 1]`

### `max_load`
- **Type**: `int`
- **Definition**: Maximum number of items hashing to a single slot
- **Description**: Indicates worst-case collision. High `max_load` suggests uneven distribution.

### `collision_rate`
- **Type**: `float`
- **Definition**: Fraction of duplicate indices within a batch (for `[B, K*H]` shape) or globally (for flat tensor)
- **Description**: Measures within-row or global collision frequency.
- **Range**: `[0, 1]`

### `unique_slots_touched`
- **Type**: `int`
- **Definition**: Number of distinct slots that received at least one write
- **Description**: Indicates memory utilization. Lower values suggest more collisions.

### `mean_load`
- **Type**: `float`
- **Definition**: Average load across all slots
- **Description**: Expected load per slot. Should be approximately `(N * K * H) / D`.

### `std_load`
- **Type**: `float`
- **Definition**: Standard deviation of slot loads
- **Description**: Measures load distribution spread. Higher values indicate more uneven distribution.

### `top_slots`
- **Type**: `List[Tuple[int, int]]`
- **Definition**: List of `(slot_index, load)` pairs for the top-K most loaded slots (default K=20)
- **Description**: Identifies hot spots in memory. Useful for failure diagnostics.

### `load_hist_bins`
- **Type**: `List[float]`
- **Definition**: Bin edges for load distribution histogram
- **Description**: Used for visualizing slot load distribution.

### `load_hist_counts`
- **Type**: `List[int]`
- **Definition**: Counts per bin for load distribution histogram
- **Description**: Used for visualizing slot load distribution.

## Performance Metrics

### `tokens_per_sec` (exp07 only)
- **Type**: `float`
- **Definition**: Number of tokens generated per second during LLM inference
- **Description**: Measures generation throughput. Only includes generation time (CUDA-synchronized).

### `median_latency_ms` (exp07 only)
- **Type**: `float`
- **Definition**: Median latency per generation in milliseconds
- **Description**: Measures typical generation latency.

### `p90_latency_ms` (exp07 only)
- **Type**: `float`
- **Definition**: 90th percentile latency per generation in milliseconds
- **Description**: Measures tail latency.

## Accuracy Metrics

### `bbpm_success` / `bbpm_accuracy`
- **Type**: `float`
- **Definition**: Fraction of queries with cosine similarity above threshold (default 0.7 for exp05, 0.9 for exp02)
- **Description**: Retrieval success rate for BBPM memory.
- **Range**: `[0, 1]`

### `cosine_mean` / `cosine_similarity`
- **Type**: `float`
- **Definition**: Mean cosine similarity between retrieved and true values
- **Description**: Average retrieval fidelity. Higher is better.
- **Range**: `[-1, 1]` (typically `[0, 1]` for normalized vectors)

### `baseline_accuracy` (exp07 only)
- **Type**: `float`
- **Definition**: Accuracy of baseline (window-based) retrieval
- **Description**: Comparison baseline for BBPM performance.

### `oracle_accuracy` / `oracle_success`
- **Type**: `float`
- **Definition**: Accuracy of oracle (perfect hash-table) retrieval
- **Description**: Upper bound on achievable accuracy. Should be 1.0 for all queries.

### `window_success` / `window_success_W_{W}`
- **Type**: `float`
- **Definition**: Success rate of window baseline with window size W
- **Description**: Baseline that only remembers last W items. Only computed when `W < N` (to avoid trivial 1.0).

## Failure Diagnostics

These fields are only present when degradation is detected (`bbpm_success < 0.5` or `cosine_mean < 0.7`).

### `failure_diagnostics`
- **Type**: `dict`
- **Contents**:
  - `top_10_slots`: `List[Tuple[int, int]]` - Top 10 most loaded slots
  - `query_hit_analysis`: `dict` - Query hit fractions for different load thresholds
    - `fraction_hit_load_gt_2`: Fraction of queries hitting slots with load > 2
    - `fraction_hit_load_gt_4`: Fraction of queries hitting slots with load > 4
    - `fraction_hit_load_gt_8`: Fraction of queries hitting slots with load > 8
    - `fraction_hit_load_gt_16`: Fraction of queries hitting slots with load > 16
  - `snr_proxy`: `float` - Signal-to-noise ratio proxy: `1.0 / sqrt(max(1e-9, load_ratio))`
- **Description**: Detailed diagnostics for understanding failure modes.

## Cosine Similarity Histogram (exp05 only)

### `cosine_hist`
- **Type**: `dict` (only present for representative capacity_units)
- **Contents**:
  - `bins`: `List[float]` - Histogram bin edges (30 bins, range [0, 1])
  - `counts`: `List[int]` - Counts per bin
- **Description**: Distribution of cosine similarities for representative load ratios. Only saved for `capacity_units` near {0.1, 0.5, 1.0, 2.0} to keep JSON size small.

## Notes

- All metrics are stored per seed in nested structures: `results["seeds"]["seed_{seed}"]`
- For multi-seed experiments, aggregate statistics (mean, std) are computed in analyze scripts
- JSON files are kept < 2MB by using compact diagnostics instead of full histograms
- Timing metrics use CUDA synchronization when `device == "cuda"` for accuracy
