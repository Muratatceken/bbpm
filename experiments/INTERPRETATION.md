# Experiment Results Interpretation Guide

This document explains how to interpret BBPM experiment results, including degradation patterns, capacity scaling, and measurement artifacts.

## Why Degradation Occurs

BBPM degradation occurs due to two main factors:

### 1. Collisions
When multiple items hash to the same slot, their values are superposed (added). During retrieval, the superposition contains contributions from all colliding items, leading to interference. As `load_ratio = (N * K * H) / D` increases, collisions become more frequent.

**Key indicators:**
- `max_load` increases
- `collision_rate` increases
- `q2_estimate` increases (higher collision probability)

### 2. Superposition Noise
Even without collisions, superposition of many items creates noise. The signal-to-noise ratio (SNR) degrades as more items are stored. The `snr_proxy = 1.0 / sqrt(load_ratio)` metric approximates this effect.

**Key indicators:**
- `cosine_mean` decreases gradually even when `max_load` is low
- `snr_proxy` decreases

## Capacity Scaling: Why N* ≈ D/(K·H)

The theoretical capacity `N* = D / (K * H)` comes from the birthday paradox and load balancing theory:

1. **Hash writes per item**: Each item writes to `K * H` slots (K hashes, H independent hash functions)
2. **Total writes**: For N items, total writes = `N * K * H`
3. **Expected collisions**: When `N * K * H ≈ D`, we expect significant collisions
4. **Capacity transition**: At `capacity_units = N / (D/(K*H)) = 1.0`, the system transitions from low-collision to high-collision regime

**Empirical validation:**
- Experiments show degradation near `capacity_units ≈ 1.0`
- The vertical line at `capacity_units = 1.0` in plots marks the theoretical transition
- Actual degradation may occur slightly before or after 1.0 due to hash function quality and load distribution

## Measurement Artifacts vs. True Limits

### Fixed Artifacts

#### 1. CUDA Timing (Fixed)
- **Issue**: GPU operations are asynchronous; timing without synchronization measures CPU time, not GPU time
- **Fix**: Added `torch.cuda.synchronize()` before and after timing blocks
- **Impact**: Timing numbers are now accurate and stable across seeds

#### 2. Window Baseline W >= N (Fixed)
- **Issue**: When window size W >= N, baseline trivially achieves 1.0 success (all queries in window)
- **Fix**: Skip window baseline computation when `W >= N`
- **Impact**: Baselines are now fair and meaningful

#### 3. Cosine Threshold Sensitivity (Fixed)
- **Issue**: Hardcoded threshold 0.9 was too strict, causing many 0.0 success rates
- **Fix**: Made threshold configurable (default 0.7 for exp05, 0.9 for exp02)
- **Impact**: Success rates are more informative and show gradual degradation

### True Algorithmic Limits

#### 1. Collision Dominance (load_ratio > 1)
When `load_ratio > 1`, collisions are inevitable. This is a fundamental limit, not a measurement artifact.

**Evidence:**
- `max_load` grows linearly with `load_ratio`
- `q2_estimate` increases sharply
- `cosine_mean` drops below 0.5

#### 2. Hash Function Quality
The quality of the hash function affects load distribution:
- **Good hash**: Uniform distribution, degradation near `capacity_units = 1.0`
- **Poor hash**: Uneven distribution, early degradation

**Evidence:**
- `std_load` indicates distribution spread
- `gini_load` measures inequality (0 = uniform, 1 = maximum inequality)
- `top_slots` identifies hot spots

#### 3. K and H Trade-offs
- **Higher K**: More writes per item, higher collision probability, but better error correction
- **Higher H**: More independent hashes, better load balancing, but more writes

**Evidence:**
- exp02 shows accuracy vs K and H
- Optimal K depends on D and N
- H > 1 helps when K is large

## Reading the Plots

### Capacity-Normalized X-Axis
All main plots use `capacity_units = N / (D/(K·H))` on the x-axis:
- **x < 1.0**: Low-collision regime, high fidelity
- **x ≈ 1.0**: Transition point (vertical red line)
- **x > 1.0**: High-collision regime, degradation

### Error Bars
When `num_seeds >= 3`, plots show mean ± std:
- **Tight error bars**: Stable, reproducible results
- **Wide error bars**: High variance, may indicate hash function sensitivity

### Multiple Baselines
- **BBPM**: Main method (colored lines)
- **Window (W=X)**: Baseline that only remembers last X items (dashed lines)
- **Oracle**: Perfect hash-table upper bound (should be flat at 1.0)

### Degradation Patterns
1. **Gradual**: Cosine similarity decreases smoothly (superposition noise)
2. **Sharp**: Success rate drops abruptly (collision threshold)
3. **Plateau**: Performance stabilizes at low level (saturation)

## Failure Diagnostics

When `bbpm_success < 0.5` or `cosine_mean < 0.7`, failure diagnostics are automatically computed:

### Top Loaded Slots
- Identifies hot spots causing interference
- High loads in top slots indicate uneven distribution

### Query Hit Analysis
- `fraction_hit_load_gt_L`: Fraction of queries hitting slots with load > L
- High fractions indicate queries are hitting overloaded slots
- Correlates with degradation

### SNR Proxy
- `snr_proxy = 1.0 / sqrt(load_ratio)`
- Lower values indicate worse signal-to-noise ratio
- Correlates with `cosine_mean`

## Best Practices

1. **Always check capacity_units**: More informative than N/D alone
2. **Look for vertical line at 1.0**: Marks theoretical transition
3. **Compare to baselines**: Window and oracle provide context
4. **Check error bars**: Wide bars indicate instability
5. **Review failure diagnostics**: When degradation occurs, diagnostics explain why
6. **Consider K and H**: Optimal values depend on use case

## Common Patterns

### Pattern 1: Clean Degradation
- Smooth decrease in accuracy/cosine
- Degradation near `capacity_units = 1.0`
- Low variance (tight error bars)
- **Interpretation**: Well-behaved hash function, predictable capacity

### Pattern 2: Early Degradation
- Degradation before `capacity_units = 1.0`
- High `max_load` or `std_load`
- **Interpretation**: Poor hash function or uneven load distribution

### Pattern 3: Late Degradation
- Degradation after `capacity_units = 1.0`
- Low `collision_rate`
- **Interpretation**: Good hash function, effective load balancing

### Pattern 4: Sudden Drop
- Sharp transition from high to low accuracy
- High `q2_estimate` jump
- **Interpretation**: Collision threshold reached

## Questions to Ask

1. **Does degradation occur near capacity_units = 1.0?**
   - Yes → Matches theory, good hash function
   - No → Investigate hash function quality

2. **Are error bars tight?**
   - Yes → Stable, reproducible
   - No → High variance, may need more seeds

3. **Do baselines make sense?**
   - Window should degrade as N increases
   - Oracle should be flat at 1.0
   - If not, check implementation

4. **What does failure diagnostics show?**
   - High `max_load` → Collision problem
   - High `fraction_hit_load_gt_L` → Queries hitting overloaded slots
   - Low `snr_proxy` → Superposition noise
