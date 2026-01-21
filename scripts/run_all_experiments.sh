#!/bin/bash
# Run all experiments and generate figures

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running all BBPM experiments"
echo "=========================================="

# Create output directories
mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/figures"

# Optionally clean old results for fresh run
# Uncomment the next line to delete old results/figures before running
# rm -rf "$PROJECT_ROOT/results"/* "$PROJECT_ROOT/figures"/*

# Experiment 1: Capacity Scaling
echo ""
echo "Running exp01: Capacity Scaling..."
python "$PROJECT_ROOT/experiments/exp01_capacity_scaling/run.py" \
    --config "$PROJECT_ROOT/experiments/exp01_capacity_scaling/config.yaml" \
    --outdir "$PROJECT_ROOT/results/exp01_capacity_scaling" \
    --device auto
python "$PROJECT_ROOT/experiments/exp01_capacity_scaling/analyze.py" \
    --results "$PROJECT_ROOT/results/exp01_capacity_scaling/metrics.json" \
    --outdir "$PROJECT_ROOT/figures/exp01_capacity_scaling"

# Experiment 2: Ablation K and H
echo ""
echo "Running exp02: Ablation K and H..."
python "$PROJECT_ROOT/experiments/exp02_ablation_K_H/run.py" \
    --config "$PROJECT_ROOT/experiments/exp02_ablation_K_H/config.yaml" \
    --outdir "$PROJECT_ROOT/results/exp02_ablation_K_H" \
    --device auto
python "$PROJECT_ROOT/experiments/exp02_ablation_K_H/analyze.py" \
    --results "$PROJECT_ROOT/results/exp02_ablation_K_H/metrics.json" \
    --outdir "$PROJECT_ROOT/figures/exp02_ablation_K_H"

# Experiment 3: Block vs Global
echo ""
echo "Running exp03: Block vs Global..."
python "$PROJECT_ROOT/experiments/exp03_block_vs_global/run.py" \
    --config "$PROJECT_ROOT/experiments/exp03_block_vs_global/config.yaml" \
    --outdir "$PROJECT_ROOT/results/exp03_block_vs_global" \
    --device auto
python "$PROJECT_ROOT/experiments/exp03_block_vs_global/analyze.py" \
    --results "$PROJECT_ROOT/results/exp03_block_vs_global/metrics.json" \
    --outdir "$PROJECT_ROOT/figures/exp03_block_vs_global"

# Experiment 4: KV Memory Scaling
echo ""
echo "Running exp04: KV Memory Scaling..."
python "$PROJECT_ROOT/experiments/exp04_kv_memory_scaling/run.py" \
    --config "$PROJECT_ROOT/experiments/exp04_kv_memory_scaling/config.yaml" \
    --outdir "$PROJECT_ROOT/results/exp04_kv_memory_scaling" \
    --device auto
python "$PROJECT_ROOT/experiments/exp04_kv_memory_scaling/analyze.py" \
    --results "$PROJECT_ROOT/results/exp04_kv_memory_scaling/metrics.json" \
    --outdir "$PROJECT_ROOT/figures/exp04_kv_memory_scaling"

# Experiment 5: Needle in Haystack
echo ""
echo "Running exp05: Needle in Haystack..."
python "$PROJECT_ROOT/experiments/exp05_needle_haystack/run.py" \
    --config "$PROJECT_ROOT/experiments/exp05_needle_haystack/config.yaml" \
    --outdir "$PROJECT_ROOT/results/exp05_needle_haystack" \
    --device auto
python "$PROJECT_ROOT/experiments/exp05_needle_haystack/analyze.py" \
    --results "$PROJECT_ROOT/results/exp05_needle_haystack/metrics.json" \
    --outdir "$PROJECT_ROOT/figures/exp05_needle_haystack"

# Experiment 6: Drift Stability
echo ""
echo "Running exp06: Drift Stability..."
python "$PROJECT_ROOT/experiments/exp06_drift_stability/run.py" \
    --config "$PROJECT_ROOT/experiments/exp06_drift_stability/config.yaml" \
    --outdir "$PROJECT_ROOT/results/exp06_drift_stability" \
    --device auto
python "$PROJECT_ROOT/experiments/exp06_drift_stability/analyze.py" \
    --results "$PROJECT_ROOT/results/exp06_drift_stability/metrics.json" \
    --outdir "$PROJECT_ROOT/figures/exp06_drift_stability"

# Experiment 7: LLM Integration
echo ""
echo "Running exp07: LLM Integration..."
if python -c "import transformers" 2>/dev/null; then
    python "$PROJECT_ROOT/experiments/exp07_llm_integration/run_eval.py" \
        --config "$PROJECT_ROOT/experiments/exp07_llm_integration/config.yaml" \
        --outdir "$PROJECT_ROOT/results/exp07_llm_integration" \
        --device auto
    python "$PROJECT_ROOT/experiments/exp07_llm_integration/analyze.py" \
        --results "$PROJECT_ROOT/results/exp07_llm_integration/metrics.json" \
        --outdir "$PROJECT_ROOT/figures/exp07_llm_integration"
else
    echo "Note: transformers not available, skipping exp07"
    echo "Install with: pip install transformers"
    echo "Exp07 will run in synthetic mode (retrieval-only) if transformers is missing"
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved to: results/"
echo "Figures saved to: figures/"
