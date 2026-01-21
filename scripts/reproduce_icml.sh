#!/bin/bash
# Reproduce ICML figures with fixed seeds

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Reproducing ICML figures (fixed seeds)"
echo "=========================================="

# Use fixed seeds for reproducibility
export BBPM_SEED=42

# Create output directories
mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/figures"

# Run all experiments with fixed seeds
# (Experiments should use seed from config, but we ensure it's set)
bash "$PROJECT_ROOT/scripts/run_all_experiments.sh"

echo ""
echo "=========================================="
echo "ICML reproduction complete!"
echo "All figures generated with fixed seed=42"
echo "=========================================="
