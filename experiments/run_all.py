"""Main experiment runner for all BBPM experiments."""

import sys
from pathlib import Path

from experiments.exp01_snr_scaling import run_exp01_snr_scaling
from experiments.exp02_k_h_ablation import run_exp02_k_h_ablation
from experiments.exp03_runtime_vs_attention import run_exp03_runtime_vs_attention
from experiments.exp04_needle import run_exp04_needle
from experiments.exp05_end2end_assoc import run_exp05_end2end_assoc
from experiments.exp06_occupancy_skew import run_exp06_occupancy_skew
from experiments.exp07_drift_reachability import run_exp07_drift_reachability


def main() -> None:
    """Run all experiments."""
    # Ensure output directories exist
    artifacts_dir = Path("artifacts")
    figures_dir = artifacts_dir / "figures"
    metrics_dir = artifacts_dir / "metrics"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        ("Exp 01: SNR Scaling", run_exp01_snr_scaling),
        ("Exp 02: K and H Ablation", run_exp02_k_h_ablation),
        ("Exp 03: Runtime vs Attention", run_exp03_runtime_vs_attention),
        ("Exp 04: Needle-in-Haystack", run_exp04_needle),
        ("Exp 05: End-to-End Associative Recall", run_exp05_end2end_assoc),
        ("Exp 06: Occupancy Skew", run_exp06_occupancy_skew),
        ("Exp 07: Drift and Reachability", run_exp07_drift_reachability),
    ]
    
    print("Running all BBPM experiments...")
    for name, experiment_fn in experiments:
        print(f"\nRunning {name}...")
        try:
            experiment_fn()
            print(f"✓ {name} completed")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            sys.exit(1)
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
