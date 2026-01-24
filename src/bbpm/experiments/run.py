"""Canonical experiment runner with --exp flag CLI."""

import argparse
from pathlib import Path

# Import experiment modules
from bbpm.experiments import (
    exp01_snr_scaling,
    exp02_k_h_ablation,
    exp03_runtime_vs_attention,
    exp04_needle,
    exp05_end2end_assoc,
    exp06_occupancy_skew,
    exp07_drift_reachability,
)

EXPERIMENTS = {
    "exp01": ("SNR Scaling", exp01_snr_scaling),
    "exp02": ("K/H Ablation", exp02_k_h_ablation),
    "exp03": ("Runtime vs Attention", exp03_runtime_vs_attention),
    "exp04": ("Needle-in-Haystack", exp04_needle),
    "exp05": ("End-to-End Associative Recall", exp05_end2end_assoc),
    "exp06": ("Occupancy Skew", exp06_occupancy_skew),
    "exp07": ("Drift and Reachability", exp07_drift_reachability),
}


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run BBPM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Common arguments
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="Experiment to run (exp01-exp07)",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("artifacts"),
        help="Output directory for metrics and figures"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of random seeds"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32",
        help="Data type"
    )
    
    # Parse known args first to get experiment ID
    args, unknown = parser.parse_known_args()
    
    # Get experiment module and add its specific arguments
    exp_id = args.exp
    _, exp_module = EXPERIMENTS[exp_id]
    
    # Create a new parser for experiment-specific args
    exp_parser = argparse.ArgumentParser()
    exp_module.add_args(exp_parser)
    
    # Parse experiment-specific args from unknown args
    exp_args, remaining = exp_parser.parse_known_args(unknown)
    
    if remaining:
        parser.error(f"Unrecognized arguments: {remaining}")
    
    # Merge common and experiment-specific args
    for key, value in vars(exp_args).items():
        setattr(args, key, value)
    
    # Run experiment
    print(f"Running {EXPERIMENTS[exp_id][0]} ({exp_id})...")
    result = exp_module.run(args)
    
    print(f"✓ {EXPERIMENTS[exp_id][0]} completed")
    print(f"  Metrics: {result.get('metrics_path', 'N/A')}")
    print(f"  Figure: {result.get('figure_path', 'N/A')}")


if __name__ == "__main__":
    main()
