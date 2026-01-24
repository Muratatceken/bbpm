"""Canonical experiment runner with subparser-based CLI."""

import argparse
import sys
from pathlib import Path

# Add src to path for bbpm imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import experiment modules
import exp01_snr_scaling
import exp02_k_h_ablation
import exp03_runtime_vs_attention
import exp04_needle
import exp05_end2end_assoc
import exp06_occupancy_skew
import exp07_drift_reachability

EXPERIMENTS = {
    "exp01": ("SNR Scaling", exp01_snr_scaling),
    "exp02": ("K/H Ablation", exp02_k_h_ablation),
    "exp03": ("Runtime vs Attention", exp03_runtime_vs_attention),
    "exp04": ("Needle-in-Haystack", exp04_needle),
    "exp05": ("End-to-End Associative Recall", exp05_end2end_assoc),
    "exp06": ("Occupancy Skew", exp06_occupancy_skew),
    "exp07": ("Drift and Reachability", exp07_drift_reachability),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser.
    
    Args:
        parser: Argument parser to add common args to
    """
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


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run BBPM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="exp", help="Experiment to run", required=True)
    
    # Create subparser for each experiment
    for exp_id, (exp_name, exp_module) in EXPERIMENTS.items():
        exp_parser = subparsers.add_parser(exp_id, help=exp_name)
        add_common_args(exp_parser)
        exp_module.add_args(exp_parser)
    
    args = parser.parse_args()
    
    # Get experiment module
    exp_id = args.exp
    _, exp_module = EXPERIMENTS[exp_id]
    
    # Run experiment
    print(f"Running {EXPERIMENTS[exp_id][0]} ({exp_id})...")
    result = exp_module.run(args)
    
    print(f"✓ {EXPERIMENTS[exp_id][0]} completed")
    print(f"  Metrics: {result.get('metrics_path', 'N/A')}")
    print(f"  Figure: {result.get('figure_path', 'N/A')}")


if __name__ == "__main__":
    main()
