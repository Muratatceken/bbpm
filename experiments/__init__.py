"""Experiments module for BBPM."""

# Add src to path for bbpm imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Export experiment modules for CLI
import exp01_snr_scaling
import exp02_k_h_ablation
import exp03_runtime_vs_attention
import exp04_needle
import exp05_end2end_assoc
import exp06_occupancy_skew
import exp07_drift_reachability

__all__ = [
    "exp01_snr_scaling",
    "exp02_k_h_ablation",
    "exp03_runtime_vs_attention",
    "exp04_needle",
    "exp05_end2end_assoc",
    "exp06_occupancy_skew",
    "exp07_drift_reachability",
]
