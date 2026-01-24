"""Experiments module for BBPM."""

# Export experiment modules for CLI
# Use relative imports to avoid circular dependency
from . import exp01_snr_scaling
from . import exp02_k_h_ablation
from . import exp03_runtime_vs_attention
from . import exp04_needle
from . import exp05_end2end_assoc
from . import exp06_occupancy_skew
from . import exp07_drift_reachability

__all__ = [
    "exp01_snr_scaling",
    "exp02_k_h_ablation",
    "exp03_runtime_vs_attention",
    "exp04_needle",
    "exp05_end2end_assoc",
    "exp06_occupancy_skew",
    "exp07_drift_reachability",
]
