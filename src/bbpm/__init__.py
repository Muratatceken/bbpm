"""BBPM: Block-Based Permutation Memory library."""

from .config import load_config
from .addressing import BBPMAddressing
from .hashing import (
    GlobalAffineHash,
    HashFunction,
    MultiHashWrapper,
    collision_rate,
    compute_capacity_metrics,
    estimate_q2,
    gini_load,
    gini_of_load,
    max_load,
    occupancy_summary,
    query_hit_analysis,
    self_collision_prob,
)
from .memory import BBPMMemoryFloat, BaseMemory
from .memory.binary_bloom import BinaryBBPMBloom
from .utils import Timer, get_device, get_logger, set_device, set_global_seed

__version__ = "0.1.0"

__all__ = [
    # Core memory
    "BBPMMemoryFloat",
    "BaseMemory",
    "BinaryBBPMBloom",
    # Addressing (theory-compatible)
    "BBPMAddressing",
    # Hashing
    "HashFunction",
    "GlobalAffineHash",
    "MultiHashWrapper",
    # Diagnostics
    "occupancy_summary",
    "max_load",
    "gini_load",
    "gini_of_load",
    "collision_rate",
    "estimate_q2",
    "self_collision_prob",
    "compute_capacity_metrics",
    "query_hit_analysis",
    # Utils
    "set_global_seed",
    "get_logger",
    "Timer",
    "get_device",
    "set_device",
    "load_config",
]
