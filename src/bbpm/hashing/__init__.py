"""Hashing modules for BBPM."""

from .base import HashFunction
from .global_hash import GlobalAffineHash
from .block_hash import BlockHash
from .multihash import MultiHashWrapper
from .diagnostics import (
    collision_rate,
    estimate_q2,
    gini_load,
    gini_of_load,
    max_load,
    occupancy_hist,  # Deprecated, kept for compatibility
    occupancy_summary,
    self_collision_prob,
    slot_loads,
)

__all__ = [
    "HashFunction",
    "GlobalAffineHash",
    "BlockHash",
    "MultiHashWrapper",
    "occupancy_summary",  # Use this instead of occupancy_hist
    "slot_loads",  # For microbenchmarks only
    "max_load",
    "gini_load",
    "gini_of_load",
    "collision_rate",
    "estimate_q2",
    "self_collision_prob",
    "occupancy_hist",  # Deprecated
]
