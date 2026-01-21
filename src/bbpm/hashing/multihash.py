"""Multi-hash wrapper for collision diagnostics."""

from typing import Dict, Optional

import torch

from .base import HashFunction
from .diagnostics import collision_rate, max_load, occupancy_summary


class MultiHashWrapper:
    """
    Wraps a base hash function to provide multi-hash behavior with diagnostics.

    Uses distinct salts to approximate independence between hash functions.
    Provides collision diagnostics and occupancy statistics.
    """

    def __init__(self, base_hash: HashFunction):
        """
        Initialize multi-hash wrapper.

        Args:
            base_hash: Base hash function to wrap
        """
        self.base_hash = base_hash

    def indices(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor:
        """
        Compute indices using multi-hash (delegates to base hash).

        Args:
            keys: Input keys of shape [B]
            K: Number of active slots per item per hash
            H: Number of independent hashes

        Returns:
            Indices tensor of shape [B, K*H] (int64)
        """
        return self.base_hash.indices(keys, K, H)

    def diagnostics(self, keys: torch.Tensor, K: int, H: int, D: int) -> Dict[str, float]:
        """
        Compute collision and occupancy diagnostics.

        Args:
            keys: Input keys of shape [B]
            K: Number of active slots per item per hash
            H: Number of independent hashes
            D: Total memory size

        Returns:
            Dictionary with diagnostic metrics:
            - collision_rate: Fraction of duplicate indices within batch
            - max_load: Maximum number of items hashing to any single slot
            - occupancy_summary: Compact summary of slot occupancies
        """
        indices = self.indices(keys, K, H)  # [B, K*H]

        # Flatten to get all indices
        all_indices = indices.flatten()  # [B*K*H]

        # Collision rate (within-batch duplicates)
        coll_rate = collision_rate(indices)

        # Max load
        max_ld = max_load(all_indices, D)

        # Occupancy summary (compact, artifact-safe)
        occupancy = occupancy_summary(all_indices, D)

        return {
            "collision_rate": coll_rate,
            "max_load": max_ld,
            "occupancy_summary": occupancy,
        }
