"""Base hash function interface."""

from typing import Protocol

import torch


class HashFunction(Protocol):
    """
    Protocol for hash functions used in BBPM.

    Hash functions map keys to memory indices. They should be deterministic
    and support multi-hash (H independent hash functions).
    """

    def indices(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor:
        """
        Compute memory indices for given keys.

        Args:
            keys: Input keys of shape [B] (batch of keys)
            K: Number of active slots per item per hash
            H: Number of independent hashes (multi-hash)

        Returns:
            Indices tensor of shape [B, K*H] (int64)
            Each row contains K*H indices that should be used for that key
        """
        ...
