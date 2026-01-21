"""Binary/uint8 variant of BBPM (Bloom-filter like)."""

from typing import Optional

import torch
import torch.nn as nn

from ..hashing.base import HashFunction
from ..hashing.global_hash import GlobalAffineHash
from .base import BaseMemory


class BinaryBBPMBloom(nn.Module, BaseMemory):
    """
    1-bit BBPM variant (Bloom-filter limit case).

    Uses uint8 memory (true 8 bits per byte).
    Independent hash functions for K bits per item per hash.
    """

    def __init__(
        self,
        D: int,
        K: int = 50,
        H: int = 3,
        device: str = "cpu",
        hash_fn: Optional[HashFunction] = None,
        seed: int = 42,
    ):
        """
        Initialize binary BBPM.

        Args:
            D: Total number of memory slots (bits)
            K: Number of bits per item per hash
            H: Number of independent hash functions
            device: Device to use
            hash_fn: Hash function to use (if None, creates GlobalAffineHash)
            seed: Seed for hash function initialization (if hash_fn is None)
        """
        super().__init__()

        self.D = D
        self.K = K
        self.H = H
        self.device = device

        # Store hash function (inject or create with deterministic seed)
        if hash_fn is None:
            self.hash_fn = GlobalAffineHash(D, seed=seed)
        else:
            self.hash_fn = hash_fn

        # Binary memory (uint8 for efficiency)
        self.register_buffer(
            "memory",
            torch.zeros(D, dtype=torch.uint8, device=device),
        )

    def clear(self) -> None:
        """Clear all memory contents."""
        self.memory.zero_()

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Write keys to binary memory (sets bits to 1).

        Args:
            keys: Key tensor of shape [B]
            values: Not used in binary variant (for API compatibility)
        """
        indices = self.hash_fn.indices(keys, self.K, self.H)  # [B, K*H]
        indices_flat = indices.flatten()

        # Set bits to 1
        self.memory[indices_flat] = 1

    def read(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Read from binary memory (returns fraction of bits set).

        Args:
            keys: Key tensor of shape [B]

        Returns:
            Score tensor of shape [B] (fraction of bits that are 1)
        """
        indices = self.hash_fn.indices(keys, self.K, self.H)  # [B, K*H]

        B = keys.shape[0]
        indices_flat = indices.flatten()

        # Read bits
        bits = self.memory[indices_flat].float()  # [B*K*H]

        # Reshape and compute mean per key
        bits_reshaped = bits.view(B, self.K * self.H)  # [B, K*H]
        scores = bits_reshaped.mean(dim=1)  # [B]

        return scores
