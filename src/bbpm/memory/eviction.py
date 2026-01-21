"""Optional eviction policies for BBPM memory."""

from typing import Literal, Optional

import torch
import torch.nn as nn


class DecayEviction:
    """Decay-based eviction policy."""

    def __init__(self, decay_rate: float = 0.99):
        """
        Initialize decay eviction.

        Args:
            decay_rate: Multiplicative decay factor per step (0 < decay_rate <= 1)
        """
        self.decay_rate = decay_rate

    def apply(self, memory: torch.Tensor, counts: torch.Tensor) -> None:
        """
        Apply decay to memory and counts.

        Args:
            memory: Memory tensor [D, d]
            counts: Counts tensor [D, 1]
        """
        memory.mul_(self.decay_rate)
        counts.mul_(self.decay_rate)


class ClippingEviction:
    """Clipping-based eviction policy (clips counts to maximum)."""

    def __init__(self, max_count: float = 100.0):
        """
        Initialize clipping eviction.

        Args:
            max_count: Maximum count value before clipping
        """
        self.max_count = max_count

    def apply(self, memory: torch.Tensor, counts: torch.Tensor) -> None:
        """
        Apply clipping to counts and adjust memory accordingly.

        Args:
            memory: Memory tensor [D, d]
            counts: Counts tensor [D, 1]
        """
        # Clip counts
        counts.clamp_(max=self.max_count)
        # Adjust memory to maintain consistency
        # This is a simplified version - full implementation would track overflow


class TTLEviction:
    """Time-to-live (TTL) eviction policy."""

    def __init__(self, ttl: int = 1000):
        """
        Initialize TTL eviction.

        Args:
            ttl: Time-to-live in steps
        """
        self.ttl = ttl
        self.step = 0

    def apply(self, memory: torch.Tensor, counts: torch.Tensor) -> None:
        """
        Apply TTL eviction (resets old entries).

        Args:
            memory: Memory tensor [D, d]
            counts: Counts tensor [D, 1]
        """
        self.step += 1
        # Simplified: reset entries older than TTL
        # Full implementation would track write timestamps
        if self.step % self.ttl == 0:
            # Reset low-count entries (heuristic for old entries)
            mask = counts.squeeze() < 0.1
            memory[mask] = 0
            counts[mask] = 0
