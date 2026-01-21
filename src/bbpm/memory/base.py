"""Base memory interface."""

from typing import Protocol

import torch


class BaseMemory(Protocol):
    """
    Protocol for BBPM memory implementations.

    Defines the core write/read API and scaling conventions.
    """

    def clear(self) -> None:
        """Clear all memory contents."""
        ...

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Write key-value pairs to memory.

        Args:
            keys: Key tensor of shape [B]
            values: Value tensor of shape [B, d]
        """
        ...

    def read(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Read values from memory using keys.

        Args:
            keys: Key tensor of shape [B]

        Returns:
            Retrieved values of shape [B, d]
        """
        ...
