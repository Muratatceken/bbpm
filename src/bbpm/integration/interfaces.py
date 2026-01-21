"""Interfaces for integrating BBPM into neural models."""

from typing import Protocol

import torch


class BBPMInterface(Protocol):
    """
    Interface for BBPM integration into models.

    Defines how BBPM plugs into neural network architectures.
    """

    def write_context(self, context_keys: torch.Tensor, context_values: torch.Tensor) -> None:
        """
        Write context information to memory.

        Args:
            context_keys: Context keys of shape [B, T] or [T]
            context_values: Context values of shape [B, T, d] or [T, d]
        """
        ...

    def retrieve(self, query_keys: torch.Tensor) -> torch.Tensor:
        """
        Retrieve information from memory.

        Args:
            query_keys: Query keys of shape [B, T] or [T]

        Returns:
            Retrieved values of shape [B, T, d] or [T, d]
        """
        ...
