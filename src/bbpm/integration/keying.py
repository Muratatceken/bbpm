"""Key generation strategies for BBPM integration."""

from typing import Optional, Protocol

import torch


class KeyStrategy(Protocol):
    """Protocol for key generation strategies."""

    def get_keys(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Generate keys from inputs.

        Args:
            inputs: Input tensor (shape depends on strategy)

        Returns:
            Keys tensor of shape [B] or [B, T]
        """
        ...


class IDKeyStrategy:
    """Key strategy using token/position IDs directly."""

    def __init__(self, vocab_size: Optional[int] = None):
        """
        Initialize ID key strategy.

        Args:
            vocab_size: Optional vocabulary size for modulo operation
        """
        self.vocab_size = vocab_size

    def get_keys(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate keys from token IDs.

        Args:
            token_ids: Token ID tensor of shape [B, T] or [T]

        Returns:
            Keys tensor (flattened if needed)
        """
        keys = token_ids.long()
        if self.vocab_size is not None:
            keys = keys % self.vocab_size
        return keys.flatten() if keys.dim() > 1 else keys


class PositionKeyStrategy:
    """Key strategy using position indices."""

    def get_keys(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate keys from position indices.

        Args:
            positions: Position tensor of shape [B, T] or [T]

        Returns:
            Keys tensor (flattened if needed)
        """
        keys = positions.long()
        return keys.flatten() if keys.dim() > 1 else keys


class ProjectionKeyStrategy:
    """Key strategy using learned projection (stop-gradient)."""

    def __init__(self, input_dim: int, key_dim: int = 64):
        """
        Initialize projection key strategy.

        Args:
            input_dim: Input embedding dimension
            key_dim: Key dimension
        """
        import torch.nn as nn

        self.projection = nn.Linear(input_dim, key_dim)

    def get_keys(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate keys from embeddings via projection.

        Args:
            embeddings: Embedding tensor of shape [B, T, d] or [B, d]

        Returns:
            Keys tensor (hashed to integers)
        """
        # Project embeddings
        projected = self.projection(embeddings)  # [B, T, key_dim] or [B, key_dim]

        # Hash to integer keys (simple hash)
        if projected.dim() == 3:
            B, T, d = projected.shape
            # Flatten and hash
            flat = projected.view(B * T, d)
            # Simple hash: sum of quantized values
            quantized = (flat * 1000).long()
            keys = quantized.sum(dim=1) % (2 ** 31)
            return keys.view(B, T)
        else:
            quantized = (projected * 1000).long()
            keys = quantized.sum(dim=1) % (2 ** 31)
            return keys
