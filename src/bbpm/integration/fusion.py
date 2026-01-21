"""Fusion modules for combining BBPM retrieval with model outputs."""

from typing import Protocol

import torch
import torch.nn as nn


class FusionModule(Protocol):
    """Protocol for fusion modules."""

    def forward(
        self,
        local_output: torch.Tensor,
        memory_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse local and memory outputs.

        Args:
            local_output: Local model output [B, T, d] or [B, d]
            memory_output: Memory retrieval output [B, T, d] or [B, d]

        Returns:
            Fused output [B, T, d] or [B, d]
        """
        ...


class GatingFusion(nn.Module):
    """Gating-based fusion (learned gating between local and memory)."""

    def __init__(self, dim: int):
        """
        Initialize gating fusion.

        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.activation = nn.Sigmoid()

    def forward(
        self,
        local_output: torch.Tensor,
        memory_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse using learned gating.

        Args:
            local_output: Local output [B, T, d] or [B, d]
            memory_output: Memory output [B, T, d] or [B, d]

        Returns:
            Fused output
        """
        # Concatenate
        combined = torch.cat([local_output, memory_output], dim=-1)  # [B, T, 2d] or [B, 2d]

        # Compute gate
        gate_values = self.activation(self.gate(combined))  # [B, T, d] or [B, d]

        # Gated combination
        fused = gate_values * local_output + (1 - gate_values) * memory_output

        return fused


class ResidualFusion(nn.Module):
    """Residual fusion (simple addition with optional projection)."""

    def __init__(self, dim: int, use_projection: bool = False):
        """
        Initialize residual fusion.

        Args:
            dim: Feature dimension
            use_projection: Whether to use projection before addition
        """
        super().__init__()
        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        local_output: torch.Tensor,
        memory_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse using residual connection.

        Args:
            local_output: Local output [B, T, d] or [B, d]
            memory_output: Memory output [B, T, d] or [B, d]

        Returns:
            Fused output
        """
        if self.use_projection:
            memory_output = self.proj(memory_output)

        return local_output + memory_output
