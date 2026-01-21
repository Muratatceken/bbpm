"""Baseline PyTorch operations for memory access."""

import torch


def index_add(
    memory: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Add values to memory at given indices (baseline implementation).

    Args:
        memory: Memory tensor
        indices: Indices tensor
        values: Values to add
        dim: Dimension along which to add

    Returns:
        Updated memory tensor
    """
    return memory.index_add_(dim, indices, values)


def index_gather(
    memory: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Gather values from memory at given indices (baseline implementation).

    Args:
        memory: Memory tensor
        indices: Indices tensor
        dim: Dimension along which to gather

    Returns:
        Gathered values tensor
    """
    return torch.index_select(memory, dim, indices.flatten())
