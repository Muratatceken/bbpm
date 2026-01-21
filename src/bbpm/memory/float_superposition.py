"""Canonical BBPM float superposition memory with counts."""

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

from ..hashing.base import HashFunction
from ..hashing.global_hash import GlobalAffineHash


class BBPMMemoryFloat(nn.Module):
    """
    Canonical BBPM memory: Float superposition with counts.

    This is the main BBPM implementation as specified in the paper.
    Uses K-sparse addressing with H independent hash functions.
    Maintains counts for debiasing during read operations.
    """

    def __init__(
        self,
        D: int,
        d: int,
        K: int,
        H: int = 1,
        hash_fn: Optional[HashFunction] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        write_scale: Literal["unit", "1/sqrt(KH)"] = "1/sqrt(KH)",
        seed: int = 42,
    ):
        """
        Initialize BBPM memory.

        Args:
            D: Total number of memory slots
            d: Value dimension
            K: Active slots per item per hash
            H: Number of independent hashes (multi-hash)
            hash_fn: Hash function to use (default: GlobalAffineHash)
            dtype: Data type for memory storage
            device: Device to use ("cpu" or "cuda")
            write_scale: Scaling for write operations
                - "unit": No scaling
                - "1/sqrt(KH)": Normalize by sqrt(K*H) for signal strength
            seed: Seed for deterministic hashing (used when hash_fn is None)
        """
        super().__init__()

        self.D = D
        self.d = d
        self.K = K
        self.H = H
        self.device = device
        self.write_scale = write_scale
        self.seed = seed

        # Hash function
        if hash_fn is None:
            self.hash_fn: HashFunction = GlobalAffineHash(D, seed=seed)
        else:
            self.hash_fn = hash_fn

        # Memory buffers
        self.register_buffer(
            "memory",
            torch.zeros(D, d, dtype=dtype, device=device),
        )
        self.register_buffer(
            "counts",
            torch.zeros(D, 1, dtype=dtype, device=device),
        )

        # Write scaling factor
        if write_scale == "1/sqrt(KH)":
            self.scale = 1.0 / (K * H) ** 0.5
        else:
            self.scale = 1.0

    def clear(self) -> None:
        """Clear all memory contents."""
        self.memory.zero_()
        self.counts.zero_()

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Write key-value pairs to memory.

        Uses index_add_ to add values into memory slots.
        Increments counts for each slot used.

        Args:
            keys: Key tensor of shape [B]
            values: Value tensor of shape [B, d]
        """
        keys = keys.to(self.device)
        values = values.to(self.device)

        # Get indices: [B, K*H]
        indices = self.hash_fn.indices(keys, self.K, self.H)  # [B, K*H]

        # Flatten indices for index_add_
        indices_flat = indices.flatten()  # [B*K*H]

        # Expand values for K*H writes per item
        B = keys.shape[0]
        values_expanded = values.unsqueeze(1).expand(B, self.K * self.H, self.d)  # [B, K*H, d]
        values_flat = values_expanded.reshape(-1, self.d)  # [B*K*H, d]

        # Apply write scaling
        values_flat = values_flat * self.scale

        # Add to memory using index_add_
        self.memory.index_add_(0, indices_flat, values_flat)

        # Increment counts
        ones = torch.ones(len(indices_flat), 1, dtype=self.counts.dtype, device=self.device)
        self.counts.index_add_(0, indices_flat, ones)

    def read(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Read values from memory using keys.

        Retrieves values from K*H slots, applies per-slot debiasing (memory/(counts+eps)),
        then pools across slots with mean.

        Args:
            keys: Key tensor of shape [B]

        Returns:
            Retrieved values of shape [B, d]
        """
        keys = keys.to(self.device)

        # Get indices: [B, K*H]
        indices = self.hash_fn.indices(keys, self.K, self.H)  # [B, K*H]

        B = keys.shape[0]

        # Gather memory and counts
        gathered_memory = self.memory[indices]  # [B, K*H, d]
        gathered_counts = self.counts[indices]  # [B, K*H, 1]

        # Per-slot debiasing: memory / (counts + eps)
        eps = 1e-8
        debiased = gathered_memory / (gathered_counts + eps)  # [B, K*H, d]

        # Pool across K*H slots with mean
        result = debiased.mean(dim=1)  # [B, d]

        return result

    def diagnostics(self, keys: torch.Tensor) -> Dict[str, float]:
        """
        Compute diagnostics for given keys.

        Args:
            keys: Key tensor of shape [B]

        Returns:
            Dictionary with diagnostic metrics
        """
        from ..hashing.diagnostics import (
            collision_rate,
            estimate_q2,
            max_load,
            occupancy_summary,
        )

        indices = self.hash_fn.indices(keys, self.K, self.H)  # [B, K*H]
        indices_flat = indices.flatten()

        # Use compact occupancy_summary instead of full histogram
        occupancy = occupancy_summary(indices_flat, self.D)

        return {
            "collision_rate": collision_rate(indices),
            "max_load": max_load(indices_flat, self.D),
            "q2_estimate": estimate_q2(indices_flat, self.D),
            "occupancy_summary": occupancy,
        }

    def memory_diagnostics(self) -> Dict[str, float]:
        """
        Compute lightweight diagnostics from memory counts.

        Optional helper that computes diagnostics from self.counts without
        requiring keys. Useful for post-write analysis.

        Returns:
            Dictionary with:
            - nonzero_slots: int (count of slots with count > 0)
            - max_count: int (maximum count value)
            - mean_count_nonzero: float (average count for nonzero slots)
            - q2_estimate_from_counts: float (computed from counts)
        """
        counts_flat = self.counts.flatten()  # [D]
        nonzero_mask = counts_flat > 0
        nonzero_counts = counts_flat[nonzero_mask]

        if len(nonzero_counts) == 0:
            return {
                "nonzero_slots": 0,
                "max_count": 0,
                "mean_count_nonzero": 0.0,
                "q2_estimate_from_counts": 0.0,
            }

        total_writes = counts_flat.sum().item()
        nonzero_slots = int(nonzero_mask.sum().item())
        max_count = int(counts_flat.max().item())
        mean_count_nonzero = float(nonzero_counts.float().mean().item())

        # Q2 estimate from counts: sum((count_i / total_writes)^2) for nonzero
        if total_writes > 0:
            probs = nonzero_counts.float() / total_writes
            q2_estimate_from_counts = float((probs ** 2).sum().item())
        else:
            q2_estimate_from_counts = 0.0

        return {
            "nonzero_slots": nonzero_slots,
            "max_count": max_count,
            "mean_count_nonzero": mean_count_nonzero,
            "q2_estimate_from_counts": q2_estimate_from_counts,
        }
