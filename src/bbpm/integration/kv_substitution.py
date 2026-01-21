"""KV cache substitution with BBPM augmentation."""

from typing import Optional

import torch
import torch.nn as nn

from ..memory.float_superposition import BBPMMemoryFloat


class LimitedKVWithBBPM(nn.Module):
    """
    Limited KV cache augmented with BBPM.

    Maintains a small KV cache for recent context and uses BBPM for long-term storage.
    """

    def __init__(
        self,
        bbpm_memory: BBPMMemoryFloat,
        kv_cache_size: int = 128,
        dim: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize limited KV with BBPM.

        Args:
            bbpm_memory: BBPM memory instance
            kv_cache_size: Size of local KV cache (recent context)
            dim: Feature dimension
            device: Device to use
        """
        super().__init__()

        self.bbpm = bbpm_memory
        self.kv_cache_size = kv_cache_size
        self.dim = dim
        self.device = device

        # Local KV cache (circular buffer)
        self.register_buffer(
            "k_cache",
            torch.zeros(kv_cache_size, dim, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(kv_cache_size, dim, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "cache_positions",
            torch.zeros(kv_cache_size, dtype=torch.long, device=device),
        )

        self.cache_ptr = 0

    def write_to_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """
        Write to both local KV cache and BBPM.

        Args:
            keys: Key tensor [B, T] or [T]
            values: Value tensor [B, T, d] or [T, d]
            positions: Position tensor [B, T] or [T]
        """
        # Flatten if needed
        if keys.dim() > 1:
            keys = keys.flatten()
            values = values.view(-1, self.dim)
            positions = positions.flatten()

        # Write to BBPM (long-term)
        self.bbpm.write(keys, values)

        # Write to local KV cache (recent context, circular buffer)
        n = len(keys)
        for i in range(n):
            idx = (self.cache_ptr + i) % self.kv_cache_size
            self.k_cache[idx] = values[i]
            self.v_cache[idx] = values[i]  # Simplified: same for K and V
            self.cache_positions[idx] = positions[i]

        self.cache_ptr = (self.cache_ptr + n) % self.kv_cache_size

    def retrieve(
        self,
        query_keys: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retrieve from both caches (prefer local, fallback to BBPM).

        Args:
            query_keys: Query keys [B, T] or [T]
            query_positions: Optional position queries [B, T] or [T]

        Returns:
            Retrieved values [B, T, d] or [T, d]
        """
        # Flatten if needed
        if query_keys.dim() > 1:
            query_keys = query_keys.flatten()
            if query_positions is not None:
                query_positions = query_positions.flatten()

        # Try local cache first
        results = []
        for i, key in enumerate(query_keys):
            # Check if in local cache
            if query_positions is not None:
                pos = query_positions[i]
                mask = self.cache_positions == pos
                if mask.any():
                    idx = mask.nonzero()[0, 0].item()
                    results.append(self.v_cache[idx])
                    continue

            # Fallback to BBPM
            retrieved = self.bbpm.read(key.unsqueeze(0))
            results.append(retrieved.squeeze(0))

        return torch.stack(results)
