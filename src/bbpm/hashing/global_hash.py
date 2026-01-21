"""Global affine hash function."""

from typing import Optional

import torch


class GlobalAffineHash:
    """
    Global affine hash function.

    Uses a simple salted affine formula: (key * A + offset * B + salt) % D
    This is a global hash that addresses the entire memory space uniformly.
    """

    def __init__(
        self,
        D: int,
        seed: int = 42,
        salt_base: int = 987654321,
        multiplier_A: int = 1315423911,
        multiplier_B: int = 2654435761,
    ):
        """
        Initialize global hash function.

        Args:
            D: Total number of memory slots
            seed: Seed for deterministic hashing (used to modify salt_base)
            salt_base: Base value for salt (used for multi-hash)
            multiplier_A: Multiplier for key term
            multiplier_B: Multiplier for offset term
        """
        self.D = D
        self.seed = seed
        self.salt_base = salt_base + seed * 123456789  # Incorporate seed into salt
        self.multiplier_A = multiplier_A
        self.multiplier_B = multiplier_B

    def indices(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor:
        """
        Compute memory indices using global affine hashing.

        Args:
            keys: Input keys of shape [B]
            K: Number of active slots per item per hash
            H: Number of independent hashes

        Returns:
            Indices tensor of shape [B, K*H] (int64)
        """
        keys = keys.long()
        B = keys.shape[0]
        device = keys.device

        indices_list = []

        for h in range(H):
            # Different salt for each hash function
            salt = h * self.salt_base + 123456789

            # Expand keys: [B] -> [B, 1]
            keys_expanded = keys.unsqueeze(-1)  # [B, 1]

            # K offsets: [1, K]
            k_offsets = torch.arange(K, device=device).view(1, -1)  # [1, K]

            # Hash formula: (key * A + offset * B + salt) % D
            idx = (
                keys_expanded * self.multiplier_A
                + k_offsets * self.multiplier_B
                + salt
            ) % self.D  # [B, K]

            indices_list.append(idx)

        # Concatenate all hashes: [B, K*H]
        return torch.cat(indices_list, dim=1)  # [B, K*H]
