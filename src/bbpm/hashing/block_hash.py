"""Block-based hash function."""

from typing import Optional

import torch


class BlockHash:
    """
    Block-based hash function.

    Memory is divided into blocks. For each key:
    1. Deterministically select a block
    2. Use offset hashing within that block (not guaranteed bijection)

    This approach helps minimize self-collisions and provides better load distribution.
    Note: This does not implement a true permutation (bijection); it uses hash-based
    offset mapping which may have collisions within a block.
    """

    def __init__(
        self,
        D: int,
        block_size: int,
        seed: int = 42,
        salt_base: int = 11400714819323198485,
        block_multiplier: int = 6364136223846793005,
        inner_multiplier_A: int = 22695477,
        inner_multiplier_B: int = 1103515245,
    ):
        """
        Initialize block-based hash.

        Args:
            D: Total number of memory slots
            block_size: Size of each block (D must be divisible by block_size)
            seed: Seed for deterministic hashing (used to modify salt_base)
            salt_base: Base value for salt (used for multi-hash)
            block_multiplier: Multiplier for block selection
            inner_multiplier_A: Multiplier for inner block addressing (key term)
            inner_multiplier_B: Multiplier for inner block addressing (offset term)
        """
        if D % block_size != 0:
            raise ValueError(f"D ({D}) must be divisible by block_size ({block_size})")

        self.D = D
        self.block_size = block_size
        self.num_blocks = D // block_size
        self.seed = seed
        self.salt_base = salt_base + seed * 123456789  # Incorporate seed into salt
        self.block_multiplier = block_multiplier
        self.inner_multiplier_A = inner_multiplier_A
        self.inner_multiplier_B = inner_multiplier_B

    def indices(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor:
        """
        Compute memory indices using block-based hashing.

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
            salt = (h + 1) * self.salt_base

            # Step 1: Select block deterministically
            # block_id: [B]
            block_id = (keys * self.block_multiplier + salt) % self.num_blocks
            block_start = block_id * self.block_size  # [B]

            # Step 2: Generate offsets within block (hash-based, not true permutation)
            # Use deterministic hash-based offset mapping
            # For each key, we generate K offsets within the block
            keys_expanded = keys.unsqueeze(-1)  # [B, 1]
            k_offsets = torch.arange(K, device=device).view(1, -1)  # [1, K]

            # Inner block addressing: hash-based offset mapping (not guaranteed bijection)
            inner = (
                keys_expanded * self.inner_multiplier_A
                + k_offsets * self.inner_multiplier_B
                + salt
            ) % self.block_size  # [B, K]

            # Step 3: Combine block start and inner offset
            global_idx = block_start.unsqueeze(-1) + inner  # [B, K]

            indices_list.append(global_idx)

        # Concatenate all hashes: [B, K*H]
        return torch.cat(indices_list, dim=1)  # [B, K*H]
