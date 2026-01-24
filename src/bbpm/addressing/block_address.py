"""Block-based addressing logic.

This module provides deterministic block-based address computation using
Feistel PRP for within-block offsets and mix64 for block selection.
Guarantees unique offsets within each hash family.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bbpm.addressing.hash_mix import make_salts, mix64, u64
from bbpm.addressing.prp import FeistelPRP

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class AddressConfig:
    """Configuration for block-based addressing.

    Attributes:
        num_blocks: Number of blocks (B)
        block_size: Size of each block (L), must be power of 2
        K: Number of slots per item
        H: Number of independent hash families
        master_seed: Master seed (uint64) for salt generation
    """

    num_blocks: int  # B
    block_size: int  # L (must be power-of-two)
    K: int  # slots per item
    H: int  # number of independent hash families
    master_seed: int  # uint64

    def __post_init__(self) -> None:
        """Validate parameters."""
        # Enforce block_size is positive
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        # Enforce block_size is power-of-two
        if not (self.block_size & (self.block_size - 1) == 0):
            raise ValueError(
                f"block_size must be power of 2, got {self.block_size}"
            )

        # PRP domain constraint: block_size must be power-of-two (2^nbits)
        # This is required because FeistelPRP operates over domain [0, 2^nbits)
        # where nbits = log2(block_size). The PRP permutes offsets k in [0, K-1]
        # within each block, ensuring unique offsets per hash family.
        # Validation: block_size is already verified as power-of-two above.
        nbits_implied = self.block_size.bit_length() - 1
        if (1 << nbits_implied) != self.block_size:
            raise ValueError(
                f"block_size ({self.block_size}) must be exact power of 2 "
                f"for PRP domain [0, 2^nbits), got {self.block_size}"
            )

        # Enforce K <= block_size
        if self.K > self.block_size:
            raise ValueError(
                f"K ({self.K}) must be <= block_size ({self.block_size})"
            )

        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.H <= 0:
            raise ValueError("H must be positive")
        if not (0 <= self.master_seed < 2**64):
            raise ValueError("master_seed must be uint64")


class BlockAddress:
    """Block-based address computation.

    Computes deterministic memory addresses using block selection and
    within-block offsets via Feistel PRP. Guarantees unique offsets
    within each hash family.
    """

    def __init__(self, cfg: AddressConfig) -> None:
        """Initialize block addresser.

        Args:
            cfg: Address configuration
        """
        self.cfg = cfg

        # Pre-compute salts for all hash families
        self.salts = make_salts(cfg.H, cfg.master_seed)

        # Pre-compute FeistelPRP for block_size domain
        # nbits = log2(block_size)
        nbits = int(math.log2(cfg.block_size))

        # Use master_seed as PRP master_key, 6 rounds minimum
        self.prp = FeistelPRP(nbits=nbits, rounds=6, master_key=cfg.master_seed)

    def _block_id(self, hx: int, h: int) -> int:
        """Compute block ID for hash family h.

        block_id = mix64(hx ^ salt[h]) % num_blocks

        Args:
            hx: Hashed item key (uint64)
            h: Hash family index [0, H)

        Returns:
            Block ID in [0, num_blocks)
        """
        salt_h = self.salts[h]
        mixed = mix64(u64(hx) ^ u64(salt_h))
        return mixed % self.cfg.num_blocks

    def _seed_h(self, hx: int, h: int) -> int:
        """Derive per-hash seed for PRP.

        seed_h = mix64(hx ^ salt[h])

        Args:
            hx: Hashed item key (uint64)
            h: Hash family index [0, H)

        Returns:
            Seed for PRP (uint64)
        """
        salt_h = self.salts[h]
        return mix64(u64(hx) ^ u64(salt_h))

    def _offset_k(self, k: int, seed_h: int) -> int:
        """Compute within-block offset for slot k.

        offset_k = prp.permute(x=k, key=seed_h) mod block_size

        Args:
            k: Slot index [0, K)
            seed_h: Per-hash seed (uint64)

        Returns:
            Offset within block [0, block_size)
        """
        # PRP permute with k as input, seed_h as key
        permuted = self.prp.permute(k, seed_h)

        # Mask to block_size (should already be in range, but ensure)
        return permuted & (self.cfg.block_size - 1)

    def addresses(self, hx: int) -> list[int]:
        """Compute all addresses for item key hx.

        Returns a flat list of length H*K of global addresses in [0, B*L).
        Guarantees within a single (h, hx): offsets are unique for k=0..K-1.
        Deterministic across runs.

        Args:
            hx: Hashed item key (uint64)

        Returns:
            Flat list of H*K global addresses

        Example:
            >>> cfg = AddressConfig(num_blocks=10, block_size=256, K=32, H=2, master_seed=42)
            >>> addr = BlockAddress(cfg)
            >>> addrs = addr.addresses(12345)
            >>> len(addrs)
            64  # H*K = 2*32
        """
        # Validate input
        if not (0 <= hx < 2**64):
            raise ValueError("hx must be uint64")
        hx = u64(hx)

        addresses = []

        for h in range(self.cfg.H):
            # Compute block_id and seed_h for this hash family
            block_id = self._block_id(hx, h)
            seed_h = self._seed_h(hx, h)

            # Compute K offsets within this block
            for k in range(self.cfg.K):
                offset_k = self._offset_k(k, seed_h)

                # Global address = block_id * block_size + offset_k
                global_addr = block_id * self.cfg.block_size + offset_k
                addresses.append(global_addr)

        return addresses

    def addresses_tensor(self, hx: int, device: "torch.device") -> "torch.LongTensor":
        """Compute addresses as pre-allocated LongTensor [H*K].

        Returns tensor directly using vectorized PRP operations.
        No Python loops over k - uses permute_tensor for vectorized offset computation.

        Args:
            hx: Hashed item key (uint64)
            device: Target device for tensor

        Returns:
            LongTensor of shape [H*K] containing global addresses

        Example:
            >>> import torch
            >>> cfg = AddressConfig(num_blocks=10, block_size=256, K=32, H=2, master_seed=42)
            >>> addr = BlockAddress(cfg)
            >>> addrs = addr.addresses_tensor(12345, torch.device("cpu"))
            >>> addrs.shape
            torch.Size([64])  # H*K = 2*32
            >>> addrs.dtype
            torch.int64
        """
        import torch

        # Validate input
        if not (0 <= hx < 2**64):
            raise ValueError("hx must be uint64")
        hx = u64(hx)

        # Create k_vec once outside loop (reused for all hash families)
        k_vec = torch.arange(self.cfg.K, dtype=torch.long, device=device)

        # Collect address groups for each hash family
        addr_groups = []

        for h in range(self.cfg.H):  # Loop over h is acceptable
            # Compute block_id and seed_h for this hash family
            block_id = self._block_id(hx, h)
            seed_h = self._seed_h(hx, h)

            # Compute offsets using vectorized PRP (no Python loop over k)
            offsets = self.prp.permute_tensor(k_vec, seed_h)

            # Mask to block_size (should already be in range, but ensure)
            offsets = offsets & (self.cfg.block_size - 1)

            # Global addresses for this hash family: block_id * block_size + offsets
            addrs_h = block_id * self.cfg.block_size + offsets
            addr_groups.append(addrs_h)

        # Concatenate all groups to get [H*K] tensor
        # Order: for h in [0..H-1], for k in [0..K-1] (matches addresses() output)
        return torch.cat(addr_groups, dim=0)

    def addresses_grouped(self, hx: int) -> list[list[int]]:
        """Compute addresses grouped by hash family.

        Returns H lists each of length K.

        Args:
            hx: Hashed item key (uint64)

        Returns:
            List of H lists, each containing K addresses

        Example:
            >>> cfg = AddressConfig(num_blocks=10, block_size=256, K=32, H=2, master_seed=42)
            >>> addr = BlockAddress(cfg)
            >>> grouped = addr.addresses_grouped(12345)
            >>> len(grouped)
            2  # H
            >>> len(grouped[0])
            32  # K
        """
        # Validate input
        if not (0 <= hx < 2**64):
            raise ValueError("hx must be uint64")
        hx = u64(hx)

        grouped = []

        for h in range(self.cfg.H):
            # Compute block_id and seed_h for this hash family
            block_id = self._block_id(hx, h)
            seed_h = self._seed_h(hx, h)

            # Compute K offsets within this block
            addresses_h = []
            for k in range(self.cfg.K):
                offset_k = self._offset_k(k, seed_h)
                global_addr = block_id * self.cfg.block_size + offset_k
                addresses_h.append(global_addr)

            grouped.append(addresses_h)

        return grouped

    def block_ids(self, hx: int) -> list[int]:
        """Compute block IDs for all hash families.

        Returns H block ids.

        Args:
            hx: Hashed item key (uint64)

        Returns:
            List of H block IDs

        Example:
            >>> cfg = AddressConfig(num_blocks=10, block_size=256, K=32, H=2, master_seed=42)
            >>> addr = BlockAddress(cfg)
            >>> block_ids = addr.block_ids(12345)
            >>> len(block_ids)
            2  # H
        """
        # Validate input
        if not (0 <= hx < 2**64):
            raise ValueError("hx must be uint64")
        hx = u64(hx)

        block_ids_list = []
        for h in range(self.cfg.H):
            block_id = self._block_id(hx, h)
            block_ids_list.append(block_id)

        return block_ids_list
