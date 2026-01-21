"""BBPM addressing: block-based permutation memory addressing.

Implements two-stage addressing:
1. Block selection via deterministic hash
2. Intra-block permutation via Feistel PRP
"""

import math

import torch

from ..hashing.base import HashFunction

from .block_selector import select_block
from .prp_feistel import prp_offsets


class BBPMAddressing:
    """BBPM addressing function using block selection + intra-block PRP.
    
    Memory is divided into B blocks, each of size L, total D = B*L slots.
    For each key:
    1. Block selection: b_x = H(h_x) mod B
    2. Intra-block permutation: offset_k = P(h_x, k) mod L (via PRP)
    3. Final: addr_k = b_x * L + offset_k
    
    This ensures no self-collisions within a block (PRP guarantees bijection).
    """

    def __init__(
        self,
        D: int,
        block_size: int,
        seed: int = 42,
        num_hashes: int = 1,
        K: int = 50,
    ):
        """Initialize BBPM addressing.

        Args:
            D: Total memory slots
            block_size: Block size L (must be power of 2, and log2(L) must be even)
            seed: Random seed
            num_hashes: Number of independent hashes H (default 1)
            K: Active slots per item per hash (must be <= block_size)

        Raises:
            ValueError: If block_size is not power of 2
            ValueError: If D % block_size != 0
            ValueError: If log2(block_size) is not even
            ValueError: If K > block_size
        """
        # Validate power of two
        if block_size & (block_size - 1) != 0:
            raise ValueError(f"block_size ({block_size}) must be power of 2")
        
        # Validate divisibility
        if D % block_size != 0:
            raise ValueError(f"D ({D}) must be divisible by block_size ({block_size})")
        
        # Validate even n_bits
        n_bits = int(math.log2(block_size))
        if n_bits % 2 != 0:
            raise ValueError(
                f"log2(block_size) ({n_bits}) must be even for Feistel split. "
                f"Valid block sizes: 256 (n=8), 1024 (n=10), 4096 (n=12), 16384 (n=14), etc."
            )
        
        # Validate K <= L for distinct offsets guarantee
        if K > block_size:
            raise ValueError(
                f"K ({K}) must be <= block_size ({block_size}) for distinct offsets guarantee"
            )
        
        self.D = D
        self.block_size = block_size
        self.num_blocks = D // block_size
        self.seed = seed
        self.num_hashes = num_hashes
        self.K = K
        self.n_bits = n_bits

    def indices(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor:
        """Return [B, K*H] int64 tensor of memory indices.

        Args:
            keys: Input keys of shape [B] (int64)
            K: Number of active slots per item per hash
            H: Number of independent hashes

        Returns:
            Memory indices of shape [B, K*H] (int64), all in [0, D)

        NOTE: Uses self.K and self.num_hashes if K and H match, otherwise uses provided values.
        For consistency, validates K <= block_size.
        """
        keys = keys.long()
        B = keys.shape[0]
        device = keys.device
        
        # Validate K <= block_size
        if K > self.block_size:
            raise ValueError(
                f"K ({K}) must be <= block_size ({self.block_size}) for distinct offsets guarantee"
            )
        
        # Use provided K and H (allows override for testing)
        K_use = K
        H_use = H
        
        indices_list = []
        
        # For each hash function
        for h in range(H_use):
            # Step 1: Block selection
            block_id = select_block(keys, self.num_blocks, self.seed, h)  # [B]
            block_start = block_id * self.block_size  # [B]
            
            # Step 2: Intra-block PRP offsets
            offsets = prp_offsets(
                keys, K_use, self.block_size, self.seed, h, num_rounds=6
            )  # [B, K]
            
            # Step 3: Combine block start and offsets
            global_idx = block_start.unsqueeze(-1) + offsets  # [B, K]
            
            indices_list.append(global_idx)
        
        # Concatenate all hashes: [B, K*H]
        result = torch.cat(indices_list, dim=1)  # [B, K*H]
        
        # Ensure all indices are in valid range [0, D)
        result = result.clamp(0, self.D - 1)
        
        return result
