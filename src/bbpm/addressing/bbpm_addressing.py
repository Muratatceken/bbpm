"""BBPM addressing: block-based permutation memory addressing.

Implements two-stage addressing:
1. Block selection via deterministic hash
2. Intra-block permutation via Feistel PRP
"""

import math
from typing import List

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
    
    @staticmethod
    def suggest_valid_block_sizes(approx_D: int, max_suggestions: int = 5) -> List[int]:
        """Suggest valid block sizes near target D.
        
        Returns block sizes that are:
        - Powers of 2
        - Have even n_bits (log2(L) is even, e.g., 256, 1024, 4096)
        - Reasonably divide D or are close to common divisors
        
        Args:
            approx_D: Approximate total memory size D
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of valid block sizes in ascending order
        """
        # Valid block sizes with even n_bits: 256 (n=8), 1024 (n=10), 4096 (n=12), etc.
        valid_sizes = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
        
        suggestions = []
        for size in valid_sizes:
            # Include if it divides D or is a reasonable fraction
            if approx_D % size == 0:
                suggestions.append(size)
            elif size * 10 < approx_D:  # Include if not too large
                suggestions.append(size)
            
            if len(suggestions) >= max_suggestions:
                break
        
        # If no suggestions yet, return common valid sizes that are smaller than D
        if not suggestions:
            for size in valid_sizes:
                if size < approx_D:
                    suggestions.append(size)
                if len(suggestions) >= max_suggestions:
                    break
        
        return sorted(suggestions)
    
    @staticmethod
    def _validate_block_size(block_size: int, D: int) -> None:
        """Validate block_size with helpful error messages.
        
        Args:
            block_size: Block size to validate
            D: Total memory size
        
        Raises:
            ValueError: With suggestions if validation fails
        """
        # Check power of two
        if block_size & (block_size - 1) != 0:
            suggestions = BBPMAddressing.suggest_valid_block_sizes(D)
            raise ValueError(
                f"block_size ({block_size}) must be power of 2.\n"
                f"Valid block sizes for D={D}: {suggestions}\n"
                f"Example: block_size={suggestions[0] if suggestions else 1024}"
            )
        
        # Check even n_bits
        n_bits = int(math.log2(block_size))
        if n_bits % 2 != 0:
            suggestions = BBPMAddressing.suggest_valid_block_sizes(D)
            raise ValueError(
                f"log2(block_size) ({n_bits}) must be even for Feistel split.\n"
                f"block_size={block_size} (2^{n_bits}) is invalid.\n"
                f"Valid block sizes with even n_bits: {suggestions}\n"
                f"Example: block_size={suggestions[0] if suggestions else 1024} (2^10)"
            )
        
        # Check divisibility
        if D % block_size != 0:
            suggestions = BBPMAddressing.suggest_valid_block_sizes(D)
            raise ValueError(
                f"D ({D}) must be divisible by block_size ({block_size}).\n"
                f"Suggested block sizes for D={D}: {suggestions}\n"
                f"Example: Use block_size={suggestions[0] if suggestions else 1024}"
            )

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
            ValueError: If block_size is not power of 2 (with suggestions)
            ValueError: If D % block_size != 0 (with suggestions)
            ValueError: If log2(block_size) is not even (with suggestions)
            ValueError: If K > block_size
        """
        # Validate block_size with helpful error messages
        self._validate_block_size(block_size, D)
        
        n_bits = int(math.log2(block_size))
        
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
