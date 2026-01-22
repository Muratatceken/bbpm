"""Block selector for BBPM addressing.

Implements deterministic block selection with independent seed from PRP.
"""

import torch

# 64-bit mask for uint64 wrap semantics
MASK64 = (1 << 64) - 1  # 0xFFFFFFFFFFFFFFFF

# Constants for seed decorrelation
CONST_A = 0x9E3779B97F4A7C15  # Block selection seed constant
CONST_C = 0x517CC1B727220A95  # Multi-hash constant for block selection


def mix64(x: torch.Tensor) -> torch.Tensor:
    """64-bit mixing function (SplitMix64-style), vectorized over tensor.
    
    CRITICAL: Enforces uint64 wrap semantics for CPU/GPU determinism.
    After every arithmetic operation, applies: x &= MASK64
    
    Args:
        x: Input tensor of shape [B] or any shape (int64)
    
    Returns:
        Mixed tensor of same shape, values treated as uint64-in-int64
    """
    x = x ^ (x >> 33)
    x = x & MASK64
    x = (x * 0xFF51AFD7ED558CCD) & MASK64
    x = x ^ (x >> 33)
    x = x & MASK64
    x = (x * 0xC4CEB9FE1A85EC53) & MASK64
    x = x ^ (x >> 33)
    x = x & MASK64
    return x


def select_block(
    keys: torch.Tensor, 
    num_blocks: int, 
    seed: int, 
    h: int
) -> torch.Tensor:
    """Select block IDs for keys. Returns [B] int64 tensor.
    
    Uses independent seed_block (decorrelated from PRP seed).
    Fully vectorized: no Python loops over batch.
    
    Args:
        keys: Key tensor of shape [B] (int64)
        num_blocks: Number of blocks B
        seed: Random seed
        h: Hash function index (for multi-hash independence)
    
    Returns:
        Block IDs of shape [B] (int64), values in [0, num_blocks)
    """
    device = keys.device
    keys = keys.long()
    
    # Derive independent block selection seed (decorrelated from PRP)
    # Mask seed to 64 bits to avoid Python int overflow
    seed_masked = (seed ^ CONST_A) & MASK64
    seed_tensor = torch.tensor([seed_masked], dtype=torch.int64, device=device)
    seed_block = mix64(seed_tensor)[0]
    
    # Incorporate h into seed (for multi-hash independence)
    # Use non-linear mixing to avoid structure
    # Mask to avoid overflow
    h_value = (seed_block ^ (h * CONST_C)) & MASK64
    h_tensor = torch.tensor([h_value], dtype=torch.int64, device=device)
    h_seed = mix64(h_tensor)[0]
    
    # Mix keys with h-dependent seed (vectorized over batch)
    mixed = mix64(keys ^ h_seed)
    block_id = (mixed % num_blocks).long()
    
    return block_id
