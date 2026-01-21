"""Feistel network PRP for BBPM addressing.

Implements a vectorized Feistel network PRP that operates on exact bit domains.
Guarantees bijection and distinct offsets for K <= L.
"""

import math

import torch

from .block_selector import MASK64

# Constants for seed decorrelation
CONST_B = 0x517CC1B727220A95  # PRP seed constant (different from block selection)
LARGE_CONST = 0x9E3779B97F4A7C15  # Large constant for non-linear h mixing


def derive_round_keys_vectorized(
    prp_seeds: torch.Tensor,
    num_rounds: int = 6
) -> torch.Tensor:
    """Derive round keys for Feistel network.
    
    Args:
        prp_seeds: Per-key PRP seeds of shape [B] (int64)
        num_rounds: Number of Feistel rounds (default 6)
    
    Returns:
        Round keys of shape [num_rounds, B] (int64)
    """
    device = prp_seeds.device
    B = prp_seeds.shape[0]
    
    # Pre-allocate round keys
    round_keys = torch.zeros(num_rounds, B, dtype=torch.int64, device=device)
    
    # Derive round keys for each round
    for r in range(num_rounds):
        # Mix seed with round number and large constant
        round_input = prp_seeds + (r * 0x9E3779B97F4A7C15)
        round_input = round_input & MASK64
        round_keys[r] = (round_input * 0xBF58476D1CE4E5B9) & MASK64
        round_keys[r] = round_keys[r] ^ (round_keys[r] >> 31)
        round_keys[r] = round_keys[r] & MASK64
    
    return round_keys


def feistel_round_function(
    right: torch.Tensor,
    round_key: torch.Tensor,
    half_mask: int
) -> torch.Tensor:
    """Feistel round function F(right, round_key).
    
    Args:
        right: Right half of Feistel input, shape matches x
        round_key: Round key, shape [B] or broadcastable to match right
        half_mask: Mask for half-width values: (1 << half_bits) - 1
    
    Returns:
        Round function output, masked to half_bits
    """
    # Broadcast round_key if needed
    if right.dim() > round_key.dim():
        round_key = round_key.unsqueeze(-1)
    
    # Round function: mix right half with round key
    f_out = right ^ round_key
    f_out = f_out & MASK64
    f_out = (f_out * 0x9E3779B97F4A7C15) & MASK64
    f_out = f_out ^ (f_out >> 16)
    f_out = f_out & MASK64
    
    # Mask to half_bits
    f_out = f_out & half_mask
    
    return f_out


def feistel_prp_vectorized(
    x: torch.Tensor,
    n_bits: int,
    round_keys: torch.Tensor,
    num_rounds: int = 6
) -> torch.Tensor:
    """Apply Feistel PRP to tensor x.
    
    Args:
        x: Input tensor of shape [B, K] or [B] (int64), values in [0, 2^n_bits)
        n_bits: Bit width of domain (L = 2^n_bits), MUST be even
        round_keys: Round keys of shape [num_rounds, B] (int64)
        num_rounds: Number of Feistel rounds (default 6)
    
    Returns:
        Permuted tensor of same shape as x, values in [0, 2^n_bits)
    
    NOTE:
    - NO modulo operation after PRP - PRP output is directly used as offset
    - Fixed loop over rounds (6 iterations) is acceptable and necessary
    - All arithmetic operations must mask with MASK64 for uint64 semantics
    """
    assert n_bits % 2 == 0, f"n_bits ({n_bits}) must be even for Feistel split"
    
    half_bits = n_bits // 2
    half_mask = (1 << half_bits) - 1
    full_mask = (1 << n_bits) - 1
    
    # Split into left and right halves
    left = (x >> half_bits) & half_mask
    right = x & half_mask
    
    # Fixed loop over rounds (acceptable - constant 6 iterations)
    for r in range(num_rounds):
        # Get round key for this round
        round_key = round_keys[r]  # [B]
        
        # Apply round function
        f_out = feistel_round_function(right, round_key, half_mask)
        
        # Feistel swap
        new_right = (left ^ f_out) & half_mask
        left = right
        right = new_right
    
    # Recombine
    result = (left << half_bits) | right
    result = result & full_mask  # Ensure in [0, 2^n_bits)
    
    return result


def prp_offsets(
    keys: torch.Tensor,
    K: int,
    L: int,
    seed: int,
    h: int,
    num_rounds: int = 6
) -> torch.Tensor:
    """Generate K distinct offsets using PRP. Returns [B, K] int64 tensor.
    
    Args:
        keys: Key tensor of shape [B] (int64)
        K: Number of offsets per key (must be <= L)
        L: Block size (must be power of 2, and log2(L) must be even)
        seed: Random seed
        h: Hash function index (for multi-hash independence)
        num_rounds: Number of Feistel rounds
    
    Returns:
        Offsets tensor of shape [B, K], all values in [0, L)
        Guaranteed distinct per key (when K <= L) by PRP bijection property
    
    Implementation:
        1. Validate: L is power of 2, n_bits = log2(L) is even
        2. Derive independent PRP seed (decorrelated from block selection):
           - seed_prp_base = mix64(seed ^ CONST_B)
           - h_seed = mix64(seed_prp_base ^ (h * LARGE_CONST))
           - prp_seed = mix64(keys ^ h_seed)  # [B]
        3. Derive round keys: [num_rounds, B] tensor (vectorized)
        4. Create input: x = arange(K)[None, :].expand(B, K)  # [B, K] with distinct rows
        5. Apply PRP: offsets = feistel_prp_vectorized(x, n_bits, round_keys)
        6. Return offsets (already in [0, L) by construction, no modulo needed)
    """
    from .block_selector import mix64
    
    device = keys.device
    keys = keys.long()
    
    # Validate L is power of 2
    if L & (L - 1) != 0:
        raise ValueError(f"L ({L}) must be power of 2")
    
    # Validate n_bits is even
    n_bits = int(math.log2(L))
    if n_bits % 2 != 0:
        raise ValueError(f"log2(L) ({n_bits}) must be even for Feistel split. Use L=256, 1024, 4096, etc.")
    
    # Validate K <= L
    if K > L:
        raise ValueError(f"K ({K}) must be <= L ({L}) for distinct offsets guarantee")
    
    # Derive independent PRP seed (decorrelated from block selection)
    seed_prp_base_tensor = torch.tensor([seed ^ CONST_B], dtype=torch.int64, device=device)
    seed_prp_base = mix64(seed_prp_base_tensor)[0]
    
    # Non-linear h mixing for multi-hash independence
    h_tensor = torch.tensor([seed_prp_base ^ (h * LARGE_CONST)], dtype=torch.int64, device=device)
    h_seed = mix64(h_tensor)[0]
    
    # Per-key PRP seeds
    prp_seeds = mix64(keys ^ h_seed)  # [B]
    
    # Derive round keys (vectorized)
    round_keys = derive_round_keys_vectorized(prp_seeds, num_rounds)  # [num_rounds, B]
    
    # Create input: distinct values 0..K-1 for each key
    x = torch.arange(K, dtype=torch.int64, device=device)[None, :].expand(keys.shape[0], K)  # [B, K]
    
    # Apply PRP (vectorized over batch)
    offsets = feistel_prp_vectorized(x, n_bits, round_keys, num_rounds)  # [B, K]
    
    # Offsets are already in [0, L) by construction (PRP output is in [0, 2^n_bits) = [0, L))
    return offsets
