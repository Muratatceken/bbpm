"""Tests for Feistel PRP implementation."""

import pytest

import torch

from bbpm.addressing.prp_feistel import (
    derive_round_keys_vectorized,
    feistel_prp_vectorized,
    prp_offsets,
)
from bbpm.addressing.block_selector import mix64


def test_prp_bijection_small():
    """Test PRP bijection using vectorized function (small domain)."""
    L = 256
    n_bits = 8
    seed = 42
    num_rounds = 6
    
    device = torch.device("cpu")
    
    # Create all inputs [256]
    x = torch.arange(L, dtype=torch.int64, device=device)
    
    # Derive round keys for single seed
    keys = torch.tensor([1], dtype=torch.int64, device=device)  # Single key
    prp_seeds = mix64(keys ^ (seed ^ 0x517CC1B727220A95))
    round_keys = derive_round_keys_vectorized(prp_seeds, num_rounds)  # [num_rounds, 1]
    
    # Apply PRP
    x_expanded = x.unsqueeze(0)  # [1, 256]
    y = feistel_prp_vectorized(x_expanded, n_bits, round_keys, num_rounds)  # [1, 256]
    y = y.squeeze(0)  # [256]
    
    # Verify bijection
    assert y.min() >= 0 and y.max() < L, f"Output out of range: [{y.min()}, {y.max()})"
    assert len(torch.unique(y)) == L, f"PRP must be bijection: {len(torch.unique(y))} != {L}"
    
    # Verify all values in domain
    assert (y >= 0).all() and (y < L).all(), "All outputs must be in [0, L)"


def test_prp_bijection_medium():
    """Test PRP bijection for medium domain."""
    L = 1024
    n_bits = 10
    seed = 42
    num_rounds = 6
    
    device = torch.device("cpu")
    
    # Create all inputs
    x = torch.arange(L, dtype=torch.int64, device=device)
    
    # Derive round keys
    keys = torch.tensor([1], dtype=torch.int64, device=device)
    prp_seeds = mix64(keys ^ (seed ^ 0x517CC1B727220A95))
    round_keys = derive_round_keys_vectorized(prp_seeds, num_rounds)
    
    # Apply PRP
    x_expanded = x.unsqueeze(0)
    y = feistel_prp_vectorized(x_expanded, n_bits, round_keys, num_rounds)
    y = y.squeeze(0)
    
    # Verify bijection
    assert y.min() >= 0 and y.max() < L
    assert len(torch.unique(y)) == L


def test_prp_vectorized_batch():
    """Test vectorized PRP works correctly for batch of keys (optimized for CI)."""
    # Use smaller L for full permutation test
    B = 10
    L = 256  # Small enough for full check
    n_bits = 8
    seed = 42
    num_rounds = 6
    
    device = torch.device("cpu")
    keys = torch.randint(0, 10000, (B,), dtype=torch.int64, device=device)
    
    # Derive per-key PRP seeds and round keys
    seed_prp_base = mix64(torch.tensor([seed ^ 0x517CC1B727220A95], dtype=torch.int64, device=device))[0]
    h_seed = mix64(torch.tensor([seed_prp_base ^ (0 * 0x9E3779B97F4A7C15)], dtype=torch.int64, device=device))[0]
    prp_seeds = mix64(keys ^ h_seed)  # [B]
    round_keys = derive_round_keys_vectorized(prp_seeds, num_rounds)  # [6, B]
    
    # Create input: each key gets distinct inputs 0..L-1
    x = torch.arange(L, dtype=torch.int64, device=device)[None, :].expand(B, L)  # [B, L]
    
    # Apply PRP
    y = feistel_prp_vectorized(x, n_bits, round_keys, num_rounds)  # [B, L]
    
    # Verify each row is a permutation (full check for small L)
    for i in range(B):
        assert len(torch.unique(y[i])) == L, f"Row {i} must be bijection"


def test_prp_vectorized_large_sample():
    """Test vectorized PRP on larger L with sampled rows (CI-friendly)."""
    B = 100
    L = 4096
    n_bits = 12
    seed = 42
    num_rounds = 6
    
    device = torch.device("cpu")
    keys = torch.randint(0, 10000, (B,), dtype=torch.int64, device=device)
    
    # Derive keys (same as above)
    seed_prp_base = mix64(torch.tensor([seed ^ 0x517CC1B727220A95], dtype=torch.int64, device=device))[0]
    h_seed = mix64(torch.tensor([seed_prp_base ^ (0 * 0x9E3779B97F4A7C15)], dtype=torch.int64, device=device))[0]
    prp_seeds = mix64(keys ^ h_seed)
    round_keys = derive_round_keys_vectorized(prp_seeds, num_rounds)
    
    # Test with K offsets (not full L)
    K = 50
    x = torch.arange(K, dtype=torch.int64, device=device)[None, :].expand(B, K)  # [B, K]
    y = feistel_prp_vectorized(x, n_bits, round_keys, num_rounds)  # [B, K]
    
    # Check distinctness on sampled rows (not all B)
    sample_indices = [0, B // 4, B // 2, 3 * B // 4, B - 1]
    for i in sample_indices:
        unique_count = len(torch.unique(y[i]))
        assert unique_count == K, f"Row {i} has {unique_count} unique values, expected {K}"


def test_prp_determinism():
    """Test that same inputs produce same outputs."""
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    K = 20
    L = 1024
    seed = 42
    h = 0
    
    # Generate offsets twice
    offsets1 = prp_offsets(keys, K, L, seed, h)
    offsets2 = prp_offsets(keys, K, L, seed, h)
    
    # Should be identical
    assert torch.equal(offsets1, offsets2), "PRP must be deterministic"


def test_prp_range():
    """Test that all PRP outputs are in valid range [0, 2^n_bits)."""
    keys = torch.randint(0, 10000, (50,), dtype=torch.int64)
    K = 30
    L = 512  # This would fail validation (n_bits=9 is odd), but let's use valid one
    L = 1024  # n_bits=10 is even
    seed = 42
    h = 0
    
    offsets = prp_offsets(keys, K, L, seed, h)
    
    assert (offsets >= 0).all(), "All offsets must be non-negative"
    assert (offsets < L).all(), f"All offsets must be < L ({L})"


def test_prp_exact_bit_domain():
    """Verify PRP operates on exact bit domain (no modulo after PRP)."""
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    K = 16
    L = 256  # 2^8
    n_bits = 8
    seed = 42
    h = 0
    
    offsets = prp_offsets(keys, K, L, seed, h)
    
    # All values should be in [0, 2^8) = [0, 256)
    assert (offsets >= 0).all() and (offsets < (1 << n_bits)).all()
    
    # Verify it's exactly the bit domain (not just < L)
    max_val = offsets.max().item()
    assert max_val < (1 << n_bits), f"Max value {max_val} should be < 2^{n_bits} = {1 << n_bits}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prp_cpu_gpu_equivalence():
    """Test CPU and GPU produce identical results (exact equality with MASK64 masking)."""
    keys = torch.randint(0, 10000, (50,), dtype=torch.int64)
    L = 1024
    K = 100
    seed = 42
    h = 0
    
    # CPU
    offsets_cpu = prp_offsets(keys, K, L, seed, h)
    
    # GPU
    keys_gpu = keys.cuda()
    offsets_gpu = prp_offsets(keys_gpu, K, L, seed, h)
    
    # Exact equality (deterministic integer ops with masking)
    assert torch.equal(offsets_cpu, offsets_gpu.cpu()), "CPU and GPU must match exactly"


def test_prp_odd_n_bits_rejected():
    """Test that odd n_bits (e.g., L=512, n_bits=9) is rejected."""
    keys = torch.tensor([1, 2], dtype=torch.int64)
    L = 512  # 2^9, n_bits=9 is odd
    K = 10
    seed = 42
    h = 0
    
    with pytest.raises(ValueError, match="must be even"):
        prp_offsets(keys, K, L, seed, h)
