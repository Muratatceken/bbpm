"""Tests for BBPM addressing implementation."""

import pytest

import torch

from bbpm.addressing.bbpm_addressing import BBPMAddressing
from bbpm.addressing.prp_feistel import prp_offsets


def test_offsets_distinct():
    """Test that offsets are distinct per key (guaranteed by PRP bijection)."""
    keys = torch.randint(0, 10000, (100,), dtype=torch.int64)
    K = 50
    L = 1024  # Power of 2, n_bits=10 is even
    
    offsets = prp_offsets(keys, K, L, seed=42, h=0)  # [100, 50]
    
    # Verify each key has K distinct offsets
    for i in range(len(keys)):
        unique_offsets = torch.unique(offsets[i])
        assert len(unique_offsets) == K, (
            f"Key {keys[i]} has collisions: {len(unique_offsets)} != {K}"
        )
        assert offsets[i].min() >= 0 and offsets[i].max() < L


def test_offsets_input_distinctness():
    """Test that distinct inputs (0..K-1) produce distinct outputs via PRP."""
    # This is the mathematical guarantee: PRP bijection ensures distinct outputs
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    K = 8
    L = 256  # Power of 2, n_bits=8 is even
    
    offsets = prp_offsets(keys, K, L, seed=42, h=0)  # [3, 8]
    
    # Each row should have K distinct values (by PRP property)
    for i in range(len(keys)):
        assert len(torch.unique(offsets[i])) == K


def test_offsets_range():
    """Test that all offsets are in valid range [0, L)."""
    keys = torch.randint(0, 10000, (50,), dtype=torch.int64)
    K = 30
    L = 1024
    
    offsets = prp_offsets(keys, K, L, seed=42, h=0)
    
    assert (offsets >= 0).all(), "All offsets must be non-negative"
    assert (offsets < L).all(), f"All offsets must be < L ({L})"


def test_addressing_determinism():
    """Test that same keys produce same addresses."""
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    D = 10240
    block_size = 1024
    K = 20
    H = 1
    seed = 42
    
    addr = BBPMAddressing(D, block_size, seed=seed, num_hashes=H, K=K)
    
    indices1 = addr.indices(keys, K, H)
    indices2 = addr.indices(keys, K, H)
    
    assert torch.equal(indices1, indices2), "Addressing must be deterministic"


def test_address_range():
    """Test that all addresses are in valid range [0, D)."""
    keys = torch.randint(0, 10000, (100,), dtype=torch.int64)
    D = 10240
    block_size = 1024
    K = 20
    H = 1
    
    addr = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    indices = addr.indices(keys, K, H)
    
    assert (indices >= 0).all(), "All addresses must be non-negative"
    assert (indices < D).all(), f"All addresses must be < D ({D})"


def test_block_selection():
    """Test that block IDs are in valid range [0, B)."""
    keys = torch.randint(0, 10000, (100,), dtype=torch.int64)
    D = 10240
    block_size = 1024
    num_blocks = D // block_size  # Should be 9
    
    addr = BBPMAddressing(D, block_size, seed=42)
    
    # We can't directly access block IDs, but we can verify addresses
    # are within expected block ranges
    indices = addr.indices(keys, K=10, H=1)
    
    # Each address should fall into one of the blocks
    block_ids_from_addr = indices // block_size
    assert (block_ids_from_addr >= 0).all()
    assert (block_ids_from_addr < num_blocks).all()


def test_multi_hash_independence():
    """Test that different H values produce different addresses."""
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    D = 10240
    block_size = 1024
    K = 20
    seed = 42
    
    # Generate addresses with H=1
    addr1 = BBPMAddressing(D, block_size, seed=seed, num_hashes=1, K=K)
    indices1 = addr1.indices(keys, K, H=1)
    
    # Generate addresses with H=3
    addr3 = BBPMAddressing(D, block_size, seed=seed, num_hashes=3, K=K)
    indices3 = addr3.indices(keys, K, H=3)
    
    # Should have different shapes
    assert indices1.shape == (len(keys), K * 1)
    assert indices3.shape == (len(keys), K * 3)
    
    # First K columns of indices3 should be different from indices1
    # (different hash functions should produce different block selections/offsets)
    # We can't guarantee all are different, but we can check they're not all identical
    first_h_of_h3 = indices3[:, :K]
    not_all_identical = not torch.equal(indices1, first_h_of_h3)
    assert not_all_identical, "Different hash indices should produce different addresses"


def test_block_size_power_of_two():
    """Test that valid block sizes (power of 2, even n_bits) work."""
    valid_block_sizes = [256, 1024, 4096, 16384]  # n_bits = 8, 10, 12, 14 (all even)
    D = 102400
    
    for block_size in valid_block_sizes:
        if D % block_size == 0:
            addr = BBPMAddressing(D, block_size, seed=42)
            assert addr.block_size == block_size


def test_block_size_not_power_of_two():
    """Test that invalid block sizes (not power of 2) raise ValueError."""
    invalid_block_sizes = [100, 500, 1000, 3000]
    D = 102400
    
    for block_size in invalid_block_sizes:
        if D % block_size == 0:  # Only test if divisible
            with pytest.raises(ValueError, match="must be power of 2"):
                BBPMAddressing(D, block_size, seed=42)


def test_block_size_odd_n_bits():
    """Test that block sizes with odd n_bits (e.g., 512, 2048) raise ValueError."""
    # 512 = 2^9 (n_bits=9 is odd), 2048 = 2^11 (n_bits=11 is odd)
    invalid_block_sizes = [512, 2048, 8192]
    D = 102400
    
    for block_size in invalid_block_sizes:
        if D % block_size == 0:
            with pytest.raises(ValueError, match="must be even"):
                BBPMAddressing(D, block_size, seed=42)


def test_k_gt_block_size_rejected():
    """Test that K > block_size raises ValueError."""
    D = 10240
    block_size = 1024
    K = 2000  # > block_size
    
    with pytest.raises(ValueError, match="must be <="):
        BBPMAddressing(D, block_size, seed=42, K=K)


def test_addressing_shape():
    """Test that addressing produces correct output shape."""
    B = 50
    keys = torch.randint(0, 10000, (B,), dtype=torch.int64)
    D = 10240
    block_size = 1024
    K = 20
    H = 3
    
    addr = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    indices = addr.indices(keys, K, H)
    
    assert indices.shape == (B, K * H), f"Expected shape {(B, K * H)}, got {indices.shape}"


def test_no_self_collisions_within_block():
    """Test that addresses for same key have no self-collisions within a block."""
    keys = torch.tensor([42], dtype=torch.int64)  # Single key
    D = 10240
    block_size = 1024
    K = 50
    H = 1
    
    addr = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    indices = addr.indices(keys, K, H)  # [1, K]
    
    # All indices should be in same block (or adjacent blocks for multi-hash)
    # But more importantly, for single hash, offsets should be distinct
    # We can verify by checking the block-local offsets
    block_start = (indices[0] // block_size) * block_size
    offsets = indices[0] - block_start
    
    # All offsets should be distinct (PRP guarantee)
    assert len(torch.unique(offsets)) == K, "Offsets within block must be distinct"
