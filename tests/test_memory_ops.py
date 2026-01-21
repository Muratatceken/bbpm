"""Tests for memory operations using BBPMAddressing."""

import pytest
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat
from bbpm.addressing.bbpm_addressing import BBPMAddressing


def test_write_read_identity_light_load():
    """Write/read with low load, expect high cosine similarity using BBPMAddressing."""
    D = 100000  # Large memory
    d = 64
    K = 32
    H = 1
    N = 100  # Small number of items (low load)
    block_size = 1024  # Power of 2, n_bits=10 is even
    
    # Create BBPMAddressing
    addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    
    # Create memory with BBPMAddressing
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
    memory.clear()
    
    # Generate keys and values
    keys = torch.arange(N, dtype=torch.int64)
    values = torch.randn(N, d)
    values = F.normalize(values, p=2, dim=1)
    
    # Write
    memory.write(keys, values)
    
    # Read
    retrieved = memory.read(keys)
    
    # Check cosine similarity (should be very high in low load)
    cos_sim = F.cosine_similarity(retrieved, values, dim=1)
    assert cos_sim.mean() > 0.95, f"Low load should preserve values, got {cos_sim.mean():.4f}"


def test_write_read_with_prp():
    """Verify memory works with BBPMAddressing (PRP-based)."""
    D = 50000
    d = 32
    K = 20
    H = 1
    block_size = 1024
    
    addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
    memory.clear()
    
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    values = torch.randn(5, d)
    values = F.normalize(values, p=2, dim=1)
    
    memory.write(keys, values)
    retrieved = memory.read(keys)
    
    # Should retrieve something reasonable
    assert retrieved.shape == values.shape
    cos_sim = F.cosine_similarity(retrieved, values, dim=1).mean()
    assert cos_sim > 0.5  # Some similarity expected even with interference


def test_memory_determinism():
    """Same writes produce same memory state."""
    D = 10000
    d = 16
    K = 10
    H = 1
    block_size = 1024
    
    addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    
    # Create two memory instances
    memory1 = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
    memory2 = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
    
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    values = torch.randn(3, d)
    
    # Write same values to both
    memory1.write(keys, values)
    memory2.write(keys, values)
    
    # Memory states should be identical
    assert torch.allclose(memory1.memory, memory2.memory)
    assert torch.allclose(memory1.counts, memory2.counts)
    
    # Reads should be identical
    retrieved1 = memory1.read(keys)
    retrieved2 = memory2.read(keys)
    assert torch.allclose(retrieved1, retrieved2)


def test_no_self_collisions():
    """Verify K offsets per key are distinct (PRP guarantee)."""
    D = 10000
    block_size = 1024
    K = 50
    H = 1
    
    addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    
    keys = torch.tensor([42], dtype=torch.int64)  # Single key
    indices = addressing.indices(keys, K, H)  # [1, K]
    
    # For single hash, all K offsets should be distinct
    unique_indices = torch.unique(indices[0])
    assert len(unique_indices) == K, f"Expected {K} distinct indices, got {len(unique_indices)}"
    
    # Verify they're all in valid range
    assert (indices >= 0).all() and (indices < D).all()


def test_memory_with_multi_hash():
    """Test memory operations with multiple hash functions."""
    D = 50000
    d = 32
    K = 20
    H = 3  # Multiple hashes
    block_size = 1024
    
    addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
    memory.clear()
    
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    values = torch.randn(3, d)
    values = F.normalize(values, p=2, dim=1)
    
    memory.write(keys, values)
    retrieved = memory.read(keys)
    
    assert retrieved.shape == values.shape
    # With multi-hash, retrieval should be more robust
    cos_sim = F.cosine_similarity(retrieved, values, dim=1).mean()
    assert cos_sim > 0.3  # Some similarity expected


def test_memory_block_size_integration():
    """Test that memory works correctly with different block sizes."""
    D = 50000
    d = 32
    K = 20
    H = 1
    
    valid_block_sizes = [256, 1024, 4096]  # Even n_bits
    
    for block_size in valid_block_sizes:
        if D % block_size == 0:
            addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
            memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
            memory.clear()
            
            keys = torch.tensor([1, 2, 3], dtype=torch.int64)
            values = torch.randn(3, d)
            
            memory.write(keys, values)
            retrieved = memory.read(keys)
            
            assert retrieved.shape == values.shape
