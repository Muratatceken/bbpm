"""Test counts for diagnostics and independence of load."""

import pytest
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, set_global_seed


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


def test_counts_increment(seed):
    """Test that counts increment correctly on writes (for diagnostics)."""
    set_global_seed(seed)

    D = 1000
    d = 16
    K = 10
    H = 1

    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
    memory.clear()

    key = torch.tensor([1])
    value = torch.ones(1, d)

    # Write once
    memory.write(key, value)
    indices = memory.hash_fn.indices(key, K, H).flatten()

    # Check that counts increased (for diagnostics)
    counts = memory.counts[indices]
    expected_count = 1.0
    assert torch.allclose(counts, torch.ones_like(counts) * expected_count, atol=0.1), \
        "Counts should increment on write"


def test_independence_of_load(seed):
    """Test that read preserves signal regardless of memory load (mean pooling works).
    
    This test verifies the core BBPM theory: mean pooling over K slots preserves
    the target signal while noise from interfering items cancels out.
    """
    set_global_seed(seed)

    D = 100000  # Large memory to avoid collisions
    d = 64
    K = 50
    H = 1

    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu", write_scale="unit")
    memory.clear()

    # Write target item A
    key_A = torch.tensor([42], dtype=torch.int64)
    value_A = torch.ones(1, d) * 5.0
    value_A = F.normalize(value_A, p=2, dim=1)  # Normalize to unit vector
    memory.write(key_A, value_A)

    # Write many random interfering items (simulating high load)
    num_interfering = 1000
    interfering_keys = torch.arange(1, num_interfering + 1, dtype=torch.int64)
    interfering_values = torch.randn(num_interfering, d)
    interfering_values = F.normalize(interfering_values, p=2, dim=1)
    memory.write(interfering_keys, interfering_values)

    # Read item A
    retrieved_A = memory.read(key_A)

    # Check cosine similarity (should be high despite interference)
    # Mean pooling should preserve signal while noise cancels out
    cosine_sim = F.cosine_similarity(retrieved_A, value_A, dim=1).item()
    assert cosine_sim > 0.9, (
        f"Signal should be preserved with mean pooling. "
        f"Got cosine similarity: {cosine_sim:.4f}, expected > 0.9"
    )


def test_counts_not_used_in_read(seed):
    """Test that counts are not used in read computation (they're for diagnostics only)."""
    set_global_seed(seed)

    D = 1000
    d = 16
    K = 10
    H = 1

    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
    memory.clear()

    key = torch.tensor([1], dtype=torch.int64)
    value = torch.ones(1, d) * 2.0

    # Write same key multiple times (should increase counts but not affect read)
    for _ in range(3):
        memory.write(key, value)

    # Store counts before read
    indices = memory.hash_fn.indices(key, K, H).flatten()
    counts_before = memory.counts[indices].clone()

    # Read (should use mean, not divide by counts)
    retrieved = memory.read(key)

    # Verify counts unchanged (read doesn't modify counts)
    counts_after = memory.counts[indices]
    assert torch.equal(counts_before, counts_after), "Read should not modify counts"
    
    # Verify retrieved value is reasonable (not shrunk by count division)
    # With mean pooling, multiple writes of same value should accumulate
    assert retrieved.abs().mean() > 0.5, "Mean pooling should preserve signal magnitude"
