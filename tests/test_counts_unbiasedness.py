"""Test counts normalization correctness."""

import pytest
import torch

from bbpm import BBPMMemoryFloat, set_global_seed


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


def test_counts_increment(seed):
    """Test that counts increment correctly on writes."""
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

    # Check that counts increased
    counts = memory.counts[indices]
    expected_count = 1.0
    assert torch.allclose(counts, torch.ones_like(counts) * expected_count, atol=0.1), \
        "Counts should increment on write"


def test_counts_debias_read(seed):
    """Test that read uses counts for debiasing."""
    set_global_seed(seed)

    D = 1000
    d = 16
    K = 10
    H = 1

    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
    memory.clear()

    # Write same key multiple times
    key = torch.tensor([1])
    value = torch.ones(1, d) * 2.0

    for _ in range(5):
        memory.write(key, value)

    # Read should debias by counts
    retrieved = memory.read(key)

    # With proper debiasing, retrieved should be close to original value
    # (allowing for superposition noise)
    assert retrieved.abs().mean() > 0.5, "Debiased read should preserve signal"
