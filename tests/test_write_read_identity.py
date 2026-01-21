"""Test write-read identity in low load conditions."""

import pytest
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, set_global_seed


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


def test_write_read_identity_low_load(seed):
    """Test that write-read preserves values in low load conditions."""
    set_global_seed(seed)

    D = 100000  # Large memory
    d = 64
    K = 32
    H = 1
    N = 100  # Small number of items (low load)

    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
    memory.clear()

    # Generate keys and values
    keys = torch.arange(N)
    values = torch.randn(N, d)
    values = F.normalize(values, p=2, dim=1)

    # Write
    memory.write(keys, values)

    # Read
    retrieved = memory.read(keys)

    # Check cosine similarity (should be very high in low load)
    cos_sim = F.cosine_similarity(retrieved, values, dim=1)
    assert cos_sim.mean() > 0.95, f"Low load should preserve values, got {cos_sim.mean():.4f}"


def test_write_read_exact_match_single_item(seed):
    """Test exact match for single item in isolation."""
    set_global_seed(seed)

    D = 10000
    d = 16
    K = 10
    H = 1

    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
    memory.clear()

    key = torch.tensor([42])
    value = torch.ones(1, d) * 5.0
    value = F.normalize(value, p=2, dim=1)

    memory.write(key, value)
    retrieved = memory.read(key)

    # Should be very close (allowing for small numerical error)
    cos_sim = F.cosine_similarity(retrieved, value, dim=1).item()
    assert cos_sim > 0.99, f"Single item should match closely, got {cos_sim:.4f}"
