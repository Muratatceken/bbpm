"""Test write-read identity in low load conditions."""

import pytest
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, set_global_seed
from bbpm.addressing.bbpm_addressing import BBPMAddressing


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


def test_write_read_with_bbpm_addressing(seed):
    """Test write-read using BBPMAddressing (PRP-based)."""
    set_global_seed(seed)

    D = 100000
    d = 64
    K = 32
    H = 1
    block_size = 1024  # Power of 2, even n_bits
    N = 100

    # Create BBPMAddressing
    addressing = BBPMAddressing(D, block_size, seed=seed, num_hashes=H, K=K)

    # Create memory with BBPMAddressing
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, hash_fn=addressing, device="cpu")
    memory.clear()

    keys = torch.arange(N, dtype=torch.int64)
    values = torch.randn(N, d)
    values = F.normalize(values, p=2, dim=1)

    memory.write(keys, values)
    retrieved = memory.read(keys)

    # Check cosine similarity (should be high in low load with PRP)
    cos_sim = F.cosine_similarity(retrieved, values, dim=1)
    assert cos_sim.mean() > 0.95, f"BBPMAddressing should preserve values, got {cos_sim.mean():.4f}"


def test_unity_gain_low_load(seed):
    """Test unity gain: write value v, read returns ≈ v in low load regime.
    
    With write_scale="unit" (default) and mean pooling, the memory should
    act as a unity-gain pass-through filter in low load conditions.
    """
    set_global_seed(seed)

    D = 1000000  # Very large memory for low load
    d = 32
    K = 50
    H = 1
    N = 100  # Small number of items (low load ratio)

    memory = BBPMMemoryFloat(
        D=D, d=d, K=K, H=H, 
        write_scale="unit",  # Unity gain
        device="cpu"
    )
    memory.clear()

    # Write normalized vectors
    keys = torch.arange(N, dtype=torch.int64)
    values = torch.randn(N, d)
    values = F.normalize(values, p=2, dim=1)
    
    memory.write(keys, values)
    retrieved = memory.read(keys)

    # Check cosine similarity (should be very high in low load)
    cos_sim = F.cosine_similarity(retrieved, values, dim=1)
    mean_cos = cos_sim.mean().item()
    
    assert mean_cos > 0.98, (
        f"Unity gain not achieved. Mean cosine similarity: {mean_cos:.4f}, expected > 0.98. "
        f"This indicates signal is being attenuated or lost."
    )
    
    # Also check magnitude preservation (allowing for small noise)
    retrieved_mags = retrieved.norm(p=2, dim=1)
    values_mags = values.norm(p=2, dim=1)  # Should be ~1.0 (normalized)
    
    # Retrieved magnitudes should be close to 1.0 (allowing for superposition noise)
    assert retrieved_mags.mean().item() > 0.9, (
        f"Signal magnitude not preserved. Mean retrieved magnitude: {retrieved_mags.mean():.4f}, "
        f"expected > 0.9. This suggests signal attenuation."
    )
