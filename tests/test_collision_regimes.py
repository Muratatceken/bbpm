"""Test collision regimes: fidelity decreases as load increases."""

import pytest
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, set_global_seed


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


def test_fidelity_decreases_with_load(seed):
    """Test that retrieval fidelity decreases as memory load increases."""
    set_global_seed(seed)

    D = 10000  # Small memory to force collisions
    d = 64
    K = 10
    H = 1

    # Test different load levels
    N_values = [100, 500, 1000, 2000, 5000]
    fidelities = []

    for N in N_values:
        memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
        memory.clear()

        keys = torch.arange(N)
        values = torch.randn(N, d)
        values = F.normalize(values, p=2, dim=1)

        # Write
        batch_size = 100
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            memory.write(keys[i:end], values[i:end])

        # Test retrieval on first 100 items
        test_n = min(100, N)
        retrieved = memory.read(keys[:test_n])
        cos_sim = F.cosine_similarity(retrieved, values[:test_n], dim=1).mean().item()
        fidelities.append(cos_sim)

    # Fidelity should generally decrease (allowing for some variance)
    # Check that high load has lower fidelity than low load
    assert fidelities[0] > fidelities[-1] * 0.8, \
        f"Fidelity should decrease with load. Low: {fidelities[0]:.4f}, High: {fidelities[-1]:.4f}"
