"""Test hash function determinism."""

import pytest
import torch

from bbpm import BlockHash, GlobalAffineHash, set_global_seed


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


def test_global_hash_determinism(seed):
    """Test that global hash produces same indices for same keys."""
    set_global_seed(seed)

    hash_fn = GlobalAffineHash(D=1000, seed=seed)
    keys = torch.tensor([1, 2, 3, 4, 5])

    indices1 = hash_fn.indices(keys, K=10, H=1)
    indices2 = hash_fn.indices(keys, K=10, H=1)

    assert torch.equal(indices1, indices2), "Hash should be deterministic"


def test_block_hash_determinism(seed):
    """Test that block hash produces same indices for same keys."""
    set_global_seed(seed)

    hash_fn = BlockHash(D=1000, block_size=100, seed=seed)
    keys = torch.tensor([1, 2, 3, 4, 5])

    indices1 = hash_fn.indices(keys, K=10, H=1)
    indices2 = hash_fn.indices(keys, K=10, H=1)

    assert torch.equal(indices1, indices2), "Block hash should be deterministic"


def test_hash_indices_range(seed):
    """Test that hash indices are within valid range."""
    set_global_seed(seed)

    D = 1000
    hash_fn = GlobalAffineHash(D=D, seed=seed)
    keys = torch.tensor([1, 2, 3, 100, 1000])

    indices = hash_fn.indices(keys, K=10, H=1)

    assert (indices >= 0).all(), "Indices should be non-negative"
    assert (indices < D).all(), f"Indices should be less than D={D}"


def test_binary_bloom_determinism(seed):
    """Test that BinaryBBPMBloom produces deterministic results."""
    from bbpm import BinaryBBPMBloom

    set_global_seed(seed)

    D = 1000
    K = 10
    H = 1

    # Create two instances with same seed
    mem1 = BinaryBBPMBloom(D=D, K=K, H=H, seed=seed)
    mem2 = BinaryBBPMBloom(D=D, K=K, H=H, seed=seed)

    keys = torch.tensor([1, 2, 3, 4, 5])

    # Write same keys to both
    mem1.write(keys, torch.zeros(len(keys), 1))  # values not used
    mem2.write(keys, torch.zeros(len(keys), 1))

    # Read should produce same results
    scores1 = mem1.read(keys)
    scores2 = mem2.read(keys)

    assert torch.allclose(scores1, scores2), "BinaryBBPMBloom should be deterministic"
