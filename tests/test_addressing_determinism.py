"""Test addressing determinism for BBPMAddressing and GlobalAffineHash."""

import pytest

import torch

from bbpm import GlobalAffineHash, set_global_seed
from bbpm.addressing.bbpm_addressing import BBPMAddressing


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


def test_bbpm_addressing_determinism(seed):
    """Test that BBPMAddressing produces same indices for same keys."""
    set_global_seed(seed)

    D = 100352  # Divisible by 1024
    block_size = 1024
    K = 20
    H = 1

    addressing = BBPMAddressing(D, block_size, seed=seed, num_hashes=H, K=K)
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)

    indices1 = addressing.indices(keys, K, H)
    indices2 = addressing.indices(keys, K, H)

    assert torch.equal(indices1, indices2), "BBPMAddressing should be deterministic"


def test_global_hash_determinism(seed):
    """Test that GlobalAffineHash produces same indices for same keys."""
    set_global_seed(seed)

    D = 100352
    hash_fn = GlobalAffineHash(D=D, seed=seed)
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)

    indices1 = hash_fn.indices(keys, K=10, H=1)
    indices2 = hash_fn.indices(keys, K=10, H=1)

    assert torch.equal(indices1, indices2), "GlobalAffineHash should be deterministic"


def test_addressing_indices_range(seed):
    """Test that addressing indices are within valid range."""
    set_global_seed(seed)

    D = 100352
    block_size = 1024
    addressing = BBPMAddressing(D, block_size, seed=seed)
    keys = torch.tensor([1, 2, 3, 100, 1000], dtype=torch.int64)

    indices = addressing.indices(keys, K=10, H=1)

    assert (indices >= 0).all(), "Indices should be non-negative"
    assert (indices < D).all(), f"Indices should be less than D={D}"


def test_global_hash_indices_range(seed):
    """Test that global hash indices are within valid range."""
    set_global_seed(seed)

    D = 100352
    hash_fn = GlobalAffineHash(D=D, seed=seed)
    keys = torch.tensor([1, 2, 3, 100, 1000], dtype=torch.int64)

    indices = hash_fn.indices(keys, K=10, H=1)

    assert (indices >= 0).all(), "Indices should be non-negative"
    assert (indices < D).all(), f"Indices should be less than D={D}"
