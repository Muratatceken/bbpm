"""Tests for BlockAddress.addresses_tensor equivalence with addresses()."""

import random

import pytest
import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.utils.seeds import seed_everything


def test_addresses_tensor_bit_exact_cpu() -> None:
    """Test that addresses_tensor is bit-exact vs addresses() on CPU."""
    seed_everything(42)

    cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=4,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)
    device = torch.device("cpu")

    # Test with multiple random hx values
    random.seed(42)
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(20)]

    for hx in test_hx:
        # Reference: list-based addresses()
        ref_list = addresser.addresses(hx)
        ref = torch.tensor(ref_list, dtype=torch.long, device=device)

        # Vectorized: addresses_tensor()
        vec = addresser.addresses_tensor(hx, device)

        # Assert bit-exact equality
        assert torch.equal(vec, ref), (
            f"addresses_tensor does not match addresses() for hx={hx}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_addresses_tensor_bit_exact_cuda() -> None:
    """Test that addresses_tensor is bit-exact vs addresses() on CUDA."""
    seed_everything(42)

    cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=4,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)
    device = torch.device("cuda")

    # Test with multiple random hx values
    random.seed(42)
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(20)]

    for hx in test_hx:
        # Reference: list-based addresses() (always computed on CPU)
        ref_list = addresser.addresses(hx)
        ref = torch.tensor(ref_list, dtype=torch.long, device=device)

        # Vectorized: addresses_tensor() on CUDA
        vec = addresser.addresses_tensor(hx, device)

        # Assert bit-exact equality
        assert torch.equal(vec, ref), (
            f"addresses_tensor does not match addresses() for hx={hx} on CUDA"
        )
