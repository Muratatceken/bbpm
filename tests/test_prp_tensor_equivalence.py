"""Tests for PRP tensor equivalence and correctness."""

import random

import pytest
import torch

from bbpm.addressing.prp import FeistelPRP
from bbpm.utils.seeds import seed_everything


def test_prp_tensor_bit_exact() -> None:
    """Test that permute_tensor produces bit-exact results matching permute."""
    seed_everything(42)

    # Test multiple nbits values
    nbits_values = [8, 12, 16]

    for nbits in nbits_values:
        prp = FeistelPRP(nbits=nbits, rounds=6, master_key=42)
        domain_size = 1 << nbits

        # Test with random keys and x values
        random.seed(42)
        test_keys = [random.randint(0, 2**64 - 1) for _ in range(5)]
        test_x_lists = [
            random.sample(range(domain_size), min(100, domain_size))
            for _ in range(3)
        ]

        for key in test_keys:
            for xs in test_x_lists:
                # Compute scalar outputs
                scalar_results = [prp.permute(x, key) for x in xs]

                # Compute tensor outputs on CPU
                device = torch.device("cpu")
                x_tensor = torch.tensor(xs, dtype=torch.long, device=device)
                tensor_results = prp.permute_tensor(x_tensor, key)

                # Convert scalar results to tensor for comparison
                scalar_tensor = torch.tensor(scalar_results, dtype=torch.long, device=device)

                # Assert exact equality (bit-exact, not tolerance)
                assert torch.equal(tensor_results, scalar_tensor), (
                    f"permute_tensor does not match permute for nbits={nbits}, "
                    f"key={key}, xs={xs[:5]}..."
                )

                # If CUDA available, also test on CUDA
                if torch.cuda.is_available():
                    device_cuda = torch.device("cuda")
                    x_tensor_cuda = torch.tensor(xs, dtype=torch.long, device=device_cuda)
                    tensor_results_cuda = prp.permute_tensor(x_tensor_cuda, key)

                    # Should match CPU results exactly
                    assert torch.equal(
                        tensor_results_cuda.cpu(), scalar_tensor
                    ), (
                        f"permute_tensor on CUDA does not match CPU for nbits={nbits}, "
                        f"key={key}"
                    )


def test_addresses_tensor_uses_permute_tensor(monkeypatch) -> None:
    """Test that addresses_tensor calls permute_tensor (no Python loop over k)."""
    from bbpm.addressing.block_address import AddressConfig, BlockAddress

    seed_everything(42)

    cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=4,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)

    # Track calls to permute_tensor
    permute_tensor_calls = []

    original_permute_tensor = FeistelPRP.permute_tensor

    def tracked_permute_tensor(self, x, key):
        permute_tensor_calls.append((key, x.device))
        return original_permute_tensor(self, x, key)

    monkeypatch.setattr(FeistelPRP, "permute_tensor", tracked_permute_tensor)

    # Call addresses_tensor
    hx = 12345
    device = torch.device("cpu")
    _ = addresser.addresses_tensor(hx, device)

    # Assert permute_tensor was called H times (once per hash family)
    assert len(permute_tensor_calls) == cfg.H, (
        f"Expected permute_tensor to be called {cfg.H} times (once per hash family), "
        f"got {len(permute_tensor_calls)}"
    )


def test_prp_validation_raises_valueerror() -> None:
    """Test that invalid inputs raise ValueError (not AssertionError)."""
    seed_everything(42)

    # Test invalid nbits
    with pytest.raises(ValueError, match="nbits must be positive"):
        FeistelPRP(nbits=0, rounds=6, master_key=42)

    with pytest.raises(ValueError, match="nbits must be positive"):
        FeistelPRP(nbits=-1, rounds=6, master_key=42)

    # Test invalid rounds
    with pytest.raises(ValueError, match="rounds should be >= 6"):
        FeistelPRP(nbits=8, rounds=5, master_key=42)

    # Test invalid master_key
    with pytest.raises(ValueError, match="master_key must be uint64"):
        FeistelPRP(nbits=8, rounds=6, master_key=2**64)

    with pytest.raises(ValueError, match="master_key must be uint64"):
        FeistelPRP(nbits=8, rounds=6, master_key=-1)

    # Test valid PRP
    prp = FeistelPRP(nbits=8, rounds=6, master_key=42)

    # Test invalid x in permute
    with pytest.raises(ValueError, match="x must be in"):
        prp.permute(256, key=123)  # 256 >= 2^8

    with pytest.raises(ValueError, match="x must be in"):
        prp.permute(-1, key=123)

    # Test invalid key in permute
    with pytest.raises(ValueError, match="key must be uint64"):
        prp.permute(100, key=2**64)

    # Test invalid y in invert
    with pytest.raises(ValueError, match="y must be in"):
        prp.invert(256, key=123)

    # Test invalid key in invert
    with pytest.raises(ValueError, match="key must be uint64"):
        prp.invert(100, key=2**64)
