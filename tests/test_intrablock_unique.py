"""Tests for intra-block offset uniqueness."""

import random

import pytest

from bbpm.addressing.block_address import AddressConfig, BlockAddress


def test_intrablock_offsets_unique() -> None:
    """Test that intra-block offsets are unique for any key (no self-collisions)."""
    # Create AddressConfig with block_size=1024, K=64, H=4, B=128
    block_size = 1024
    K = 64
    H = 4
    B = 128
    master_seed = 42

    addr_cfg = AddressConfig(
        num_blocks=B,
        block_size=block_size,
        K=K,
        H=H,
        master_seed=master_seed,
    )
    addresser = BlockAddress(addr_cfg)

    # Test with random hx values
    random.seed(42)
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(10)]

    for hx in test_hx:
        # Get addresses grouped by hash family
        grouped = addresser.addresses_grouped(hx)

        # Should have H groups
        assert len(grouped) == H, f"Expected {H} groups, got {len(grouped)}"

        # For each h-group, check offsets are unique
        block_ids = addresser.block_ids(hx)
        for h_idx, addresses_h in enumerate(grouped):
            # Extract block_id and offsets
            block_id = block_ids[h_idx]
            offsets = [addr - block_id * block_size for addr in addresses_h]

            # Assert K unique offsets
            assert len(set(offsets)) == K, (
                f"Group {h_idx}: expected {K} unique offsets, got {len(set(offsets))}"
            )
            assert len(offsets) == K, (
                f"Group {h_idx}: expected {K} offsets, got {len(offsets)}"
            )

            # Assert all offsets in [0, block_size)
            for offset in offsets:
                assert 0 <= offset < block_size, (
                    f"Offset {offset} out of range [0, {block_size})"
                )

        # Assert all addresses in [0, B*block_size)
        all_addresses = addresser.addresses(hx)
        for addr in all_addresses:
            assert 0 <= addr < B * block_size, (
                f"Address {addr} out of range [0, {B * block_size})"
            )
