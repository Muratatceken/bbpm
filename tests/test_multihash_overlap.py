"""Tests for multi-hash H overlap bounds."""

import pytest

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.metrics.occupancy import overlap_rate


def test_multihash_overlap_bounds() -> None:
    """Test that multi-hash H overlap is within expected random bounds."""
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

    # Fixed hx for testing
    hx = 12345

    # Get addresses grouped by hash family
    grouped = addresser.addresses_grouped(hx)

    # Extract offsets for each hash family (within their blocks)
    block_ids = addresser.block_ids(hx)
    offset_groups = []
    for h_idx, addresses_h in enumerate(grouped):
        block_id = block_ids[h_idx]
        offsets = [addr - block_id * block_size for addr in addresses_h]
        offset_groups.append(set(offsets))

    # Measure pairwise overlap
    # Expected overlap for random: ~ K^2 / block_size
    expected_overlap = (K * K) / block_size

    # Test all pairs
    for i in range(H):
        for j in range(i + 1, H):
            # Convert sets to lists for overlap_rate
            overlap = overlap_rate(list(offset_groups[i]), list(offset_groups[j]))

            # Allow loose bound: <= 5 * expected + 2
            threshold = 5 * expected_overlap + 2
            assert overlap <= threshold, (
                f"Overlap between groups {i} and {j} ({overlap}) "
                f"exceeds threshold ({threshold})"
            )
