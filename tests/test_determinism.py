"""Tests for determinism across runs."""

import pytest
import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.memory.interfaces import MemoryConfig
from bbpm.utils.seeds import seed_everything


def test_block_address_determinism() -> None:
    """Test that BlockAddress.addresses is deterministic."""
    seed_everything(42)

    addr_cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=2,
        master_seed=42,
    )
    addresser = BlockAddress(addr_cfg)

    hx = 12345

    # Get addresses multiple times
    addrs1 = addresser.addresses(hx)
    addrs2 = addresser.addresses(hx)
    addrs3 = addresser.addresses(hx)

    # All should be identical
    assert addrs1 == addrs2 == addrs3, "Addresses not deterministic"


def test_memory_determinism() -> None:
    """Test that BBPMMemory read/write is deterministic."""
    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=10,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="float32",
        accumulate="native",
        output_dtype="float32",
        device="cpu",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Write sequence
    hx1, hx2, hx3 = 100, 200, 300
    v1 = torch.randn(cfg.key_dim)
    v2 = torch.randn(cfg.key_dim)
    v3 = torch.randn(cfg.key_dim)

    mem.write(hx1, v1)
    mem.write(hx2, v2)
    mem.write(hx3, v3)

    # Read back
    r1 = mem.read(hx1)
    r2 = mem.read(hx2)
    r3 = mem.read(hx3)

    # Read again - should be identical
    r1_repeat = mem.read(hx1)
    r2_repeat = mem.read(hx2)
    r3_repeat = mem.read(hx3)

    # Check determinism
    assert torch.allclose(r1, r1_repeat), "Read not deterministic"
    assert torch.allclose(r2, r2_repeat), "Read not deterministic"
    assert torch.allclose(r3, r3_repeat), "Read not deterministic"
