"""Tests for memory read/write correctness."""

import random

import pytest
import torch

from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.memory.interfaces import MemoryConfig
from bbpm.metrics.retrieval import cosine_similarity
from bbpm.utils.seeds import seed_everything


def test_memory_write_read_identity() -> None:
    """Test that write followed by read recovers original values."""
    seed_everything(42)

    # Small sizes for fast testing
    cfg = MemoryConfig(
        num_blocks=8,
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

    # Write random vectors for many items
    num_items = 50
    random.seed(42)
    torch.manual_seed(42)

    hx_list = [random.randint(0, 2**64 - 1) for _ in range(num_items)]
    values = []

    for hx in hx_list:
        v = torch.randn(cfg.key_dim)
        values.append(v)
        mem.write(hx, v)

    # Read back immediately
    retrieved = []
    for hx in hx_list:
        r = mem.read(hx)
        retrieved.append(r)

    # Ensure self-retrieval cosine similarity is high in low-load regime
    cosines = []
    for v, r in zip(values, retrieved):
        cos = cosine_similarity(v, r)
        cosines.append(cos)

    # In low-load regime, cosine should be high (>0.8)
    mean_cosine = sum(cosines) / len(cosines)
    assert mean_cosine > 0.8, f"Mean cosine similarity too low: {mean_cosine}"


def test_memory_reset() -> None:
    """Test that reset() zeros memory."""
    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=4,
        block_size=128,
        key_dim=16,
        K=8,
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

    # Write some values
    hx = 12345
    v = torch.randn(cfg.key_dim)
    mem.write(hx, v)

    # Read should be non-zero
    r_before = mem.read(hx)
    assert not torch.allclose(
        r_before, torch.zeros_like(r_before)
    ), "Memory should contain values before reset"

    # Reset
    mem.reset()

    # Read should return zeros
    r_after = mem.read(hx)
    assert torch.allclose(
        r_after, torch.zeros_like(r_after)
    ), "Memory should be zero after reset"
