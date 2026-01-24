"""Tests for PRP bijection property."""

import random

import pytest

from bbpm.addressing.prp import FeistelPRP


def test_prp_bijection() -> None:
    """Test that PRP is a bijection (permutation property)."""
    # Create FeistelPRP for nbits=12 (L=4096), rounds>=6
    nbits = 12
    L = 2**nbits  # 4096
    prp = FeistelPRP(nbits=nbits, rounds=6, master_key=42)

    # Fixed key for testing
    key = 12345

    # Compute permute(x) for all x in [0, L)
    outputs = []
    for x in range(L):
        y = prp.permute(x, key)
        outputs.append(y)
        # Assert output is in range
        assert 0 <= y < L, f"Output {y} out of range [0, {L})"

    # Assert all outputs unique (bijection property)
    assert len(outputs) == len(set(outputs)), "PRP outputs are not unique"
    assert len(set(outputs)) == L, "PRP does not cover full domain"


def test_prp_inverse() -> None:
    """Test that PRP inverse correctly recovers original values."""
    nbits = 12
    L = 2**nbits
    prp = FeistelPRP(nbits=nbits, rounds=6, master_key=42)
    key = 12345

    # Test on random samples
    random.seed(42)
    test_samples = random.sample(range(L), min(100, L))

    for x in test_samples:
        y = prp.permute(x, key)
        x_recovered = prp.invert(y, key)
        assert x == x_recovered, f"Inverse failed: {x} -> {y} -> {x_recovered}"
