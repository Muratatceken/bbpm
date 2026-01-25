"""Tests for batch addressing correctness and equivalence."""

import random

import pytest
import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.addressing.prp import u64_to_i64
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.utils.seeds import seed_everything


def test_addresses_batch_vs_scalar_cpu() -> None:
    """Test that addresses_batch matches addresses() for each key (CPU)."""
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
    
    # Generate random hx values including values >= 2^63 (to test uint64 handling)
    random.seed(42)
    test_hx = [
        random.randint(0, 2**63 - 1) for _ in range(10)
    ] + [
        random.randint(2**63, 2**64 - 1) for _ in range(10)
    ]  # Mix of small and large values
    
    for hx in test_hx:
        # Reference: scalar addresses()
        ref_list = addresser.addresses(hx)
        ref = torch.tensor(ref_list, dtype=torch.long, device=device)
        
        # Batch: addresses_batch([hx])
        # Convert uint64 to int64 representation for PyTorch
        hx_i64 = u64_to_i64(hx)
        hx_tensor = torch.tensor([hx_i64], dtype=torch.long, device=device)
        batch_result = addresser.addresses_batch(hx_tensor, device)
        
        # Assert bit-exact equality
        assert torch.equal(batch_result[0], ref), (
            f"addresses_batch does not match addresses() for hx={hx}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_addresses_batch_vs_scalar_cuda() -> None:
    """Test that addresses_batch matches addresses() for each key (CUDA)."""
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
    
    # Generate random hx values including values >= 2^63
    random.seed(42)
    test_hx = [
        random.randint(0, 2**63 - 1) for _ in range(10)
    ] + [
        random.randint(2**63, 2**64 - 1) for _ in range(10)
    ]
    
    for hx in test_hx:
        # Reference: scalar addresses() (always computed on CPU)
        ref_list = addresser.addresses(hx)
        ref = torch.tensor(ref_list, dtype=torch.long, device=device)
        
        # Batch: addresses_batch([hx]) on CUDA
        # Convert uint64 to int64 representation for PyTorch
        hx_i64 = u64_to_i64(hx)
        hx_tensor = torch.tensor([hx_i64], dtype=torch.long, device=device)
        batch_result = addresser.addresses_batch(hx_tensor, device)
        
        # Assert bit-exact equality
        assert torch.equal(batch_result[0], ref), (
            f"addresses_batch does not match addresses() for hx={hx} on CUDA"
        )


def test_addresses_batch_uniqueness() -> None:
    """Test that offsets are unique within each hash family."""
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
    
    # Test with multiple random keys
    random.seed(42)
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(20)]
    
    # Convert uint64 to int64 representation for PyTorch
    test_hx_i64 = [u64_to_i64(hx) for hx in test_hx]
    hx_tensor = torch.tensor(test_hx_i64, dtype=torch.long, device=device)
    batch_addrs = addresser.addresses_batch(hx_tensor, device)  # [T, H*K]
    
    T = batch_addrs.shape[0]
    H = cfg.H
    K = cfg.K
    
    for t in range(T):
        # Reshape row to [H, K]
        addrs_row = batch_addrs[t].reshape(H, K)  # [H, K]
        
        # For each hash family h, check that offsets are unique
        for h in range(H):
            # Extract offsets for this hash family (addr % block_size)
            offsets = addrs_row[h] % cfg.block_size  # [K]
            unique_offsets = len(torch.unique(offsets))
            
            # Should have K unique offsets (PRP guarantees uniqueness)
            assert unique_offsets == K, (
                f"Offsets not unique for hx={test_hx_i64[t]}, h={h}: "
                f"expected {K} unique, got {unique_offsets}"
            )


def test_write_read_batch_equivalence() -> None:
    """Test that write_batch/read_batch matches write/read."""
    seed_everything(42)
    
    # Memory configuration
    mem_cfg = MemoryConfig(
        num_blocks=10,
        block_size=256,
        key_dim=64,
        K=32,
        H=4,
        dtype="float32",
        device="cpu",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )
    
    # Generate test data
    random.seed(42)
    T = 20
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(T)]
    test_values = torch.randn(T, 64)
    
    # Method 1: Scalar write/read
    mem1 = BBPMMemory(mem_cfg)
    for i, hx in enumerate(test_hx):
        mem1.write(hx, test_values[i])
    scalar_results = []
    for hx in test_hx:
        r = mem1.read(hx)
        scalar_results.append(r)
    scalar_tensor = torch.stack(scalar_results)  # [T, d]
    
    # Method 2: Batch write/read
    mem2 = BBPMMemory(mem_cfg)
    # Convert uint64 to int64 representation for PyTorch
    test_hx_i64 = [u64_to_i64(hx) for hx in test_hx]
    hx_tensor = torch.tensor(test_hx_i64, dtype=torch.long)
    mem2.write_batch(hx_tensor, test_values)
    batch_tensor = mem2.read_batch(hx_tensor)  # [T, d]
    
    # Assert equivalence (should be very close, exact if no collisions)
    assert torch.allclose(scalar_tensor, batch_tensor, atol=1e-5), (
        "write_batch/read_batch does not match write/read"
    )


def test_addresses_batch_multiple_keys() -> None:
    """Test addresses_batch with multiple keys in one call."""
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
    
    # Generate multiple keys
    random.seed(42)
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(50)]
    # Convert uint64 to int64 representation for PyTorch
    test_hx_i64 = [u64_to_i64(hx) for hx in test_hx]
    hx_tensor = torch.tensor(test_hx_i64, dtype=torch.long, device=device)
    
    # Batch computation
    batch_addrs = addresser.addresses_batch(hx_tensor, device)  # [50, H*K]
    
    # Verify shape
    assert batch_addrs.shape == (50, cfg.H * cfg.K), (
        f"Expected shape [50, {cfg.H * cfg.K}], got {batch_addrs.shape}"
    )
    
    # Verify each row matches scalar addresses()
    for i, hx in enumerate(test_hx):
        ref_list = addresser.addresses(hx)
        ref = torch.tensor(ref_list, dtype=torch.long, device=device)
        assert torch.equal(batch_addrs[i], ref), (
            f"addresses_batch row {i} does not match addresses() for hx={hx}"
        )
