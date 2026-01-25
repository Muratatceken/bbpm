#!/usr/bin/env python3
"""Quick test script to verify CUDA batch addressing works correctly."""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.addressing.prp import u64_to_i64
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.utils.seeds import seed_everything


def test_cuda_batch_addressing():
    """Test batch addressing on CUDA device."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system")
        print("   This test requires a CUDA-enabled GPU")
        return False
    
    print("=" * 80)
    print("CUDA Batch Addressing Test")
    print("=" * 80)
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    seed_everything(42)
    device = torch.device("cuda")
    
    # Setup
    cfg = AddressConfig(
        num_blocks=2**14,
        block_size=256,
        K=32,
        H=4,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)
    
    # Test data
    import random
    random.seed(42)
    test_hx = [random.randint(0, 2**63 - 1) for _ in range(100)]
    test_hx_i64 = [u64_to_i64(hx) for hx in test_hx]
    hx_tensor = torch.tensor(test_hx_i64, dtype=torch.long, device=device)
    
    print("Testing addresses_batch on CUDA...")
    # Test addresses_batch
    batch_addrs = addresser.addresses_batch(hx_tensor, device)
    print(f"  ✓ addresses_batch shape: {batch_addrs.shape}")
    
    # Verify correctness by comparing with CPU scalar version
    print("  Verifying correctness (comparing with CPU scalar version)...")
    all_match = True
    for i, hx in enumerate(test_hx[:10]):  # Check first 10
        ref_list = addresser.addresses(hx)
        ref = torch.tensor(ref_list, dtype=torch.long, device=device)
        if not torch.equal(batch_addrs[i], ref):
            print(f"  ❌ Mismatch for hx={hx} at index {i}")
            all_match = False
            break
    
    if all_match:
        print("  ✓ All addresses match scalar version!")
    
    # Test write_batch and read_batch
    print("\nTesting write_batch and read_batch on CUDA...")
    mem_cfg = MemoryConfig(
        num_blocks=2**14,
        block_size=256,
        key_dim=64,
        K=32,
        H=4,
        dtype="float32",
        device="cuda",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )
    mem = BBPMMemory(mem_cfg)
    
    values = torch.randn(100, 64, device=device)
    mem.write_batch(hx_tensor, values)
    print("  ✓ write_batch completed")
    
    results = mem.read_batch(hx_tensor)
    print(f"  ✓ read_batch completed, shape: {results.shape}")
    
    # Performance test
    print("\nPerformance test (warmup + 100 iterations)...")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(10):
        _ = addresser.addresses_batch(hx_tensor, device)
    torch.cuda.synchronize()
    
    # Timed
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = addresser.addresses_batch(hx_tensor, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / 100) * 1000
    throughput = len(test_hx) / (elapsed / 100)
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {throughput:,.0f} keys/sec")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ All CUDA tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = test_cuda_batch_addressing()
    sys.exit(0 if success else 1)
