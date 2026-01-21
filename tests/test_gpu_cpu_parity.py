"""Test GPU/CPU parity."""

import pytest
import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, set_global_seed


@pytest.fixture
def seed():
    """Fixed seed for tests."""
    return 42


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_cpu_parity(seed):
    """Test that GPU and CPU produce same results."""
    set_global_seed(seed)

    D = 10000
    d = 64
    K = 32
    H = 1
    N = 100

    # Generate test data
    keys = torch.arange(N)
    values = torch.randn(N, d)
    values = F.normalize(values, p=2, dim=1)

    # CPU version
    memory_cpu = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cpu")
    memory_cpu.clear()
    memory_cpu.write(keys, values)
    retrieved_cpu = memory_cpu.read(keys)

    # GPU version
    memory_gpu = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device="cuda")
    memory_gpu.clear()
    memory_gpu.write(keys.cuda(), values.cuda())
    retrieved_gpu = memory_gpu.read(keys.cuda())

    # Compare results
    retrieved_gpu_cpu = retrieved_gpu.cpu()
    cos_sim = F.cosine_similarity(retrieved_cpu, retrieved_gpu_cpu, dim=1).mean().item()

    assert cos_sim > 0.99, f"GPU and CPU should produce similar results, got {cos_sim:.4f}"
