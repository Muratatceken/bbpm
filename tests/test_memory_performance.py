"""Tests for memory performance and correctness guarantees."""

import pytest
import torch

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.memory.interfaces import MemoryConfig
from bbpm.utils.seeds import seed_everything


def test_addresses_tensor_shape_and_dtype() -> None:
    """Test that addresses_tensor returns correct shape and dtype."""
    seed_everything(42)

    cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=4,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)

    hx = 12345
    device = torch.device("cpu")
    addr_tensor = addresser.addresses_tensor(hx, device)

    # Check shape
    assert addr_tensor.shape == (cfg.H * cfg.K,), (
        f"Expected shape ({cfg.H * cfg.K},), got {addr_tensor.shape}"
    )

    # Check dtype
    assert addr_tensor.dtype == torch.long, (
        f"Expected dtype torch.long, got {addr_tensor.dtype}"
    )

    # Check device
    assert addr_tensor.device == device, (
        f"Expected device {device}, got {addr_tensor.device}"
    )

    # Check values are in valid range [0, B*L)
    max_addr = cfg.num_blocks * cfg.block_size
    assert torch.all(addr_tensor >= 0), "Addresses must be non-negative"
    assert torch.all(addr_tensor < max_addr), (
        f"Addresses must be < {max_addr}, got max {addr_tensor.max().item()}"
    )


def test_addresses_tensor_deterministic() -> None:
    """Test that addresses_tensor is deterministic."""
    seed_everything(42)

    cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=2,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)

    hx = 12345
    device = torch.device("cpu")

    addr1 = addresser.addresses_tensor(hx, device)
    addr2 = addresser.addresses_tensor(hx, device)

    assert torch.equal(addr1, addr2), "addresses_tensor must be deterministic"


def test_no_full_tensor_cast_float32() -> None:
    """Test that write() does not cast full memory tensor for float32."""
    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="float32",
        device="cpu",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Store original dtype
    original_dtype = mem.memory.dtype

    # Write a value
    hx = 12345
    v = torch.randn(cfg.key_dim)
    mem.write(hx, v)

    # Memory dtype should remain unchanged
    assert mem.memory.dtype == original_dtype, (
        f"Memory dtype changed from {original_dtype} to {mem.memory.dtype}"
    )

    # Memory should still be float32
    assert mem.memory.dtype == torch.float32, "Memory should remain float32"


def test_no_full_tensor_cast_bfloat16() -> None:
    """Test that write() does not cast full memory tensor for bfloat16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for bfloat16 test")

    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="bfloat16",
        accumulate="fast_inexact",
        device="cuda",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Store original dtype
    original_dtype = mem.memory.dtype

    # Write a value
    hx = 12345
    v = torch.randn(cfg.key_dim, device="cuda")
    mem.write(hx, v)

    # Memory dtype should remain unchanged
    assert mem.memory.dtype == original_dtype, (
        f"Memory dtype changed from {original_dtype} to {mem.memory.dtype}"
    )

    # Memory should still be bfloat16
    assert mem.memory.dtype == torch.bfloat16, "Memory should remain bfloat16"


def test_bfloat16_fast_inexact_works() -> None:
    """Test that bfloat16 fast_inexact mode works correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for bfloat16 test")

    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="bfloat16",
        accumulate="fast_inexact",
        output_dtype="float32",
        device="cuda",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Write and read
    hx = 12345
    v = torch.randn(cfg.key_dim, device="cuda")
    mem.write(hx, v)

    # Read should return float32 (output_dtype)
    result = mem.read(hx)
    assert result.dtype == torch.float32, (
        f"Read should return float32, got {result.dtype}"
    )
    assert result.shape == (cfg.key_dim,), (
        f"Read should return shape ({cfg.key_dim},), got {result.shape}"
    )


def test_bfloat16_invalid_accumulate_raises_error() -> None:
    """Test that bfloat16 with invalid accumulate mode raises error."""
    seed_everything(42)

    # Try to create config with bfloat16 and native accumulate
    with pytest.raises(ValueError, match="bfloat16 requires accumulate='fast_inexact'"):
        MemoryConfig(
            num_blocks=8,
            block_size=256,
            key_dim=32,
            K=16,
            H=1,
            dtype="bfloat16",
            accumulate="native",  # Invalid for bfloat16
            device="cpu",
            normalize_values="none",
            read_mode="raw_mean",
            master_seed=42,
        )


def test_read_output_dtype() -> None:
    """Test that read() returns correct output_dtype."""
    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="float32",
        output_dtype="float32",
        device="cpu",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Write a value
    hx = 12345
    v = torch.randn(cfg.key_dim)
    mem.write(hx, v)

    # Read should return float32
    result = mem.read(hx)
    assert result.dtype == torch.float32, (
        f"Read should return float32, got {result.dtype}"
    )


def test_addresses_tensor_matches_addresses() -> None:
    """Test that addresses_tensor returns same values as addresses()."""
    seed_everything(42)

    cfg = AddressConfig(
        num_blocks=10,
        block_size=256,
        K=32,
        H=2,
        master_seed=42,
    )
    addresser = BlockAddress(cfg)

    hx = 12345
    device = torch.device("cpu")

    # Get addresses as list
    addr_list = addresser.addresses(hx)

    # Get addresses as tensor
    addr_tensor = addresser.addresses_tensor(hx, device)

    # Convert list to tensor for comparison
    addr_list_tensor = torch.tensor(addr_list, dtype=torch.long, device=device)

    # Should be equal
    assert torch.equal(addr_tensor, addr_list_tensor), (
        "addresses_tensor() should return same values as addresses()"
    )


def test_addresses_tensor_vectorized() -> None:
    """Test that addresses_tensor matches addresses() for multiple random hx values."""
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
    import random
    random.seed(42)
    test_hx = [random.randint(0, 2**64 - 1) for _ in range(10)]

    for hx in test_hx:
        # Get addresses as list
        addr_list = addresser.addresses(hx)

        # Get addresses as tensor
        addr_tensor = addresser.addresses_tensor(hx, device)

        # Convert list to tensor for comparison
        addr_list_tensor = torch.tensor(addr_list, dtype=torch.long, device=device)

        # Should be equal
        assert torch.equal(addr_tensor, addr_list_tensor), (
            f"addresses_tensor() should return same values as addresses() for hx={hx}"
        )


def test_cuda_device_validation() -> None:
    """Test that CUDA device validation raises error when CUDA unavailable."""
    import torch

    # Only run this test if CUDA is NOT available
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, skipping CUDA validation test")

    seed_everything(42)

    # Should raise ValueError when CUDA requested but unavailable
    with pytest.raises(ValueError, match="CUDA device requested but CUDA is not available"):
        MemoryConfig(
            num_blocks=8,
            block_size=256,
            key_dim=32,
            K=16,
            H=1,
            dtype="float32",
            device="cuda",  # Request CUDA but it's not available
            normalize_values="none",
            read_mode="raw_mean",
            master_seed=42,
        )


def test_metrics_float32_stability() -> None:
    """Test that metrics functions work correctly with bfloat16 inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for bfloat16 test")

    from bbpm.metrics.retrieval import cosine_similarity, mse, snr_proxy

    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="bfloat16",
        accumulate="fast_inexact",
        output_dtype="bfloat16",  # Output as bfloat16
        device="cuda",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Write and read
    hx = 12345
    v = torch.randn(cfg.key_dim, device="cuda")
    mem.write(hx, v)

    # Read should return bfloat16 (output_dtype)
    result = mem.read(hx)
    assert result.dtype == torch.bfloat16, f"Read should return bfloat16, got {result.dtype}"

    # Convert v to bfloat16 for comparison
    v_bf16 = v.to(torch.bfloat16)

    # Metrics should work with bfloat16 inputs (cast internally to float32)
    cos = cosine_similarity(v_bf16, result)
    mse_val = mse(v_bf16, result)
    snr = snr_proxy(v_bf16, result)

    # Verify metrics return floats and are reasonable
    assert isinstance(cos, float), "cosine_similarity should return float"
    assert isinstance(mse_val, float), "mse should return float"
    assert isinstance(snr, float), "snr_proxy should return float"

    # In low-load regime, cosine should be high
    assert cos > 0.8, f"Cosine similarity too low: {cos}"


def test_count_normalized_correctness() -> None:
    """Test that count_normalized mode produces correct results."""
    seed_everything(42)

    # Test with normalize_values="none"
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
        read_mode="count_normalized",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Write single vector in empty memory
    hx = 12345
    v = torch.randn(cfg.key_dim)
    mem.write(hx, v)

    # Read back
    result = mem.read(hx)

    # In low-load regime with count_normalized, result should be close to v
    from bbpm.metrics.retrieval import cosine_similarity, mse

    cos = cosine_similarity(v, result)
    mse_val = mse(v, result)

    assert cos > 0.99, f"Cosine similarity too low: {cos}"
    assert mse_val < 1e-6, f"MSE too high: {mse_val}"

    # Test with normalize_values="l2"
    cfg_l2 = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="float32",
        accumulate="native",
        output_dtype="float32",
        device="cpu",
        normalize_values="l2",
        read_mode="count_normalized",
        master_seed=42,
    )

    mem_l2 = BBPMMemory(cfg_l2)

    # Write and read
    v_l2 = torch.randn(cfg_l2.key_dim)
    mem_l2.write(hx, v_l2)
    result_l2 = mem_l2.read(hx)

    # Result should be close to normalized v
    v_l2_norm = v_l2 / (torch.norm(v_l2, p=2) + 1e-8)
    cos_l2 = cosine_similarity(v_l2_norm, result_l2)

    assert cos_l2 > 0.99, f"Cosine similarity too low with l2 normalization: {cos_l2}"


def test_count_normalized_unseen_key() -> None:
    """Test that reading unseen key returns near-zero vector (not NaN)."""
    seed_everything(42)

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
        read_mode="count_normalized",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)

    # Write to one key
    hx1 = 12345
    v1 = torch.randn(cfg.key_dim)
    mem.write(hx1, v1)

    # Read from different unseen key
    hx2 = 67890
    result = mem.read(hx2)

    # Result should be near-zero (not NaN)
    assert not torch.isnan(result).any(), "Result should not contain NaN"
    assert torch.allclose(
        result, torch.zeros_like(result), atol=1e-6
    ), "Unseen key should return near-zero vector"


def test_no_full_tensor_cast_monkeypatch(monkeypatch) -> None:
    """Test that write() does not create full-memory float32 tensor for bf16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for bfloat16 test")

    seed_everything(42)

    cfg = MemoryConfig(
        num_blocks=8,
        block_size=256,
        key_dim=32,
        K=16,
        H=1,
        dtype="bfloat16",
        accumulate="fast_inexact",
        device="cuda",
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )

    mem = BBPMMemory(cfg)
    D = cfg.num_blocks * cfg.block_size
    d = cfg.key_dim

    # Track calls to torch.Tensor.to
    full_cast_detected = []
    original_to = torch.Tensor.to

    def tracked_to(self, *args, **kwargs):
        # Check if this is a full-memory cast: shape [D, d], dtype bfloat16 -> float32
        if (
            self.shape == (D, d)
            and self.dtype == torch.bfloat16
            and args
            and args[0] == torch.float32
        ):
            full_cast_detected.append(True)
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)

    # Write a value
    hx = 12345
    v = torch.randn(cfg.key_dim, device="cuda")
    mem.write(hx, v)

    # No full-memory cast should have occurred
    assert len(full_cast_detected) == 0, (
        f"Full-memory cast detected {len(full_cast_detected)} times. "
        "This violates O(H*K*d) complexity requirement."
    )


def test_write_device_mismatch_auto_move() -> None:
    """Test that write() automatically moves tensor to correct device."""
    seed_everything(42)

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

    # Create tensor on CPU (same as memory device)
    hx = 12345
    v = torch.randn(cfg.key_dim, device="cpu")

    # Write should succeed
    mem.write(hx, v)

    # Read back to verify
    result = mem.read(hx)
    assert result.shape == (cfg.key_dim,), "Read should return correct shape"

    # If CUDA is available, test cross-device move
    if torch.cuda.is_available():
        cfg_cuda = MemoryConfig(
            num_blocks=8,
            block_size=256,
            key_dim=32,
            K=16,
            H=1,
            dtype="float32",
            accumulate="native",
            output_dtype="float32",
            device="cuda",
            normalize_values="none",
            read_mode="raw_mean",
            master_seed=42,
        )

        mem_cuda = BBPMMemory(cfg_cuda)

        # Create tensor on CPU, write to CUDA memory
        v_cpu = torch.randn(cfg_cuda.key_dim, device="cpu")
        mem_cuda.write(hx, v_cpu)  # Should auto-move to CUDA

        # Verify write succeeded
        result_cuda = mem_cuda.read(hx)
        assert result_cuda.device.type == "cuda", "Result should be on CUDA"
        assert result_cuda.shape == (cfg_cuda.key_dim,), "Read should return correct shape"
