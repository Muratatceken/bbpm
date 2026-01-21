"""Quick start guide for BBPM with theory-compatible addressing.

Demonstrates:
1. Basic usage with automatic BBPMAddressing (via block_size)
2. Explicit BBPMAddressing usage
3. Unity gain behavior
4. Invalid block size handling with helpful suggestions
"""

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, BBPMAddressing


def example_1_basic_usage():
    """Example 1: Simplest usage - let BBPMAddressing be created automatically."""
    print("=" * 60)
    print("Example 1: Basic Usage with Automatic BBPMAddressing")
    print("=" * 60)
    
    # Configuration
    D = 1_000_000  # Total memory slots
    d = 64         # Value dimension
    K = 50         # Active slots per item per hash
    H = 1          # Number of hash functions
    block_size = 1024  # Valid: power of 2, even n_bits (2^10)
    
    # Create memory - block_size automatically creates BBPMAddressing
    memory = BBPMMemoryFloat(
        D=D,
        d=d,
        K=K,
        H=H,
        block_size=block_size,  # This creates BBPMAddressing internally
        seed=42,
        device="cpu"
    )
    print(f"Created memory: D={D:,}, d={d}, K={K}, block_size={block_size}")
    print()
    
    # Write some vectors
    keys = torch.arange(100, dtype=torch.int64)
    values = torch.randn(100, d)
    values = F.normalize(values, p=2, dim=1)  # Normalize to unit vectors
    
    print("Writing 100 vectors...")
    memory.write(keys, values)
    print("Write complete")
    print()
    
    # Read back
    print("Reading back...")
    retrieved = memory.read(keys)
    print("Read complete")
    print()
    
    # Check quality (unity gain with mean pooling)
    cosine_sim = F.cosine_similarity(retrieved, values, dim=1)
    print(f"Results:")
    print(f"  Mean cosine similarity: {cosine_sim.mean().item():.4f}")
    print(f"  Min cosine similarity:  {cosine_sim.min().item():.4f}")
    print(f"  Max cosine similarity:  {cosine_sim.max().item():.4f}")
    print()
    print("✓ Unity gain: write v → read ≈ v (in low load regime)")
    print()


def example_2_explicit_addressing():
    """Example 2: Explicit BBPMAddressing creation for more control."""
    print("=" * 60)
    print("Example 2: Explicit BBPMAddressing Creation")
    print("=" * 60)
    
    D = 500_000
    d = 32
    K = 30
    H = 2  # Multi-hash for better error correction
    block_size = 4096  # 2^12 (even n_bits)
    
    # Create addressing explicitly
    addressing = BBPMAddressing(
        D=D,
        block_size=block_size,
        seed=42,
        num_hashes=H,
        K=K
    )
    print(f"Created BBPMAddressing: block_size={block_size}, H={H}, K={K}")
    
    # Create memory with explicit addressing
    memory = BBPMMemoryFloat(
        D=D,
        d=d,
        K=K,
        H=H,
        hash_fn=addressing,  # Use explicit addressing
        device="cpu"
    )
    print(f"Created memory with explicit addressing")
    print()
    
    # Use the memory
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    values = torch.randn(5, d)
    values = F.normalize(values, p=2, dim=1)
    
    memory.write(keys, values)
    retrieved = memory.read(keys)
    
    cosine_sim = F.cosine_similarity(retrieved, values, dim=1)
    print(f"Mean cosine similarity: {cosine_sim.mean().item():.4f}")
    print()


def example_3_unity_gain():
    """Example 3: Demonstrate unity gain behavior."""
    print("=" * 60)
    print("Example 3: Unity Gain Verification")
    print("=" * 60)
    
    D = 10_000_000  # Very large memory
    d = 64
    K = 50
    H = 1
    block_size = 1024
    
    # Create with default write_scale="unit" (unity gain)
    memory = BBPMMemoryFloat(
        D=D, d=d, K=K, H=H,
        block_size=block_size,
        write_scale="unit",  # Default: unity gain
        seed=42
    )
    
    # Write a single vector
    key = torch.tensor([42], dtype=torch.int64)
    original_value = torch.ones(1, d) * 3.0
    original_value = F.normalize(original_value, p=2, dim=1)
    
    memory.write(key, original_value)
    retrieved = memory.read(key)
    
    # Check preservation
    cosine = F.cosine_similarity(retrieved, original_value, dim=1).item()
    magnitude = retrieved.norm(p=2, dim=1).item()
    
    print(f"Original value magnitude: {original_value.norm(p=2, dim=1).item():.4f}")
    print(f"Retrieved value magnitude: {magnitude:.4f}")
    print(f"Cosine similarity: {cosine:.4f}")
    print()
    
    if cosine > 0.99 and magnitude > 0.9:
        print("✓ Unity gain confirmed: signal preserved with mean pooling")
    else:
        print("⚠ Signal may be attenuated (check load ratio)")
    print()


def example_4_error_handling():
    """Example 4: See helpful error messages for invalid block sizes."""
    print("=" * 60)
    print("Example 4: Invalid Block Size Handling (with suggestions)")
    print("=" * 60)
    
    D = 100_000
    
    # Try invalid block size (not power of 2)
    print("Attempting invalid block_size=5000 (not power of 2)...")
    try:
        memory = BBPMMemoryFloat(
            D=D, d=64, K=50, H=1,
            block_size=5000,  # Invalid: not power of 2
            seed=42
        )
    except ValueError as e:
        print(f"Error (as expected): {e}")
        print()
    
    # Try invalid block size (odd n_bits)
    print("Attempting invalid block_size=512 (odd n_bits=9)...")
    try:
        memory = BBPMMemoryFloat(
            D=D, d=64, K=50, H=1,
            block_size=512,  # Invalid: 2^9 (odd)
            seed=42
        )
    except ValueError as e:
        print(f"Error (as expected): {e}")
        print()
    
    # Show valid suggestions
    print("Valid block sizes you can use:")
    suggestions = BBPMAddressing.suggest_valid_block_sizes(D)
    for i, size in enumerate(suggestions[:5], 1):
        n_bits = int(torch.log2(torch.tensor(size)).item())
        print(f"  {i}. block_size={size:,} (2^{n_bits}, n_bits={n_bits} is even)")
    print()


def example_5_comparison_unit_vs_scaled():
    """Example 5: Compare unit vs 1/sqrt(KH) scaling."""
    print("=" * 60)
    print("Example 5: Unit vs Scaled Write Scaling")
    print("=" * 60)
    
    D = 1_000_000
    d = 32
    K = 50
    H = 1
    block_size = 1024
    
    key = torch.tensor([1], dtype=torch.int64)
    value = torch.ones(1, d) * 5.0
    value = F.normalize(value, p=2, dim=1)
    
    # With unit scaling (default, recommended)
    mem_unit = BBPMMemoryFloat(
        D=D, d=d, K=K, H=H,
        block_size=block_size,
        write_scale="unit",  # Unity gain
        seed=42
    )
    mem_unit.write(key, value)
    retrieved_unit = mem_unit.read(key)
    
    # With 1/sqrt(KH) scaling
    mem_scaled = BBPMMemoryFloat(
        D=D, d=d, K=K, H=H,
        block_size=block_size,
        write_scale="1/sqrt(KH)",  # Attenuates signal
        seed=42
    )
    mem_scaled.write(key, value)
    retrieved_scaled = mem_scaled.read(key)
    
    print(f"Original magnitude: {value.norm(p=2, dim=1).item():.4f}")
    print(f"Unit scaling retrieved: {retrieved_unit.norm(p=2, dim=1).item():.4f} (preserved)")
    print(f"Scaled retrieved: {retrieved_scaled.norm(p=2, dim=1).item():.4f} (attenuated)")
    print()
    print("Note: Unit scaling preserves signal, scaled version reduces it by 1/sqrt(KH)")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("BBPM Theory-Compatible Implementation - Usage Examples")
    print("=" * 60)
    print()
    
    example_1_basic_usage()
    example_2_explicit_addressing()
    example_3_unity_gain()
    example_4_error_handling()
    example_5_comparison_unit_vs_scaled()
    
    print("=" * 60)
    print("All examples complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Use block_size parameter to automatically create BBPMAddressing")
    print("  2. Default write_scale='unit' provides unity gain (recommended)")
    print("  3. Mean pooling preserves signal while noise cancels out")
    print("  4. Invalid block_size will show helpful suggestions")
    print()


if __name__ == "__main__":
    main()
