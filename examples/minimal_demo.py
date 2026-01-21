"""Minimal demo of BBPM with PRP-based addressing.

Demonstrates basic usage of BBPMAddressing for storing and retrieving vectors.
"""

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, BBPMAddressing


def main():
    """Run minimal demo."""
    print("BBPM Minimal Demo")
    print("=" * 50)
    
    # Configuration
    D = 1_000_000  # Total memory slots
    d = 64  # Value dimension
    K = 50  # Active slots per item per hash
    H = 1  # Number of hash functions
    block_size = 1024  # Block size (power of 2, even n_bits)
    N = 1000  # Number of items to store
    
    print(f"Memory size: {D:,} slots")
    print(f"Block size: {block_size}")
    print(f"Value dimension: {d}")
    print(f"K (sparsity): {K}")
    print(f"H (hashes): {H}")
    print(f"Storing {N:,} items")
    print()
    
    # Create BBPMAddressing
    addressing = BBPMAddressing(D, block_size, seed=42, num_hashes=H, K=K)
    print("Created BBPMAddressing (PRP-based)")
    
    # Create memory with PRP-based addressing
    memory = BBPMMemoryFloat(
        D=D,
        d=d,
        K=K,
        H=H,
        block_size=block_size,  # This creates BBPMAddressing internally
        seed=42,
        device="cpu"
    )
    print("Created BBPMMemoryFloat with BBPMAddressing")
    print()
    
    # Generate random keys and values
    keys = torch.arange(N, dtype=torch.int64)
    values = torch.randn(N, d)
    values = F.normalize(values, p=2, dim=1)  # Normalize to unit vectors
    
    print(f"Writing {N:,} vectors...")
    memory.write(keys, values)
    print("Write complete")
    print()
    
    # Read back
    print("Reading back...")
    retrieved = memory.read(keys)
    print("Read complete")
    print()
    
    # Compute metrics
    cosine_similarity = F.cosine_similarity(retrieved, values, dim=1)
    mean_cosine = cosine_similarity.mean().item()
    min_cosine = cosine_similarity.min().item()
    max_cosine = cosine_similarity.max().item()
    
    mse = F.mse_loss(retrieved, values).item()
    
    print("Results:")
    print(f"  Mean cosine similarity: {mean_cosine:.4f}")
    print(f"  Min cosine similarity:  {min_cosine:.4f}")
    print(f"  Max cosine similarity:  {max_cosine:.4f}")
    print(f"  MSE: {mse:.6f}")
    print()
    
    # Memory diagnostics
    diag = memory.memory_diagnostics()
    print("Memory diagnostics:")
    print(f"  Nonzero slots: {diag['nonzero_slots']:,}")
    print(f"  Max count: {diag['max_count']}")
    print(f"  Mean count (nonzero): {diag['mean_count_nonzero']:.2f}")
    print()
    
    print("Demo complete!")


if __name__ == "__main__":
    main()
