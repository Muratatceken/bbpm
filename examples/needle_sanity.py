"""Needle-in-a-haystack sanity check.

Demonstrates that BBPM retrieval quality is invariant to distance in sequence.
Inserts a key-value pair at the start, then adds N distractors, then retrieves.
"""

import torch
import torch.nn.functional as F

from bbpm import BBPMMemoryFloat, BBPMAddressing


def main():
    """Run needle-in-haystack demo."""
    print("BBPM Needle-in-Haystack Demo")
    print("=" * 50)
    
    # Configuration
    D = 1_000_000
    d = 64
    K = 50
    H = 1
    block_size = 1024
    
    print(f"Memory size: {D:,} slots")
    print(f"Block size: {block_size}")
    print(f"K (sparsity): {K}")
    print(f"H (hashes): {H}")
    print()
    
    # Test with different numbers of distractors
    distractor_counts = [100, 1000, 10000, 100000]
    
    results = []
    
    for N_distractors in distractor_counts:
        print(f"Testing with {N_distractors:,} distractors...")
        
        # Create fresh memory for each test
        memory = BBPMMemoryFloat(
            D=D,
            d=d,
            K=K,
            H=H,
            block_size=block_size,
            seed=42,
            device="cpu"
        )
        memory.clear()
        
        # Insert "needle" at key 0
        needle_key = torch.tensor([0], dtype=torch.int64)
        needle_value = torch.ones(1, d)
        needle_value = F.normalize(needle_value, p=2, dim=1)
        
        memory.write(needle_key, needle_value)
        
        # Add distractors
        if N_distractors > 0:
            distractor_keys = torch.arange(1, N_distractors + 1, dtype=torch.int64)
            distractor_values = torch.randn(N_distractors, d)
            distractor_values = F.normalize(distractor_values, p=2, dim=1)
            
            memory.write(distractor_keys, distractor_values)
        
        # Retrieve needle
        retrieved = memory.read(needle_key)
        
        # Compute cosine similarity
        cosine = F.cosine_similarity(retrieved, needle_value, dim=1).item()
        
        results.append((N_distractors, cosine))
        print(f"  Cosine similarity: {cosine:.4f}")
        print()
    
    print("Results Summary:")
    print("-" * 50)
    print(f"{'Distractors':<15} {'Cosine Similarity':<20}")
    print("-" * 50)
    for N, cosine in results:
        print(f"{N:>12,}     {cosine:>18.4f}")
    print()
    
    # Check if similarity is relatively stable (should be invariant to distance)
    cosines = [c for _, c in results]
    mean_cosine = sum(cosines) / len(cosines)
    std_cosine = (sum((c - mean_cosine) ** 2 for c in cosines) / len(cosines)) ** 0.5
    
    print(f"Mean cosine similarity: {mean_cosine:.4f}")
    print(f"Std deviation: {std_cosine:.4f}")
    print()
    
    if std_cosine < 0.1:
        print("✓ Retrieval quality is stable across different distances")
    else:
        print("⚠ Retrieval quality varies with distance (may indicate issues)")
    print()
    
    print("Demo complete!")


if __name__ == "__main__":
    main()
