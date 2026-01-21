# Using the New Theory-Compatible BBPM Code

This guide shows how to use the updated BBPM implementation with theory-compatible addressing and fixed read operations.

## Quick Start

### Simplest Usage (Recommended)

```python
import torch
from bbpm import BBPMMemoryFloat

# Create memory with PRP-based addressing
memory = BBPMMemoryFloat(
    D=1_000_000,      # Total memory slots
    d=64,             # Value dimension
    K=50,             # Active slots per item per hash
    H=1,              # Number of hash functions
    block_size=1024,  # Block size (must be power of 2, even n_bits)
    seed=42
)

# Write vectors
keys = torch.arange(100, dtype=torch.int64)
values = torch.randn(100, 64)
memory.write(keys, values)

# Read back (mean pooling, no count debiasing)
retrieved = memory.read(keys)

# Check quality
import torch.nn.functional as F
cosine_sim = F.cosine_similarity(retrieved, values, dim=1)
print(f"Mean cosine similarity: {cosine_sim.mean():.4f}")
```

## Key Changes from Old Code

### 1. **Read Operation** (Fixed)
- **Old:** Divided memory by counts (`memory / counts`) - destroyed signal
- **New:** Pure mean pooling (`memory.mean(dim=1)`) - preserves signal
- **Why:** Mean pooling leverages Law of Large Numbers to cancel noise

### 2. **Default Scaling** (Fixed)
- **Old:** `write_scale="1/sqrt(KH)"` - permanently attenuated signal
- **New:** `write_scale="unit"` (default) - unity gain, preserves magnitude
- **Why:** Unity gain ensures `write(v) → read() ≈ v`

### 3. **Block Size Parameter** (New)
- Use `block_size` parameter to automatically create theory-compatible BBPMAddressing
- No need to manually create addressing classes
- Invalid sizes show helpful suggestions

## Common Patterns

### Pattern 1: Basic Memory Operations

```python
from bbpm import BBPMMemoryFloat

# Create memory
memory = BBPMMemoryFloat(
    D=1_000_000,
    d=64,
    K=50,
    H=1,
    block_size=1024,  # Automatic BBPMAddressing
    seed=42
)

# Write
keys = torch.tensor([1, 2, 3], dtype=torch.int64)
values = torch.randn(3, 64)
memory.write(keys, values)

# Read
retrieved = memory.read(keys)

# Clear for new sequence
memory.clear()
```

### Pattern 2: Explicit Addressing Control

```python
from bbpm import BBPMMemoryFloat, BBPMAddressing

# Create addressing explicitly
addressing = BBPMAddressing(
    D=1_000_000,
    block_size=1024,
    seed=42,
    num_hashes=2,  # Multi-hash for better error correction
    K=50
)

# Use with memory
memory = BBPMMemoryFloat(
    D=1_000_000,
    d=64,
    K=50,
    H=2,
    hash_fn=addressing  # Use explicit addressing
)
```

### Pattern 3: Multi-Hash for Robustness

```python
memory = BBPMMemoryFloat(
    D=1_000_000,
    d=64,
    K=30,
    H=3,  # 3 independent hashes
    block_size=1024,
    seed=42
)

# Each item writes to K*H = 30*3 = 90 slots
# Reduces variance, improves retrieval quality
```

### Pattern 4: GPU Usage

```python
memory = BBPMMemoryFloat(
    D=10_000_000,
    d=128,
    K=50,
    H=1,
    block_size=4096,
    device="cuda",  # GPU memory
    seed=42
)

# Keys and values automatically moved to GPU
keys = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
values = torch.randn(3, 128, device="cuda")
memory.write(keys, values)
retrieved = memory.read(keys)  # Returns GPU tensor
```

## Valid Block Sizes

Block size must be:
1. **Power of 2**: 256, 512, 1024, 2048, 4096, ...
2. **Even n_bits**: log2(block_size) must be even
   - ✓ Valid: 256 (2^8), 1024 (2^10), 4096 (2^12)
   - ✗ Invalid: 512 (2^9), 2048 (2^11)

### Getting Suggestions

```python
from bbpm import BBPMAddressing

# Get valid suggestions for your D
D = 100_000
suggestions = BBPMAddressing.suggest_valid_block_sizes(D)
print(f"Valid block sizes for D={D}: {suggestions}")
# Output: [256, 1024, 4096, ...]
```

### Handling Invalid Block Sizes

```python
try:
    memory = BBPMMemoryFloat(
        D=100_000,
        d=64,
        K=50,
        H=1,
        block_size=512,  # Invalid: odd n_bits
        seed=42
    )
except ValueError as e:
    print(e)
    # Error includes helpful suggestions!
```

## Understanding the Theory

### Mean Pooling (Why It Works)

The read operation uses **mean pooling** across K*H slots:

```python
# What happens internally:
indices = addressing.indices(keys, K, H)  # Get K*H addresses
gathered = memory[indices]                 # Gather from those slots
result = gathered.mean(dim=1)              # Mean across K*H
```

**Theory:**
- Each slot contains: `slot = v_target + Σ(v_noise_i)`
- Mean over K slots: `mean([v+noise1, v+noise2, ...]) = v + mean(noise)`
- Law of Large Numbers: `mean(noise) → 0` as K increases
- Result: `output ≈ v_target` (signal preserved)

### Unity Gain Pipeline

With `write_scale="unit"` (default):

```
Write:  memory[addr] += v              (add full value)
Read:   output = mean(memory[indices]) (mean pooling)
Result: output ≈ v                     (unity gain)
```

### Why Count Debiasing Was Wrong

**Old approach (wrong):**
```
Read: output = mean(memory[indices] / counts[indices])
```

Problem: As counts grow, `v/count → 0`, signal shrinks.

**New approach (correct):**
```
Read: output = mean(memory[indices])
```

Solution: Mean preserves signal while noise cancels.

## Diagnostics

Counts are still tracked for diagnostics (not used in computation):

```python
# Get memory diagnostics
diag = memory.memory_diagnostics()
print(f"Nonzero slots: {diag['nonzero_slots']}")
print(f"Max count: {diag['max_count']}")
print(f"Mean count: {diag['mean_count_nonzero']:.2f}")

# Get diagnostics for specific keys
keys = torch.tensor([1, 2, 3], dtype=torch.int64)
diag = memory.diagnostics(keys)
print(f"Max load: {diag['max_load']}")
print(f"Collision rate: {diag['collision_rate']:.4f}")
```

## Migration from Old Code

### If you were using BlockHash:

```python
# Old code (deprecated)
from bbpm import BlockHash
hash_fn = BlockHash(D=1_000_000, block_size=10_000, seed=42)
memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, hash_fn=hash_fn)

# New code (recommended)
memory = BBPMMemoryFloat(
    D=1_000_000, d=64, K=50, H=1,
    block_size=1024,  # Use valid block_size instead
    seed=42
)
```

### If you were relying on count debiasing:

The read behavior has changed. Your code should work better now:

```python
# No code changes needed - just works better!
# Old: read() used counts → signal degraded
# New: read() uses mean → signal preserved
```

## Examples

Run the example scripts:

```bash
# Quick start examples
python examples/quick_start.py

# Minimal demo
python examples/minimal_demo.py

# Needle-in-haystack (distance invariance)
python examples/needle_sanity.py
```

## Troubleshooting

### "block_size must be power of 2"
- Use `BBPMAddressing.suggest_valid_block_sizes(D)` to get valid options
- Common valid sizes: 256, 1024, 4096, 16384

### "log2(block_size) must be even"
- You provided 512, 2048, 8192 (odd n_bits)
- Use 256, 1024, 4096 instead (even n_bits)

### Low cosine similarity
- Check load ratio: `load_ratio = (N * K * H) / D`
- Keep `load_ratio < 1.0` for high quality
- Increase `D` or decrease `N`, `K`, or `H`

### GPU out of memory
- Reduce `D` (memory slots)
- Reduce `d` (value dimension)
- Use `device="cpu"` for large memories

## Advanced Usage

### Custom Write Scaling

```python
# For special cases, you can use scaled writes
memory = BBPMMemoryFloat(
    D=1_000_000, d=64, K=50, H=1,
    write_scale="1/sqrt(KH)",  # Not recommended (attenuates signal)
    seed=42
)
```

### Combining with Transformers

```python
# BBPM as auxiliary memory in Transformer
from bbpm import BBPMMemoryFloat

bbpm_memory = BBPMMemoryFloat(
    D=10_000_000, d=768, K=50, H=1,
    block_size=4096,
    device="cuda"
)

# In your forward pass:
# Write current hidden states
bbpm_memory.write(token_ids, hidden_states)

# Retrieve relevant context
retrieved = bbpm_memory.read(retrieval_keys)

# Fuse with current hidden states
output = hidden_states + 0.1 * retrieved
```
