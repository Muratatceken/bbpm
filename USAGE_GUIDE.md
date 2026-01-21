# BBPM Usage Guide

This guide provides comprehensive instructions for using the Block-Based Permutation Memory (BBPM) library, running experiments, and understanding the codebase.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Library Usage](#library-usage)
4. [Running Experiments](#running-experiments)
5. [Running Tests](#running-tests)
6. [Codebase Structure](#codebase-structure)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12.0 (install from [pytorch.org](https://pytorch.org))
- CUDA (optional, for GPU support)

### Basic Installation

```bash
# Clone the repository (if not already done)
cd /path/to/bbpm

# Install in editable mode
pip install -e .

# Or using make
make install
```

### Installation with LLM Support

For experiment 07 (LLM integration), you need additional dependencies:

```bash
pip install -e ".[llm]"

# Or using make
make install-llm
```

This installs `transformers` and related packages for Hugging Face model support.

### Verify Installation

```bash
python -c "import bbpm; print(bbpm.__version__)"
# Should output: 0.1.0
```

---

## Quick Start

### Basic Example: Write and Read

```python
import torch
from bbpm import BBPMMemoryFloat, set_global_seed

# Set seed for reproducibility
set_global_seed(42)

# Create memory: 1M slots, 64-dim values, K=50 active slots, H=1 hash
memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, device="cpu")

# Write some key-value pairs
keys = torch.tensor([1, 2, 3, 4, 5])
values = torch.randn(5, 64)  # Random 64-dimensional vectors
memory.write(keys, values)

# Read back the values
retrieved = memory.read(keys)

# Check similarity (should be high for small K*N/D)
import torch.nn.functional as F
cosine_sim = F.cosine_similarity(retrieved, values, dim=1)
print(f"Mean cosine similarity: {cosine_sim.mean().item():.4f}")
```

### Using Different Hash Functions

```python
from bbpm import BBPMMemoryFloat, BlockHash, GlobalAffineHash

# Option 1: Use default GlobalAffineHash
memory1 = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1)

# Option 2: Use BBPMAddressing (recommended, PRP-based addressing)
memory2 = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, block_size=1024, seed=42)

# Option 3: Use custom hash function
from bbpm import GlobalAffineHash
custom_hash = GlobalAffineHash(D=1_000_000, seed=123)
memory3 = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, hash_fn=custom_hash)
```

### GPU Usage

```python
import torch
from bbpm import BBPMMemoryFloat, get_device

# Auto-detect device
device = get_device("auto")  # Returns "cuda" if available, else "cpu"

# Or specify explicitly
memory = BBPMMemoryFloat(
    D=10_000_000,  # 10M slots
    d=128,         # 128-dim vectors
    K=50,          # 50 active slots per item
    H=3,           # 3 independent hashes (multi-hash)
    device="cuda"  # Use GPU
)

# Keys and values will be automatically moved to the correct device
keys = torch.tensor([1, 2, 3], device="cuda")
values = torch.randn(3, 128, device="cuda")
memory.write(keys, values)
retrieved = memory.read(keys)
```

---

## Library Usage

### Core Memory Classes

#### `BBPMMemoryFloat`

The canonical BBPM implementation with float superposition and counts.

```python
from bbpm import BBPMMemoryFloat

memory = BBPMMemoryFloat(
    D=1_000_000,              # Total memory slots
    d=64,                    # Value dimension
    K=50,                    # Active slots per item per hash
    H=1,                     # Number of independent hashes
    block_size=None,         # Optional: creates BBPMAddressing if provided (must be power of 2, even n_bits)
    hash_fn=None,            # Optional: custom hash function (overrides block_size if provided)
    dtype=torch.float32,     # Data type
    device="cpu",            # Device ("cpu" or "cuda")
    write_scale="1/sqrt(KH)", # Scaling: "unit" or "1/sqrt(KH)"
    seed=42,                 # Seed for deterministic hashing
)

# Methods
memory.write(keys, values)  # Write key-value pairs
memory.read(keys)           # Read values for keys
memory.clear()              # Clear all memory
memory.diagnose()           # Get diagnostics (occupancy, collisions, etc.)
```

#### `BinaryBBPMBloom`

Binary variant for membership testing (Bloom filter-like).

```python
from bbpm import BinaryBBPMBloom

bloom = BinaryBBPMBloom(
    D=1_000_000,
    K=50,
    H=3,
    device="cpu",
    hash_fn=None,  # Optional
    seed=42,
)

# Write (sets bits to 1)
bloom.write(keys, values)  # values are ignored, only keys matter

# Read (returns fraction of bits set, 0.0 to 1.0)
scores = bloom.read(keys)  # Shape: [B]
```

### Hash Functions

#### `GlobalAffineHash`

Global hash function that maps keys to indices across the entire memory.

```python
from bbpm import GlobalAffineHash

hash_fn = GlobalAffineHash(D=1_000_000, seed=42)
indices = hash_fn.indices(keys, K=50, H=1)  # Shape: [B, K*H]
```

#### `BBPMAddressing` (Recommended)

Theory-compatible PRP-based addressing that guarantees no self-collisions.

```python
from bbpm import BBPMAddressing

# Create BBPMAddressing
addressing = BBPMAddressing(
    D=1_000_000,
    block_size=1024,  # Must be power of 2, and log2(block_size) must be even
    seed=42,
    num_hashes=1,  # H
    K=50           # Active slots per item per hash
)

# Use with memory
from bbpm import BBPMMemoryFloat
memory = BBPMMemoryFloat(
    D=1_000_000, d=64, K=50, H=1,
    hash_fn=addressing  # Use BBPMAddressing
)

# Or use block_size parameter (creates BBPMAddressing internally)
memory = BBPMMemoryFloat(
    D=1_000_000, d=64, K=50, H=1,
    block_size=1024  # Automatically creates BBPMAddressing
)
```

**Key differences from BlockHash:**
- **Guaranteed bijection**: PRP ensures no self-collisions within a block
- **Theory-compatible**: Matches paper specification exactly
- **Constraints**: block_size must be power of 2 AND log2(block_size) must be even (e.g., 256, 1024, 4096, not 512, 2048)

#### `BlockHash` (Deprecated)

⚠️ **DEPRECATED**: Block-based hash that uses hash-based offset mapping (not guaranteed bijection). 
Use `BBPMAddressing` instead for theory-compatible addressing.

```python
from bbpm import BlockHash  # DeprecationWarning will be issued

block_hash = BlockHash(D=1_000_000, block_size=10_000, seed=42)
indices = block_hash.indices(keys, K=50, H=1)  # Shape: [B, K*H]
```

**Migration path from BlockHash to BBPMAddressing:**

```python
# Old code (deprecated)
from bbpm import BlockHash
block_hash = BlockHash(D=1_000_000, block_size=10_000, seed=42)
memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, hash_fn=block_hash)

# New code (recommended)
from bbpm import BBPMMemoryFloat
# Option 1: Use block_size parameter (simplest)
memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, block_size=1024, seed=42)

# Option 2: Create BBPMAddressing explicitly
from bbpm import BBPMAddressing
addressing = BBPMAddressing(D=1_000_000, block_size=1024, seed=42, num_hashes=1, K=50)
memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1, hash_fn=addressing)
```

**Important notes:**
- BlockHash allowed arbitrary block_size; BBPMAddressing requires power-of-2 with even n_bits
- If your old block_size doesn't meet constraints, choose closest valid size (e.g., 512 → 256 or 1024)

### Diagnostics

```python
from bbpm import (
    occupancy_summary,
    collision_rate,
    max_load,
    gini_load,
    self_collision_prob,
)

# Get occupancy statistics
indices = hash_fn.indices(keys, K=50, H=1).flatten()
summary = occupancy_summary(indices, D=1_000_000)
print(f"Mean load: {summary['mean_load']:.2f}")
print(f"Max load: {summary['max_load']}")
print(f"Gini coefficient: {summary['gini_load']:.4f}")

# Collision rate
coll_rate = collision_rate(indices)
print(f"Collision rate: {coll_rate:.4f}")

# Self-collision probability (for a single item)
prob = self_collision_prob(K=50, L=1_000_000)
print(f"Self-collision prob: {prob:.6f}")
```

### Utilities

```python
from bbpm import set_global_seed, get_device, get_logger, Timer

# Set random seed for reproducibility
set_global_seed(42)

# Get device
device = get_device("auto")  # "cuda" or "cpu"

# Get logger
logger = get_logger("my_experiment", log_file="log.txt")
logger.info("Starting experiment")

# Timer context manager
with Timer("Operation name"):
    # Your code here
    memory.write(keys, values)
# Prints: "Operation name: 0.123s"
```

---

## Running Experiments

### Running All Experiments

```bash
# From project root
bash scripts/run_all_experiments.sh

# Or using make
make exp
```

This will:
1. Run experiments exp01 through exp06
2. Conditionally run exp07 if `transformers` is available
3. Generate all figures in `figures/`
4. Save metrics in `results/`

### Running Individual Experiments

Each experiment has three files:
- `run.py`: Executes the experiment
- `analyze.py`: Generates figures from results
- `config.yaml`: Configuration parameters

```bash
# Example: Run exp01
cd experiments/exp01_capacity_scaling

# Run experiment
python run.py \
    --config config.yaml \
    --outdir ../../results/exp01_capacity_scaling \
    --device auto

# Generate figure
python analyze.py \
    --results ../../results/exp01_capacity_scaling/metrics.json \
    --outdir ../../figures/exp01_capacity_scaling
```

### Experiment Descriptions

#### exp01_capacity_scaling
Tests capacity vs fidelity. Sweeps number of items (N) and measures cosine similarity.

**Output**: `capacity_vs_fidelity.png`

#### exp02_ablation_K_H
Ablation study on K (sparsity) and H (multi-hash) parameters.

**Output**: `ablation_K_H.png`

#### exp03_block_vs_global
Compares block-based vs global hashing strategies.

**Output**: `block_vs_global.png`

#### exp04_kv_memory_scaling
Compares KV cache memory scaling vs BBPM (constant memory).

**Output**: `kv_vs_bbpm_memory.png`

#### exp05_needle_haystack
Long context retrieval test (needle in a haystack).

**Output**: `needle_accuracy.png`

#### exp06_drift_stability
Tests stability under key drift over time.

**Output**: `drift_stability.png`

#### exp07_llm_integration
Controller-based retrieval injection for LLMs (requires `transformers`).

**Output**: `llm_accuracy_vs_N.png`

### Reproducing ICML Results

To reproduce paper figures with fixed seeds:

```bash
bash scripts/reproduce_icml.sh

# Or using make
make reproduce
```

This uses fixed random seeds for reproducibility.

### Customizing Experiments

Edit the `config.yaml` file in each experiment directory:

```yaml
# Example: experiments/exp01_capacity_scaling/config.yaml
D: 1000000        # Total memory slots
d: 64            # Value dimension
K: 50            # Active slots per item
H: 1             # Number of hashes
N_values: [1000, 5000, 10000, 50000, 100000]  # Items to test
seed: 42         # Random seed
```

---

## Running Tests

### Run All Tests

```bash
pytest -q

# Or using make
make test
```

### Run Specific Test Files

```bash
# Test hashing determinism
pytest tests/test_hashing_determinism.py -v

# Test write-read identity
pytest tests/test_write_read_identity.py -v

# Test GPU/CPU parity
pytest tests/test_gpu_cpu_parity.py -v
```

### Run Benchmarks

```bash
# Microbenchmark: write/read latency
python benchmarks/microbench_write_read.py \
    --D 1000000 \
    --K 50 \
    --device cpu

# Occupancy benchmark
python benchmarks/occupancy_bench.py \
    --D 1000000 \
    --N 10000 50000 100000 \
    --device cpu
```

---

## Codebase Structure

```
bbpm/
├── src/bbpm/              # Core library
│   ├── __init__.py       # Main exports
│   ├── config.py         # Configuration loading
│   ├── memory/           # Memory implementations
│   │   ├── float_superposition.py  # Canonical BBPM
│   │   ├── binary_bloom.py         # Binary variant
│   │   └── eviction.py             # Eviction policies
│   ├── hashing/          # Hash functions
│   │   ├── global_hash.py          # GlobalAffineHash
│   │   ├── block_hash.py           # BlockHash
│   │   ├── multihash.py            # MultiHashWrapper
│   │   └── diagnostics.py          # Occupancy, collisions, etc.
│   └── utils/            # Utilities
│       ├── seed.py       # Random seed management
│       ├── device.py     # Device detection
│       └── logging.py    # Logging utilities
│
├── experiments/          # Paper experiments
│   ├── exp01_capacity_scaling/
│   ├── exp02_ablation_K_H/
│   ├── exp03_block_vs_global/
│   ├── exp04_kv_memory_scaling/
│   ├── exp05_needle_haystack/
│   ├── exp06_drift_stability/
│   └── exp07_llm_integration/
│
├── tests/               # Test suite
│   ├── test_hashing_determinism.py
│   ├── test_write_read_identity.py
│   ├── test_counts_unbiasedness.py
│   ├── test_collision_regimes.py
│   └── test_gpu_cpu_parity.py
│
├── benchmarks/          # Performance benchmarks
│   ├── microbench_write_read.py
│   ├── occupancy_bench.py
│   └── kernel_bench.py
│
├── scripts/             # Automation scripts
│   ├── run_all_experiments.sh
│   ├── reproduce_icml.sh
│   └── format_and_lint.sh
│
├── prototypes/          # Reference implementations (do not modify)
│   ├── bbpm_prototype.py
│   └── poc_for_bbpm_prototype.py
│
├── results/             # Experiment outputs (gitignored)
├── figures/             # Generated figures (gitignored)
└── paper/               # LaTeX paper scaffold
```

---

## Common Workflows

### Workflow 1: Basic Memory Usage

```python
import torch
from bbpm import BBPMMemoryFloat, set_global_seed

set_global_seed(42)

# Initialize memory
memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1)

# Write data
keys = torch.arange(1000)
values = torch.randn(1000, 64)
memory.write(keys, values)

# Read data
retrieved = memory.read(keys)

# Check quality
import torch.nn.functional as F
similarity = F.cosine_similarity(retrieved, values, dim=1)
print(f"Mean similarity: {similarity.mean():.4f}")
```

### Workflow 2: Streaming Data

```python
import torch
from bbpm import BBPMMemoryFloat

memory = BBPMMemoryFloat(D=10_000_000, d=128, K=50, H=3, device="cuda")

# Stream data in batches
batch_size = 1000
for i in range(0, 100000, batch_size):
    keys = torch.arange(i, i + batch_size, device="cuda")
    values = torch.randn(batch_size, 128, device="cuda")
    memory.write(keys, values)

# Query random subset
query_keys = torch.randint(0, 100000, (100,), device="cuda")
retrieved = memory.read(query_keys)
```

### Workflow 3: Custom Hash Function

```python
from bbpm import BBPMMemoryFloat, BlockHash

# Create custom block hash
block_hash = BlockHash(D=1_000_000, block_size=10_000, seed=42)

# Use with memory
memory = BBPMMemoryFloat(
    D=1_000_000,
    d=64,
    K=50,
    H=1,
    hash_fn=block_hash  # Use custom hash
)

# Use as normal
memory.write(keys, values)
retrieved = memory.read(keys)
```

### Workflow 4: Diagnostics and Monitoring

```python
from bbpm import BBPMMemoryFloat, occupancy_summary, collision_rate

memory = BBPMMemoryFloat(D=1_000_000, d=64, K=50, H=1)

# Write data
keys = torch.arange(10000)
values = torch.randn(10000, 64)
memory.write(keys, values)

# Get diagnostics
diagnostics = memory.diagnose()
print(f"Collision rate: {diagnostics['collision_rate']:.4f}")
print(f"Max load: {diagnostics['max_load']}")

# Detailed occupancy summary
occupancy = diagnostics['occupancy_summary']
print(f"Mean load: {occupancy['mean_load']:.2f}")
print(f"Unique slots: {occupancy['unique_slots_touched']}")
print(f"Gini coefficient: {occupancy['gini_load']:.4f}")
```

### Workflow 5: Experiment Pipeline

```python
from pathlib import Path
from bbpm import BBPMMemoryFloat, set_global_seed, get_device
import torch
import json

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
set_global_seed(42)
device = get_device("auto")

# Configuration
D = 1_000_000
d = 64
K = 50
H = 1
N_values = [1000, 5000, 10000, 50000]

results = []

for N in N_values:
    # Create fresh memory
    memory = BBPMMemoryFloat(D=D, d=d, K=K, H=H, device=device)
    memory.clear()
    
    # Generate data
    keys = torch.arange(N, device=device)
    values = torch.randn(N, d, device=device)
    values = torch.nn.functional.normalize(values, p=2, dim=1)
    
    # Write
    memory.write(keys, values)
    
    # Read and evaluate
    retrieved = memory.read(keys)
    cosine_sim = torch.nn.functional.cosine_similarity(
        retrieved, values, dim=1
    ).mean().item()
    
    # Diagnostics
    diag = memory.diagnose()
    
    results.append({
        "N": N,
        "cosine_similarity": cosine_sim,
        "collision_rate": diag["collision_rate"],
        "max_load": diag["max_load"],
    })

# Save results
outdir = PROJECT_ROOT / "results" / "my_experiment"
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'bbpm'`

**Solution**: Install the package in editable mode:
```bash
pip install -e .
```

### CUDA Out of Memory

**Problem**: GPU runs out of memory when using large D or d.

**Solutions**:
1. Reduce `D` (memory size) or `d` (value dimension)
2. Use `dtype=torch.float16` instead of `float32`
3. Process data in smaller batches
4. Use CPU: `device="cpu"`

### Low Retrieval Accuracy

**Problem**: Cosine similarity is low when reading back values.

**Possible causes**:
1. **Too many items (N) for memory size (D)**: Increase D or decrease N
2. **K too small**: Increase K (more active slots per item)
3. **H too small**: Increase H (use multi-hash)
4. **Collisions**: Check `collision_rate` in diagnostics

**Solution**: Adjust parameters:
```python
# Increase capacity
memory = BBPMMemoryFloat(D=10_000_000, d=64, K=100, H=3)

# Or reduce items
N = 5000  # Instead of 50000
```

### Experiments Not Running

**Problem**: `bash scripts/run_all_experiments.sh` fails.

**Solutions**:
1. Check Python path: `which python`
2. Install dependencies: `pip install -e .`
3. Check device availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. For exp07, install LLM deps: `pip install -e ".[llm]"`

### Tests Hanging

**Problem**: `pytest -q` hangs or takes too long.

**Solutions**:
1. Run specific tests: `pytest tests/test_hashing_determinism.py -v`
2. Check for infinite loops in test code
3. Use timeout: `pytest --timeout=10`

### Deterministic Results

**Problem**: Results vary between runs.

**Solution**: Always set seeds:
```python
from bbpm import set_global_seed
set_global_seed(42)  # Use fixed seed
```

### Path Issues in Experiments

**Problem**: Experiments fail with "File not found" errors.

**Solution**: Always run from project root, or use `PROJECT_ROOT`:
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

---

## Additional Resources

- **Paper**: See `paper/icml2026/main.tex` for the full paper
- **Prototypes**: See `prototypes/` for reference implementations (do not modify)
- **Experiments README**: See `experiments/README.md` for detailed experiment descriptions
- **API Documentation**: See docstrings in `src/bbpm/` for detailed API docs

---

## Getting Help

If you encounter issues:

1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review experiment logs in `results/expXX_*/log.txt`
3. Run tests to verify installation: `pytest -q`
4. Check that all dependencies are installed: `pip list | grep torch`

---

## Next Steps

1. **Try the Quick Start examples** above
2. **Run a simple experiment**: `python experiments/exp01_capacity_scaling/run.py`
3. **Explore the codebase**: Start with `src/bbpm/__init__.py` to see available functions
4. **Read the paper**: Understand the theory behind BBPM
5. **Modify experiments**: Customize `config.yaml` files to test your own hypotheses

Happy coding! 🚀
