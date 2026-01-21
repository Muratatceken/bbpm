# BBPM: Block-Based Permutation Memory

A production-ready implementation of Block-Based Permutation Memory (BBPM) for efficient long-context neural memory systems.

## What is BBPM?

Block-Based Permutation Memory (BBPM) is a constant-time (O(1)) external memory mechanism that provides deterministic, address-based read/write operations independent of sequence length. Unlike attention mechanisms that use similarity-based search, BBPM uses deterministic hashing and intra-block permutations to achieve predictable retrieval fidelity.

### Key Features

- **Constant-time access**: O(K·d) per token, independent of sequence length
- **Fixed memory footprint**: O(D·d) regardless of history length
- **Theory-compatible addressing**: PRP-based intra-block addressing guarantees no self-collisions
- **Differentiable**: Gradients flow through stored payloads despite discrete addressing
- **GPU-friendly**: Fully vectorized operations for efficient GPU execution

## BBPM Theory Alignment

BBPM implements the theory from "Attention Is Not All You Need: Augmenting Transformers with a Constant-Time (O(1)) Sparse Memory Layer" with strict adherence to mathematical guarantees:

### Addressing Specification

BBPM uses two-stage addressing:

1. **Block selection**: `b_x = H(h_x) mod B`
   - Deterministic hash maps key to one of B blocks
   - Uses independent seed decorrelated from PRP

2. **Intra-block PRP**: `offset_k = P(h_x, k) mod L`
   - Feistel network PRP operating on exact bit domain `n_bits = log2(L)`
   - Guarantees bijection: distinct inputs produce distinct outputs
   - **No modulo after PRP** - PRP output is directly used as offset

3. **Final address**: `addr_k = b_x · L + offset_k`
   - Combines block start and PRP-generated offset

### Constraints

- **block_size must be power of 2**: L = 2^n
- **n_bits must be even**: log2(block_size) must be even (e.g., 256, 1024, 4096, 16384)
  - This simplifies Feistel network split into equal left/right halves
- **K <= block_size**: Required for distinct offsets guarantee
- **D % block_size == 0**: Memory must be evenly divided into blocks

### Signal-to-Noise Ratio

BBPM's retrieval fidelity follows: `E[SNR] ≈ √(D/N)`

- SNR depends only on memory-to-load ratio D/N, not sequence length
- Variance reduction: `Var[SNR] ∝ 1/K`
- Capacity transition: `N* = D/(K·H)` marks transition from low-collision to high-collision regime

### Minimal Example

```python
from bbpm import BBPMMemoryFloat, BBPMAddressing

# Configuration
D, d, K, H = 1_000_000, 64, 50, 1
block_size = 1024  # Power of 2, even n_bits

# Create memory with PRP-based addressing
memory = BBPMMemoryFloat(
    D=D, d=d, K=K, H=H,
    block_size=block_size,  # Creates BBPMAddressing internally
    seed=42
)

# Write vectors
keys = torch.arange(1000, dtype=torch.int64)
values = torch.randn(1000, d)
memory.write(keys, values)

# Read back
retrieved = memory.read(keys)
```

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With LLM support (for exp07)
pip install -e ".[llm]"
```

### Running Experiments

```bash
# Run all experiments
make exp
# or
bash scripts/run_all_experiments.sh

# Reproduce ICML figures with fixed seeds
make reproduce
# or
bash scripts/reproduce_icml.sh
```

### Running Tests

```bash
make test
# or
pytest -q
```

## Documentation

📖 **For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md)**

The usage guide covers:
- Installation and setup
- Library API with examples
- Running experiments
- Running tests and benchmarks
- Common workflows
- Troubleshooting

## Project Structure

```
bbpm/
├── src/bbpm/          # Core library
├── experiments/       # Paper experiments (exp01-exp07)
├── tests/            # Test suite
├── benchmarks/       # Performance benchmarks
├── scripts/         # Automation scripts
├── paper/           # LaTeX paper scaffold
├── results/         # Experiment outputs (gitignored)
└── figures/         # Generated figures (gitignored)
```

## Experiments

See [experiments/README.md](experiments/README.md) for detailed information about each experiment.

- **exp01_capacity_scaling**: Capacity vs fidelity analysis
- **exp02_ablation_K_H**: Ablation study on K and H parameters
- **exp03_block_vs_global**: Block vs global hashing comparison
- **exp04_kv_memory_scaling**: KV cache vs BBPM memory scaling
- **exp05_needle_haystack**: Long context retrieval test
- **exp06_drift_stability**: Key drift stability analysis
- **exp07_llm_integration**: LLM integration with controller-based retrieval

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy, Matplotlib, PyYAML, tqdm, pytest
- Optional: transformers (for LLM experiments)

## Citation

If you use this code, please cite:

```bibtex
@article{bbpm2026,
  title={Block-Based Permutation Memory for Long-Context Neural Systems},
  author={...},
  journal={ICML},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.
