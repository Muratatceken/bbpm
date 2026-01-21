# BBPM: Block-Based Permutation Memory

A production-ready implementation of Block-Based Permutation Memory (BBPM) for efficient long-context neural memory systems.

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
