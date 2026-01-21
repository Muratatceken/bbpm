.PHONY: install test lint exp reproduce clean help

help:
	@echo "BBPM Makefile Commands:"
	@echo "  make install     - Install package in editable mode"
	@echo "  make test        - Run pytest test suite"
	@echo "  make lint        - Run ruff linter"
	@echo "  make exp         - Run all experiments"
	@echo "  make reproduce   - Reproduce ICML figures with fixed seeds"
	@echo "  make clean       - Clean generated files"

install:
	pip install -e .

install-llm:
	pip install -e ".[llm]"

test:
	pytest -q

lint:
	ruff check src/ tests/ experiments/ benchmarks/
	ruff format --check src/ tests/ experiments/ benchmarks/

format:
	ruff format src/ tests/ experiments/ benchmarks/

exp:
	bash scripts/run_all_experiments.sh

reproduce:
	bash scripts/reproduce_icml.sh

clean:
	rm -rf results/* figures/*
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
