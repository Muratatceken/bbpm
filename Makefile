.PHONY: lint test experiments all clean

lint:
	@echo "Running ruff..."
	ruff check src/ tests/
	@echo "Running black --check..."
	black --check src/ tests/
	@echo "Running mypy..."
	mypy src/bbpm/

test:
	@echo "Running pytest..."
	pytest -q tests/

experiments:
	@echo "Running all experiments..."
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp01 --device cpu --seeds 3 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp02 --device cpu --seeds 3 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp03 --device cpu --seeds 1 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp04 --device cpu --seeds 3 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp05 --device cpu --seeds 2 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp06 --device cpu --seeds 3 || exit 1
	@PYTHONPATH=src python -m bbpm.experiments.run --exp exp07 --device cpu --seeds 3 || exit 1
	@echo "All experiments completed!"

all: lint test experiments
	@echo "All checks passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
