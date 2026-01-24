.PHONY: lint test experiments all clean

lint:
	@echo "Running ruff..."
	ruff check src/ experiments/ tests/
	@echo "Running black --check..."
	black --check src/ experiments/ tests/
	@echo "Running mypy..."
	mypy src/bbpm/

test:
	@echo "Running pytest..."
	pytest -q tests/

experiments:
	@echo "Running all experiments..."
	@python experiments/run.py exp01 --device cpu --seeds 3 || exit 1
	@python experiments/run.py exp02 --device cpu --seeds 3 || exit 1
	@python experiments/run.py exp03 --device cpu --seeds 1 || exit 1
	@python experiments/run.py exp04 --device cpu --seeds 3 || exit 1
	@python experiments/run.py exp05 --device cpu --seeds 2 || exit 1
	@python experiments/run.py exp06 --device cpu --seeds 3 || exit 1
	@python experiments/run.py exp07 --device cpu --seeds 3 || exit 1
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
