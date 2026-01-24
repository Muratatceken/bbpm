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
	python -m experiments.run_all

all: lint test experiments
	@echo "All checks passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
