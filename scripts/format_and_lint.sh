#!/bin/bash
# Format and lint code using ruff

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Formatting and linting code..."

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    echo "Installing ruff..."
    pip install ruff
fi

# Format code
echo "Formatting code..."
ruff format src/ tests/ experiments/ benchmarks/

# Lint code
echo "Linting code..."
ruff check src/ tests/ experiments/ benchmarks/

echo ""
echo "Formatting and linting complete!"
