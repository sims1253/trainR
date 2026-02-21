#!/usr/bin/env bash
# Run all Python linting and type checking

set -e

echo "=== Ruff Lint ==="
uv run ruff check .

echo "=== Ruff Format Check ==="
uv run ruff format --check .

echo "=== Ty Type Check ==="
uvx ty check task_generator/ evaluation/ optimization/ experiment/

echo "=== All checks passed! ==="
