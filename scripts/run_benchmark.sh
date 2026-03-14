#!/bin/bash
# Run benchmark for a specific competition
# Usage: ./scripts/run_benchmark.sh <competition> <script>
# Example: ./scripts/run_benchmark.sh titanic solutions/titanic/baseline.R

set -e

COMPETITION=${1:-titanic}
SCRIPT=${2}

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <competition> <script>"
    echo "Example: $0 titanic solutions/titanic/baseline.R"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script not found: $SCRIPT"
    exit 1
fi

# Get the script path relative to the workspace
SCRIPT_PATH=$(realpath --relative-to="$(pwd)" "$SCRIPT")

echo "Running benchmark for: $COMPETITION"
echo "Script: $SCRIPT_PATH"

docker compose run --rm -e KAGGLE_COMPETITION="$COMPETITION" benchmark Rscript "/workspace/$SCRIPT_PATH"
