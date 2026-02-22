#!/bin/bash
set -e

LOGS_DIR="logs/baselines"
mkdir -p "$LOGS_DIR"

echo "Starting baseline runs at $(date)"

# Run baselines sequentially
for MODEL in stepfun openai nvidia minimax; do
  echo "=== Running ${MODEL} no-skill baseline ===" 
  make baseline-${MODEL}-no-skill 2>&1 | tee "$LOGS_DIR/${MODEL}-no-skill.log"
  
  echo "=== Running ${MODEL} skill baseline ===" 
  make baseline-${MODEL}-skill 2>&1 | tee "$LOGS_DIR/${MODEL}-skill.log"
done

echo "All baselines complete at $(date)"
