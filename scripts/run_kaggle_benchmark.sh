#!/bin/bash
# Run a Kaggle benchmark test in the Docker container
# Usage: ./scripts/run_kaggle_benchmark.sh <competition> <task_file>

set -e

COMPETITION=${1:-titanic}
TASK_FILE=${2:-tasks/kaggle/kaggle_titanic_exploring-survival-on-the-titanic.json}
MODEL=${3:-zai/glm-5}

echo "=== Kaggle Benchmark Test ==="
echo "Competition: $COMPETITION"
echo "Task: $TASK_FILE"
echo "Model: $MODEL"
echo ""

# Run in Docker container with data mounted
docker run --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/data/kaggle:/opt/kaggle-data:ro \
    -v $(pwd)/output:/workspace/output \
    --env-file .env \
    -e KAGGLE_COMPETITION=$COMPETITION \
    -e KAGGLE_TASK=$TASK_FILE \
    -e MODEL=$MODEL \
    posit-gskill-eval:latest \
    bash -c "
        set -e
        echo '=== Inside Container ==='
        echo 'Competition: '\$KAGGLE_COMPETITION
        echo 'Task: '\$KAGGLE_TASK
        
        # Create workspace data directory and copy competition data
        mkdir -p /workspace/data
        cp -r /opt/kaggle-data/\$KAGGLE_COMPETITION/* /workspace/data/
        
        echo ''
        echo 'Data files:'
        ls -la /workspace/data/
        
        echo ''
        echo '=== Running Benchmark ==='
        cd /workspace
        
        # Run the Python benchmark script
        uv run python scripts/test_benchmark.py \
            --task \$KAGGLE_TASK \
            --model \$MODEL \
            --data-dir /workspace/data \
            --output-dir /workspace/output/benchmark_\$KAGGLE_COMPETITION
    "

echo ""
echo "=== Benchmark Complete ==="
echo "Check output/benchmark_$COMPETITION/ for results"
