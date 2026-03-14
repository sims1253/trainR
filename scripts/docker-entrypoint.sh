#!/bin/bash
set -e

echo "=== Benchmark Container Starting ==="
echo "Competition: ${KAGGLE_COMPETITION:-not specified}"

# Ensure workspace data directory exists (tmpfs may need initialization)
mkdir -p /workspace/data /workspace/output

# Copy competition data to workspace if specified
if [ -n "$KAGGLE_COMPETITION" ]; then
    if [ -d "/opt/kaggle-data/$KAGGLE_COMPETITION" ]; then
        echo "Copying data from read-only source to workspace..."
        echo "  Source: /opt/kaggle-data/$KAGGLE_COMPETITION/ (read-only)"
        echo "  Destination: /workspace/data/ (writable)"
        cp -r "/opt/kaggle-data/$KAGGLE_COMPETITION"/* /workspace/data/
        echo "Data ready. Files:"
        ls -la /workspace/data/
    else
        echo "ERROR: Competition data not found at /opt/kaggle-data/$KAGGLE_COMPETITION/"
        echo "Available competitions:"
        ls /opt/kaggle-data/ 2>/dev/null || echo "  (none - check volume mounts)"
        exit 1
    fi
else
    echo "WARNING: No KAGGLE_COMPETITION specified. No data copied."
fi

echo "=== Ready to run benchmark ==="
echo ""

# Execute command
exec "$@"
