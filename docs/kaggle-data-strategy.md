# Kaggle Data Strategy for Benchmarking

## Problem
- Downloading datasets for each benchmark run is wasteful
- Some datasets are large (100MB - 10GB+)
- Need efficient access during containerized benchmark runs

## Solution: Layered Approach

### 1. Local Cache (development)
```
data/kaggle/
├── titanic/
├── house-prices-advanced-regression-techniques/
├── manifest.json
└── ...
```

Download once, reuse across runs:
```bash
uv run python scripts/download_kaggle_data.py --all
```

### 2. Docker Image (small datasets < 100MB)

For small datasets, bake into Docker image:

```dockerfile
# Dockerfile.benchmark
FROM r-base:4.3

# Copy pre-downloaded data (read-only in image)
COPY data/kaggle/ /opt/kaggle-data/

# Create working directory
RUN mkdir -p /workspace/data

# Entry point copies data to workspace if needed
COPY scripts/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

```bash
# entrypoint.sh
#!/bin/bash
# Copy requested competition data to workspace
if [ -n "$KAGGLE_COMPETITION" ]; then
    cp -r "/opt/kaggle-data/$KAGGLE_COMPETITION" /workspace/data/
fi
exec "$@"
```

### 3. Volume Mount (large datasets > 100MB)

For large datasets, mount as volume:

```yaml
# docker-compose.yml
services:
  benchmark:
    image: r-benchmark:latest
    volumes:
      - kaggle-data:/opt/kaggle-data:ro
      - ./output:/workspace/output
    environment:
      - KAGGLE_COMPETITION=titanic

volumes:
  kaggle-data:
    external: true  # Pre-populated volume
```

### 4. Dataset Size Tiers

| Tier | Size | Strategy |
|------|------|----------|
| Tiny (< 1MB) | titanic, house-prices | Bake into image |
| Small (1-100MB) | Most tabular | Bake into image or volume |
| Medium (100MB-1GB) | Some competitions | Volume mount |
| Large (> 1GB) | Image, audio, text | Volume mount + lazy download |

## Implementation

### Step 1: Download datasets
```bash
# Accept competition rules on Kaggle website first!
# Then download:
uv run python scripts/download_kaggle_data.py --all
```

### Step 2: Build Docker image with data
```bash
# Build with small datasets baked in
docker build -f Dockerfile.benchmark -t r-benchmark:latest .
```

### Step 3: Run benchmark
```bash
# Data automatically available at /workspace/data/
docker run -e KAGGLE_COMPETITION=titanic r-benchmark:latest \
    Rscript solution.R
```

## Data Access in Benchmark Tasks

Each task's `grading` field includes:
```json
{
  "competition_slug": "titanic",
  "data_available": true,
  "grading_method": "metric"
}
```

Benchmark runner:
1. Reads `competition_slug` from task
2. Copies data from `/opt/kaggle-data/{slug}/` to `/workspace/data/`
3. Worker code reads from `/workspace/data/`
4. Predictions written to `/workspace/output/`
5. Grader evaluates against ground truth

## Pre-flight Checklist

Before running benchmarks:
- [ ] Accept competition rules on Kaggle website
- [ ] Run `download_kaggle_data.py --all`
- [ ] Verify with `download_kaggle_data.py --status`
- [ ] Build Docker image with data
