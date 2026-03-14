# Benchmark Architecture

## Data Isolation Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST MACHINE                            │
│                                                             │
│  data/kaggle/           ← Source data (immutable)          │
│  ├── titanic/                                              │
│  ├── house-prices/                                        │
│  └── ...                                                   │
│                                                             │
│  output/                ← Benchmark results                │
└─────────────────────────────────────────────────────────────┘
              │                           │
              │ volume :ro                │ volume :rw
              ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTAINER 1                              │
│                                                             │
│  /opt/kaggle-data/      ← READ-ONLY (shared source)        │
│  /workspace/data/       ← WRITABLE (copy for this run)     │
│  /workspace/output/     ← Results go here                  │
│                                                             │
│  Agent reads from /workspace/data/                         │
│  Agent writes to /workspace/output/                        │
│  Agent CANNOT modify /opt/kaggle-data/                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    CONTAINER 2                              │
│                    (same pattern)                           │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

1. **Source data is immutable**: `/opt/kaggle-data/` mounted read-only (`:ro`)
2. **Each container gets a fresh copy**: Entrypoint copies data to `/workspace/data/`
3. **Multiple containers can run in parallel**: Each has isolated workspace
4. **Results are persisted**: `/workspace/output/` mounted to host `./output/`

## Running Benchmarks

### Option 1: Docker Compose (recommended)

```bash
# Enable WSL2 integration in Docker Desktop first!
# Settings > Resources > WSL Integration > Enable for your distro

# Run single benchmark
KAGGLE_COMPETITION=titanic docker-compose run benchmark Rscript solution.R

# Run multiple in parallel
KAGGLE_COMPETITION=titanic docker-compose run benchmark Rscript solution1.R &
KAGGLE_COMPETITION=house-prices docker-compose run benchmark Rscript solution2.R &
wait
```

### Option 2: Plain Docker

```bash
# Build image
docker build -f Dockerfile.benchmark -t r-benchmark:latest .

# Run with volume mounts
docker run --rm \
  -v $(pwd)/data/kaggle:/opt/kaggle-data:ro \
  -v $(pwd)/output:/workspace/output \
  -e KAGGLE_COMPETITION=titanic \
  r-benchmark:latest \
  Rscript solution.R
```

### Option 3: Helper Script

```bash
./scripts/run_benchmark.sh titanic solution.R
```

## Parallel Execution

Multiple benchmarks can run simultaneously:

```bash
# In separate terminals or background processes
./scripts/run_benchmark.sh titanic agent1.R &
./scripts/run_benchmark.sh titanic agent2.R &
./scripts/run_benchmark.sh house-prices agent3.R &
```

Each container:
- Shares the same read-only source data
- Has its own isolated `/workspace/data/` copy
- Writes to its own `/workspace/output/` directory

## Docker Desktop WSL2 Setup

If `docker compose` doesn't work:

1. Open Docker Desktop
2. Go to Settings > Resources > WSL Integration
3. Enable integration for your WSL distro
4. Click "Apply & Restart"
5. Test: `docker compose version`
