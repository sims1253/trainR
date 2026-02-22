# trainR Visualizer

This visualizer provides a web-based interface to explore and compare benchmark results for the `trainR` project.

## Local Development

To run the visualizer locally:

```bash
cd visualizer
bun dev
```

## Data Management

The visualizer consumes aggregated benchmark data. To regenerate the data from the raw results, run the following command from the project root:

```bash
uv run python scripts/aggregate_results.py
```

## Deployment

Deployment is handled automatically via Vercel whenever changes are merged into the `main` branch. 
Target domain: [rbench.scholzmx.com](https://rbench.scholzmx.com)
