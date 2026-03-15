# User Testing

Testing surface, tools, and resource cost classification.

---

## Validation Surfaces

### CLI (Primary)
- **Tool:** tuistory
- **Commands:** `grist-mill run`, `grist-mill validate`, `grist-mill list`, `grist-mill optimize`, `grist-mill report`, `grist-mill export`
- **Setup:** `uv sync` must be run first
- **Coverage:** All subcommands, help text, error handling, dry-run mode

### Programmatic API (Secondary)
- **Tool:** pytest (no special tooling)
- **Surface:** `from grist_mill import ...` — import and use as library
- **Coverage:** Public API surface matches documented interfaces

### Docker Evaluation (Integration)
- **Tool:** Docker CLI (via subprocess)
- **Surface:** End-to-end task execution in Docker container
- **Setup:** Docker daemon must be running
- **Coverage:** Container lifecycle, artifact injection, result capture, cleanup

## Validation Concurrency

**Machine specs:** 31 GB RAM, 24 CPU cores, ~2.8 GB baseline usage.

### CLI (tuistory)
- Each tuistory instance: ~50 MB RAM
- No shared infrastructure between instances
- Max concurrent: **5** (250 MB total, well within budget)

### Docker Evaluation
- Each Docker container: ~200-500 MB RAM depending on image
- Shared Docker daemon
- Max concurrent: **3** (conservative, avoids Docker daemon strain)

## Known Constraints

- Integration tests requiring Docker skip gracefully when Docker is unavailable
- Integration tests requiring API keys skip gracefully without credentials
- No database or persistent state between test runs
