#!/usr/bin/env python
"""DEPRECATED: This script has been removed.

This legacy executor is no longer supported. All benchmark execution must go through
the canonical runner API.

Migration:
    # Old:
    uv run python scripts/run_parallel_benchmark.py --config configs/benchmark.yaml

    # New:
    uv run python scripts/run_experiment.py --config configs/experiments/smoke.yaml

For help with the canonical runner:
    uv run python scripts/run_experiment.py --help

See ARCH_REMEDIATION_PLAN.md for architecture details.
"""

import sys


def main():
    print(
        "ERROR: This script is deprecated and has been removed.\n"
        "Use the canonical runner instead:\n"
        "  uv run python scripts/run_experiment.py --config <config.yaml>\n"
        "For help: uv run python scripts/run_experiment.py --help",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
