#!/usr/bin/env python3
"""DEPRECATED: This script has been removed.

This legacy executor is no longer supported. Multi-model testing must go through
the canonical runner API with a config file specifying multiple models.

Migration:
    # Old:
    uv run python scripts/mini_benchmark.py
    (tested all models on a single task: tasks/mined/tidyverse_readr_1615.json)

    # New:
    # Create a config file with multiple models in models.names, then:
    uv run python scripts/run_experiment.py --config configs/experiments/multi_model.yaml

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
        "For multi-model testing, specify multiple models in your config file.\n"
        "For help: uv run python scripts/run_experiment.py --help",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
