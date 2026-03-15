"""CLI entrypoint for grist-mill.

Provides the ``grist-mill`` command with subcommands:
- ``run``: Execute or preview benchmark evaluations
- ``validate``: Validate configuration files
- ``list``: List registered artifacts and harnesses
"""

from grist_mill.cli.main import cli

__all__ = ["cli"]
