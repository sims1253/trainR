"""Regression tests for module import wiring."""

from __future__ import annotations

import subprocess
import sys


def test_config_module_imports_without_circular_error() -> None:
    """Importing config directly should not trigger bench/evaluation cycles."""
    proc = subprocess.run(
        [sys.executable, "-c", "import config; assert hasattr(config, 'get_llm_config')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
