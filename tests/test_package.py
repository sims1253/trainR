"""Tests for grist-mill package scaffolding."""

from __future__ import annotations


class TestPackageVersion:
    """Verify package version is accessible and valid."""

    def test_version_is_string(self) -> None:
        from grist_mill import __version__

        assert isinstance(__version__, str)

    def test_version_follows_semver(self) -> None:
        from grist_mill import __version__

        parts = __version__.split(".")
        assert len(parts) >= 3
        major, minor, patch = parts[0], parts[1], parts[2].split("+")[0].split("-")[0]
        assert major.isdigit()
        assert minor.isdigit()
        assert patch.isdigit()


class TestCLIEntrypoint:
    """Verify CLI entrypoint is installed and functional."""

    def test_cli_group_exists(self) -> None:
        from grist_mill.cli.main import cli

        assert cli is not None

    def test_cli_has_version_option(self) -> None:
        from grist_mill.cli.main import cli

        params = {p.name for p in cli.params}
        assert "version" in params
