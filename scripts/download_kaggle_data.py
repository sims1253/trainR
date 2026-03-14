#!/usr/bin/env python3
"""
Kaggle Competition Data Downloader for Benchmarking

Downloads competition datasets for all competitions referenced in tasks/kaggle/
or for specific competitions. Handles authentication, error handling, and
tracks downloads in a manifest file.

Authentication:
    Requires ~/.kaggle/kaggle.json with API credentials.
    Get your credentials from: https://www.kaggle.com/settings/account

Usage:
    # Download data for all competitions in tasks/kaggle/
    uv run python scripts/download_kaggle_data.py --all

    # Download specific competition
    uv run python scripts/download_kaggle_data.py --competition titanic

    # List what's available/needed
    uv run python scripts/download_kaggle_data.py --status

    # Output directory (default: data/kaggle/)
    uv run python scripts/download_kaggle_data.py --all --output /path/to/data

    # Force re-download even if data exists
    uv run python scripts/download_kaggle_data.py --competition titanic --force
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "data/kaggle"
MANIFEST_FILENAME = "manifest.json"
TASKS_DIR = "tasks/kaggle"

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CompetitionDownloadStatus:
    """Status of a competition download"""

    competition_slug: str
    downloaded: bool
    files: list[str] = field(default_factory=list)
    size_bytes: int = 0
    downloaded_at: str | None = None
    error: str | None = None
    needs_rules_acceptance: bool = False


@dataclass
class Manifest:
    """Manifest tracking all downloads"""

    competitions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"competitions": self.competitions}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(competitions=data.get("competitions", {}))

    def update_competition(
        self,
        slug: str,
        downloaded_at: str,
        size_bytes: int,
        files: list[str],
        error: str | None = None,
        needs_rules_acceptance: bool = False,
    ) -> None:
        self.competitions[slug] = {
            "downloaded_at": downloaded_at,
            "size_bytes": size_bytes,
            "files": files,
            "error": error,
            "needs_rules_acceptance": needs_rules_acceptance,
        }

    def get_competition(self, slug: str) -> dict[str, Any] | None:
        return self.competitions.get(slug)


# =============================================================================
# Kaggle API Client
# =============================================================================


class KaggleDownloader:
    """Wrapper for Kaggle API data downloads"""

    KAGGLE_CREDS_PATH = Path.home() / ".kaggle" / "kaggle.json"

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose
        self._api = None
        self._check_credentials()

    def _check_credentials(self) -> None:
        """Check if Kaggle credentials are available"""
        if not self.KAGGLE_CREDS_PATH.exists():
            raise RuntimeError(
                f"Kaggle credentials not found at {self.KAGGLE_CREDS_PATH}\n\n"
                "To set up Kaggle API access:\n"
                "1. Go to https://www.kaggle.com/settings/account\n"
                "2. Scroll to 'API' section and click 'Create New API Token'\n"
                "3. This will download kaggle.json\n"
                "4. Move it to ~/.kaggle/kaggle.json\n"
                "5. Run: chmod 600 ~/.kaggle/kaggle.json\n\n"
                "Then install the kaggle package:\n"
                "  uv pip install kaggle"
            )

    @property
    def api(self):
        """Lazy-load the Kaggle API"""
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi

                self._api = KaggleApi()
                self._api.authenticate()
            except ImportError as exc:
                raise RuntimeError(
                    "Kaggle package not installed.\nInstall it with: uv pip install kaggle"
                ) from exc
            except Exception as e:
                raise RuntimeError(f"Failed to authenticate with Kaggle API: {e}") from e
        return self._api

    def download_competition(
        self, competition_slug: str, force: bool = False
    ) -> CompetitionDownloadStatus:
        """
        Download all files for a competition.

        Args:
            competition_slug: Competition slug (e.g., 'titanic')
            force: Force re-download even if files exist

        Returns:
            CompetitionDownloadStatus with results
        """
        competition_dir = self.output_dir / competition_slug

        # Check if already downloaded
        if not force and competition_dir.exists():
            existing_files = list(competition_dir.glob("*"))
            # Filter out hidden files and directories
            existing_files = [
                f for f in existing_files if not f.name.startswith(".") and f.is_file()
            ]
            if existing_files:
                if self.verbose:
                    print(f"  Data already exists for {competition_slug}, skipping...")
                total_size = sum(f.stat().st_size for f in existing_files)
                return CompetitionDownloadStatus(
                    competition_slug=competition_slug,
                    downloaded=True,
                    files=[f.name for f in existing_files],
                    size_bytes=total_size,
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                )

        # Create directory
        competition_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.verbose:
                print(f"  Downloading data for {competition_slug}...")

            # Download competition files
            # The API downloads a zip file and extracts it
            self.api.competition_download_files(competition_slug, path=str(competition_dir))

            # Find and extract the zip file
            zip_files = list(competition_dir.glob("*.zip"))
            extracted_files = []

            for zip_path in zip_files:
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        # Get list of files in archive
                        file_list = zf.namelist()
                        zf.extractall(competition_dir)
                        extracted_files.extend(file_list)
                    # Remove the zip file after extraction
                    zip_path.unlink()
                except zipfile.BadZipFile:
                    if self.verbose:
                        print(f"  Warning: Could not extract {zip_path}")

            # Get final list of files
            final_files = [
                f.name
                for f in competition_dir.glob("*")
                if not f.name.startswith(".") and f.is_file()
            ]

            total_size = sum(f.stat().st_size for f in competition_dir.glob("*") if f.is_file())

            return CompetitionDownloadStatus(
                competition_slug=competition_slug,
                downloaded=True,
                files=final_files,
                size_bytes=total_size,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            error_msg = str(e)

            # Check for 403 Forbidden (rules not accepted)
            if "403" in error_msg or "Forbidden" in error_msg:
                return CompetitionDownloadStatus(
                    competition_slug=competition_slug,
                    downloaded=False,
                    error=error_msg,
                    needs_rules_acceptance=True,
                )

            return CompetitionDownloadStatus(
                competition_slug=competition_slug,
                downloaded=False,
                error=error_msg,
            )


# =============================================================================
# Task Parsing
# =============================================================================


def extract_competition_from_filename(filename: str) -> str | None:
    """
    Extract competition slug from task filename.

    Filename format: kaggle_{competition}_{kernel}.json
    Example: kaggle_titanic_random-forest-benchmark-r.json -> titanic
    """
    if not filename.startswith("kaggle_"):
        return None

    # Remove prefix and extension
    base = filename[len("kaggle_") :]
    if base.endswith(".json"):
        base = base[: -len(".json")]

    # Split on underscore - competition is everything before the kernel name
    # But competition slugs can contain hyphens, so we need to be careful
    # The format is: kaggle_{competition-slug}_{kernel-slug}.json
    # Kernel slugs are typically separated by underscore from competition

    # Find the last underscore that separates competition from kernel
    # But this is tricky because competition names can have underscores too
    # Let's look for known patterns

    # Strategy: split on first underscore after 'kaggle_'
    parts = base.split("_", 1)
    if len(parts) < 2:
        return parts[0] if parts else None

    # The competition slug is the first part (may contain hyphens)
    return parts[0]


def get_competitions_from_tasks(tasks_dir: Path) -> set[str]:
    """
    Scan tasks directory and extract unique competition slugs.

    Args:
        tasks_dir: Path to tasks/kaggle directory

    Returns:
        Set of unique competition slugs
    """
    competitions = set()

    if not tasks_dir.exists():
        print(f"Warning: Tasks directory {tasks_dir} does not exist")
        return competitions

    for task_file in tasks_dir.glob("kaggle_*.json"):
        competition = extract_competition_from_filename(task_file.name)
        if competition:
            competitions.add(competition)

    return competitions


# =============================================================================
# Manifest Management
# =============================================================================


def load_manifest(manifest_path: Path) -> Manifest:
    """Load manifest from file or create empty one"""
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            return Manifest.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not parse manifest file, creating new one")
            return Manifest()
    return Manifest()


def save_manifest(manifest: Manifest, manifest_path: Path) -> None:
    """Save manifest to file"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format"""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of all competitions"""
    output_dir = Path(args.output)
    tasks_dir = Path(args.tasks_dir)
    manifest_path = output_dir / MANIFEST_FILENAME

    # Get competitions from tasks
    competitions = get_competitions_from_tasks(tasks_dir)

    if not competitions:
        print("No competitions found in tasks directory")
        return

    # Load manifest
    manifest = load_manifest(manifest_path)

    print("Kaggle Competition Data Status")
    print("=" * 60)
    print(f"Tasks directory: {tasks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total competitions: {len(competitions)}")
    print()

    downloaded = 0
    pending = 0
    failed = 0

    for competition in sorted(competitions):
        competition_dir = output_dir / competition
        manifest_entry = manifest.get_competition(competition)

        # Check actual files on disk
        if competition_dir.exists():
            files = [
                f.name
                for f in competition_dir.glob("*")
                if not f.name.startswith(".") and f.is_file()
            ]
            if files:
                total_size = sum(f.stat().st_size for f in competition_dir.glob("*") if f.is_file())
                downloaded += 1
                print(f"  [OK] {competition}")
                print(f"       Files: {len(files)}, Size: {format_size(total_size)}")
                if manifest_entry and manifest_entry.get("downloaded_at"):
                    print(f"       Downloaded: {manifest_entry['downloaded_at']}")
            else:
                pending += 1
                print(f"  [EMPTY] {competition}")
        else:
            # Check if there was a previous error
            if manifest_entry and manifest_entry.get("error"):
                if manifest_entry.get("needs_rules_acceptance"):
                    print(f"  [RULES] {competition}")
                    print(f"       Visit: https://www.kaggle.com/competitions/{competition}/rules")
                else:
                    print(f"  [FAILED] {competition}")
                    print(f"       Error: {manifest_entry['error'][:100]}")
                failed += 1
            else:
                pending += 1
                print(f"  [PENDING] {competition}")

    print()
    print(f"Summary: {downloaded} downloaded, {pending} pending, {failed} failed")


def cmd_download(args: argparse.Namespace) -> None:
    """Download competition data"""
    output_dir = Path(args.output)
    tasks_dir = Path(args.tasks_dir)
    manifest_path = output_dir / MANIFEST_FILENAME

    # Determine which competitions to download
    if args.competition:
        competitions = {args.competition}
    elif args.all:
        competitions = get_competitions_from_tasks(tasks_dir)
    else:
        print("Error: Specify --competition <name> or --all")
        sys.exit(1)

    if not competitions:
        print("No competitions to download")
        return

    print(f"Will download data for {len(competitions)} competition(s)")
    if args.dry_run:
        for c in sorted(competitions):
            print(f"  - {c}")
        return

    # Initialize downloader
    try:
        downloader = KaggleDownloader(output_dir, verbose=args.verbose)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load existing manifest
    manifest = load_manifest(manifest_path)

    # Download each competition
    success_count = 0
    fail_count = 0
    skip_count = 0

    for competition in sorted(competitions):
        print(f"\n[{competition}]")

        status = downloader.download_competition(competition, force=args.force)

        if status.downloaded:
            success_count += 1
            manifest.update_competition(
                slug=competition,
                downloaded_at=status.downloaded_at or datetime.now(timezone.utc).isoformat(),
                size_bytes=status.size_bytes,
                files=status.files,
            )
            print(f"  Downloaded {len(status.files)} files ({format_size(status.size_bytes)})")
        elif status.needs_rules_acceptance:
            fail_count += 1
            manifest.update_competition(
                slug=competition,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                size_bytes=0,
                files=[],
                error=status.error,
                needs_rules_acceptance=True,
            )
            print("  ERROR: You must accept the competition rules")
            print(
                f"  Visit https://www.kaggle.com/competitions/{competition}/rules to accept rules, then re-run"
            )
        else:
            fail_count += 1
            manifest.update_competition(
                slug=competition,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                size_bytes=0,
                files=[],
                error=status.error,
            )
            print(f"  ERROR: {status.error}")

    # Save manifest
    save_manifest(manifest, manifest_path)
    if args.verbose:
        print(f"\nManifest saved to {manifest_path}")

    # Summary
    print()
    print(
        f"Download complete: {success_count} succeeded, {fail_count} failed, {skip_count} skipped"
    )


def cmd_list(args: argparse.Namespace) -> None:
    """List all competitions found in tasks"""
    tasks_dir = Path(args.tasks_dir)
    competitions = get_competitions_from_tasks(tasks_dir)

    if not competitions:
        print("No competitions found in tasks directory")
        return

    print(f"Competitions found in {tasks_dir}:")
    for competition in sorted(competitions):
        print(f"  {competition}")


# =============================================================================
# Main CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle competition data for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for all competitions in tasks/kaggle/
  uv run python scripts/download_kaggle_data.py --all

  # Download specific competition
  uv run python scripts/download_kaggle_data.py --competition titanic

  # List what's available/needed
  uv run python scripts/download_kaggle_data.py --status

  # Custom output directory
  uv run python scripts/download_kaggle_data.py --all --output /path/to/data

  # Force re-download
  uv run python scripts/download_kaggle_data.py --competition titanic --force
""",
    )

    # Action flags (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--all",
        action="store_true",
        help="Download data for all competitions in tasks directory",
    )
    action_group.add_argument(
        "--competition",
        metavar="SLUG",
        help="Download data for a specific competition (e.g., 'titanic')",
    )
    action_group.add_argument(
        "--status",
        action="store_true",
        help="Show download status for all competitions",
    )
    action_group.add_argument(
        "--list",
        action="store_true",
        dest="list_competitions",
        help="List all competitions found in tasks directory",
    )

    # Options
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for downloaded data (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tasks-dir",
        default=TASKS_DIR,
        help=f"Directory containing task JSON files (default: {TASKS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Dispatch to appropriate command
    if args.status:
        cmd_status(args)
    elif args.list_competitions:
        cmd_list(args)
    else:
        cmd_download(args)


if __name__ == "__main__":
    main()
