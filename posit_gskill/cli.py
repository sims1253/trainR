"""CLI entrypoint for posit-gskill."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point for the posit-gskill CLI."""
    parser = argparse.ArgumentParser(
        prog="posit-gskill",
        description="Evolutionary optimization of Claude Skills for R package development using GEPA",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize subcommand
    optimize_parser = subparsers.add_parser("optimize", help="Run skill optimization using GEPA")
    optimize_parser.add_argument(
        "--seed-skill",
        default="skills/testing-r-packages-orig.md",
        help="Path to seed skill markdown",
    )
    optimize_parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task JSONs",
    )
    optimize_parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for optimization results",
    )
    optimize_parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=50,
        help="Maximum number of evaluations",
    )
    optimize_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Generate tasks subcommand
    generate_parser = subparsers.add_parser("generate", help="Generate tasks from patterns")
    generate_parser.add_argument(
        "--output-dir",
        default="tasks",
        help="Directory to save generated tasks",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a skill against tasks")
    eval_parser.add_argument(
        "--skill",
        required=True,
        help="Path to skill markdown file",
    )
    eval_parser.add_argument(
        "--tasks-dir",
        default="tasks",
        help="Directory containing task JSONs",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "optimize":
        from posit_gskill.commands import run_optimize

        run_optimize(args)
    elif args.command == "generate":
        from posit_gskill.commands import run_generate

        run_generate(args)
    elif args.command == "evaluate":
        from posit_gskill.commands import run_evaluate

        run_evaluate(args)


if __name__ == "__main__":
    main()
