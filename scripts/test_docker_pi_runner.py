#!/usr/bin/env python
"""Test the Docker Pi runner with a simple task."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.pi_runner import DockerPiRunner, DockerPiRunnerConfig
from task_generator import TaskGenerator


def main():
    parser = argparse.ArgumentParser(description="Test Docker Pi runner")
    parser.add_argument(
        "--model",
        "-m",
        default="openrouter/openai/gpt-oss-20b:free",
        help="Model to use",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index from dev set",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Run without skill",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds",
    )
    args = parser.parse_args()

    # Load task
    generator = TaskGenerator(Path("tasks"))
    tasks = generator.load_all_tasks(split="dev")

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    task = tasks[args.task_index]
    print(f"Task: {task.task_id}")
    print(f"Package: {task.source_package}")
    print(f"Model: {args.model}")

    # Create runner
    config = DockerPiRunnerConfig(
        model=args.model,
        timeout=args.timeout,
    )
    runner = DockerPiRunner(config)

    # Load skill
    skill_content = ""
    if not args.no_skill:
        skill_path = Path("skills/testing-r-packages-orig.md")
        skill_content = skill_path.read_text() if skill_path.exists() else ""
        print(f"Skill: {skill_path.name if skill_path.exists() else 'none'}")
    else:
        print("Skill: no-skill (baseline)")

    # Run evaluation
    package_dir = Path("packages") / task.source_package
    result = runner.run_evaluation(
        skill_content=skill_content,
        task_instruction=task.instruction,
        task_context=task.context,
        package_dir=package_dir,
    )

    print(f"\n{'=' * 50}")
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Score: {result['score']}")
    print(f"Time: {result['execution_time']:.1f}s")
    if result.get("test_results"):
        tr = result["test_results"]
        print(f"Tests: {tr['num_passed']} passed, {tr['num_failed']} failed")
    if result.get("error"):
        print(f"Error: {result['error'][:500]}")


if __name__ == "__main__":
    main()
