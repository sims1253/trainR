#!/usr/bin/env python
"""Test the Pi runner with a simple task."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.pi_runner import PiRunner, PiRunnerConfig
from task_generator import TaskGenerator


def main():
    parser = argparse.ArgumentParser(description="Test Pi runner with a task")
    parser.add_argument(
        "--model",
        "-m",
        default="openrouter/openai/gpt-oss-20b:free",
        help="Model to use (default: openrouter/openai/gpt-oss-20b:free)",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index of task to use from dev set (default: 0)",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Run without skill (baseline)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    # Load tasks
    generator = TaskGenerator(Path("tasks"))
    tasks = generator.load_all_tasks(split="dev")

    if not tasks:
        print("No tasks found!")
        sys.exit(1)

    if args.task_index >= len(tasks):
        print(f"Task index {args.task_index} out of range (max: {len(tasks) - 1})")
        sys.exit(1)

    task = tasks[args.task_index]
    print(f"Testing with task: {task.task_id}")
    print(f"Package: {task.source_package}")
    print(f"Instruction: {task.instruction[:100]}...")
    print(f"Model: {args.model}")

    # Create runner
    config = PiRunnerConfig(
        model=args.model,
        timeout=args.timeout,
    )
    runner = PiRunner(config)

    # Load skill (or use empty for no-skill baseline)
    if args.no_skill:
        skill_content = ""
        print("Skill: no-skill (baseline)")
    else:
        skill_path = Path("skills/testing-r-packages-orig.md")
        skill_content = skill_path.read_text() if skill_path.exists() else ""
        print(f"Skill: {skill_path.name if skill_path.exists() else 'none'}")

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
    print(f"Turns: {result['turns_used']}")
    if result.get("test_results"):
        print(
            f"Tests: {result['test_results']['num_passed']} passed, "
            f"{result['test_results']['num_failed']} failed"
        )
    if result.get("error"):
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
