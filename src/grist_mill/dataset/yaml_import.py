"""Manual task authoring via YAML import.

Supports importing tasks from YAML files or strings, with full metadata
validation via Pydantic models.

Validates VAL-TASKFMT-03.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from grist_mill.schemas import Task

logger = logging.getLogger(__name__)


def import_tasks_from_yaml(source: str | Path) -> list[Task]:
    """Import tasks from a YAML source.

    The YAML must contain a top-level 'tasks' key with a list of task
    definitions. Each task definition is validated against the Task schema.

    Example YAML:
        tasks:
          - id: task-001
            prompt: Fix the off-by-one error.
            language: python
            test_command: pytest test_loop.py
            timeout: 60
            difficulty: MEDIUM

    Args:
        source: YAML string or file path to read from.

    Returns:
        List of validated Task objects.

    Raises:
        FileNotFoundError: If source is a path that doesn't exist.
        ValueError: If the YAML contains duplicate task IDs.
        ValidationError: If any task definition fails Pydantic validation.
        yaml.YAMLError: If the YAML is malformed.
    """
    if isinstance(source, Path):
        if not source.exists():
            msg = f"YAML file not found: {source}"
            raise FileNotFoundError(msg)
        yaml_content = source.read_text(encoding="utf-8")
        logger.info("Importing tasks from YAML file: %s", source)
    else:
        yaml_content = source
        logger.info("Importing tasks from YAML string.")

    data = yaml.safe_load(yaml_content)

    if not isinstance(data, dict) or "tasks" not in data:
        msg = "YAML must contain a top-level 'tasks' key with a list of task definitions."
        raise ValueError(msg)

    raw_tasks = data["tasks"]
    if not isinstance(raw_tasks, list):
        msg = "'tasks' must be a list."
        raise ValueError(msg)

    # Validate and parse tasks
    tasks: list[Task] = []
    seen_ids: set[str] = set()

    for raw_task in raw_tasks:
        # Ensure difficulty string is uppercase for enum matching
        if isinstance(raw_task, dict) and "difficulty" in raw_task:
            raw_task["difficulty"] = str(raw_task["difficulty"]).upper()

        task = Task.model_validate(raw_task)
        tasks.append(task)

        # Check for duplicate IDs within the YAML
        if task.id in seen_ids:
            msg = f"Duplicate task ID in YAML: '{task.id}'"
            raise ValueError(msg)
        seen_ids.add(task.id)

    logger.info("Successfully imported %d tasks from YAML.", len(tasks))
    return tasks
