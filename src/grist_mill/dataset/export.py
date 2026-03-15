"""Dataset export to JSON and CSV formats.

Exports datasets while preserving all task metadata. JSON export includes
a schema version and timestamp for self-description.

Validates VAL-DATASET-05.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grist_mill.dataset.core import Dataset

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"


class DatasetExport:
    """Exports datasets to JSON and CSV formats.

    All exports preserve complete task metadata including id, prompt,
    language, test_command, timeout, difficulty, setup_command, constraints,
    and dependencies.
    """

    def to_json(self, dataset: Dataset) -> str:
        """Export the dataset to a JSON string.

        The output includes a schema_version, generated_at timestamp,
        total task count, and the full list of tasks with all metadata.

        Args:
            dataset: The dataset to export.

        Returns:
            JSON string representation of the dataset.
        """
        data = {
            "schema_version": _SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_tasks": dataset.task_count,
            "tasks": [task.model_dump(mode="json") for task in dataset.list_tasks()],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_json_file(self, dataset: Dataset, path: Path | str) -> None:
        """Export the dataset to a JSON file.

        Args:
            dataset: The dataset to export.
            path: File path to write to.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        json_str = self.to_json(dataset)
        path_obj.write_text(json_str, encoding="utf-8")
        logger.info("Exported dataset (%d tasks) to %s", dataset.task_count, path_obj)

    def to_csv(self, dataset: Dataset) -> str:
        """Export the dataset to a CSV string.

        List fields (constraints, dependencies) are serialized as
        comma-separated strings.

        Args:
            dataset: The dataset to export.

        Returns:
            CSV string representation of the dataset.
        """
        output = io.StringIO()
        fieldnames = [
            "id",
            "prompt",
            "language",
            "test_command",
            "timeout",
            "difficulty",
            "setup_command",
            "constraints",
            "dependencies",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for task in dataset.list_tasks():
            row = {
                "id": task.id,
                "prompt": task.prompt,
                "language": task.language,
                "test_command": task.test_command,
                "timeout": task.timeout,
                "difficulty": task.difficulty.value,
                "setup_command": task.setup_command or "",
                "constraints": ",".join(task.constraints),
                "dependencies": ",".join(task.dependencies),
            }
            writer.writerow(row)

        return output.getvalue()

    def to_csv_file(self, dataset: Dataset, path: Path | str) -> None:
        """Export the dataset to a CSV file.

        Args:
            dataset: The dataset to export.
            path: File path to write to.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        csv_str = self.to_csv(dataset)
        path_obj.write_text(csv_str, encoding="utf-8")
        logger.info("Exported dataset (%d tasks) to %s", dataset.task_count, path_obj)
