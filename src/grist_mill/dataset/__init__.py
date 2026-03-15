"""Dataset management module.

Provides dataset operations for the grist-mill framework:
- Dataset: Core container for tasks with ID uniqueness enforcement
- StratifiedSplitter: Train/dev/test splitting with difficulty/language stratification
- DatasetVersioningStore: Immutable snapshot versioning
- DatasetDecontamination: Duplicate and near-duplicate detection
- DifficultyEstimator: Automatic difficulty estimation (easy/medium/hard)
- DatasetExport: Export to JSON and CSV with full metadata preservation
- DatasetQualityReport: Quality health reports
- import_tasks_from_yaml: Manual task authoring via YAML

Validates:
- VAL-DATASET-01 through VAL-DATASET-06
- VAL-TASKFMT-02 through VAL-TASKFMT-04
"""

from __future__ import annotations

# Import from submodule to avoid circular imports
from grist_mill.dataset.core import (
    Dataset,
    DuplicateEntry,
)
from grist_mill.dataset.decontamination import (
    DatasetDecontamination,
    DatasetDecontaminationResult,
)
from grist_mill.dataset.difficulty import (
    DifficultyEstimator,
)
from grist_mill.dataset.export import (
    DatasetExport,
)
from grist_mill.dataset.quality import (
    DatasetQualityIssue,
    DatasetQualityReport,
)
from grist_mill.dataset.splitting import (
    DatasetSplit,
    StratifiedSplitter,
)
from grist_mill.dataset.versioning import (
    DatasetVersion,
    DatasetVersioningStore,
)
from grist_mill.dataset.yaml_import import (
    import_tasks_from_yaml,
)

__all__ = [
    "Dataset",
    "DatasetDecontamination",
    "DatasetDecontaminationResult",
    "DatasetExport",
    "DatasetQualityIssue",
    "DatasetQualityReport",
    "DatasetSplit",
    "DatasetVersion",
    "DatasetVersioningStore",
    "DifficultyEstimator",
    "DuplicateEntry",
    "StratifiedSplitter",
    "import_tasks_from_yaml",
]
