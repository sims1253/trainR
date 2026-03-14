#!/usr/bin/env python3
"""
Kaggle Kernel Mining System for R Benchmarking Tasks

This script mines R notebooks from Kaggle competitions to extract
high-quality benchmarking tasks. It uses:
- Kaggle's official Python API for programmatic access
- Provider-native inference for LLM-based task evaluation
- Pydantic for structured outputs

Authentication:
    Requires ~/.kaggle/kaggle.json with API credentials.
    Get your credentials from: https://www.kaggle.com/settings/account

Usage:
    python scripts/mine_kaggle.py --competition titanic --max-kernels 10
    python scripts/mine_kaggle.py --list-competitions
    python scripts/mine_kaggle.py --dry-run --competition house-prices-advanced-regression-techniques
    python scripts/mine_kaggle.py --competitions titanic,house-prices-advanced-regression-techniques --max-kernels 5
    python scripts/mine_kaggle.py --competition titanic --skip-quality-check --max-kernels 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field, field_validator

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.provider.inference import embed_schema_in_messages, generate_structured
from config import get_llm_config
from task_generator.mined_task import Difficulty, TaskType

# =============================================================================
# Kaggle-Specific Pydantic Models
# =============================================================================


class KaggleKernelSource(BaseModel):
    """Source information for a Kaggle kernel task"""

    type: str = Field(default="kaggle_kernel", description="Source type identifier")
    competition: str = Field(description="Competition slug (e.g., 'titanic')")
    competition_title: str = Field(description="Competition display title")
    kernel_slug: str = Field(description="Kernel URL slug")
    kernel_ref: str = Field(description="Full kernel reference (owner/kernel-slug)")
    author: str = Field(description="Kernel author username")
    votes: int = Field(default=0, description="Number of upvotes")
    url: str = Field(description="Full URL to the kernel")
    language: str = Field(default="r", description="Programming language")
    kernel_type: str = Field(default="notebook", description="Kernel type (notebook/script)")


class KaggleTaskMetadata(BaseModel):
    """Metadata for a Kaggle task"""

    task_type: str = Field(description="Type of task")
    difficulty: str = Field(description="Difficulty level")
    quality_score: int = Field(ge=1, le=10, description="Quality rating 1-10")
    mined_at: str = Field(description="When this task was mined (ISO format)")
    competition_type: str | None = Field(
        default=None, description="Competition type (e.g., 'featured', 'research')"
    )
    tags: list[str] = Field(default_factory=list, description="Competition/kernel tags")
    dataset_size: str | None = Field(default=None, description="Size category of dataset")
    evaluation_metric: str | None = Field(default=None, description="Competition evaluation metric")


class KaggleTaskDefinition(BaseModel):
    """Task definition extracted from competition"""

    instruction: str = Field(description="Task instruction for the developer")
    context: str = Field(default="", description="Competition description and context")
    problem_type: str | None = Field(
        default=None, description="Problem type (classification, regression, etc.)"
    )
    evaluation_criteria: str | None = Field(default=None, description="How solutions are evaluated")


class KaggleTaskSolution(BaseModel):
    """Solution-related information"""

    code: str = Field(description="The R code solution from the kernel")
    key_techniques: list[str] = Field(default_factory=list, description="Key techniques used")
    potential_improvements: list[str] = Field(
        default_factory=list, description="Potential improvements to the solution"
    )


class KaggleMinedTask(BaseModel):
    """Complete mined task from Kaggle"""

    task_id: str = Field(description="Unique identifier (kaggle_{competition}_{kernel_id})")
    source: KaggleKernelSource = Field(description="Source information")
    metadata: KaggleTaskMetadata = Field(description="Task metadata")
    task: KaggleTaskDefinition = Field(description="Task definition")
    solution: KaggleTaskSolution = Field(description="Solution information")


class KaggleKernelAnalysis(BaseModel):
    """
    Structured output from LLM judge for Kaggle kernel analysis.
    Used internally for evaluating kernel quality and extracting task data.
    """

    # Classification
    task_type: TaskType = Field(
        description="Type of task this kernel represents",
    )
    difficulty: Difficulty = Field(
        description="Estimated difficulty for solving this task",
    )
    quality_score: int = Field(
        ge=1,
        le=10,
        description="Task quality rating from 1-10. "
        "7+ is good for evaluation. "
        "Consider: clarity, completeness, educational value.",
    )

    # Task content
    instruction: str = Field(
        description="Human-readable task description. "
        "Should describe what the competition problem is and what needs to be done.",
    )
    context: str = Field(
        description="Competition description and relevant context. "
        "Include problem type, evaluation metric, and any constraints.",
        default="",
    )

    # Solution analysis
    key_techniques: list[str] = Field(
        default_factory=list,
        description="Key techniques/packages used in the solution. "
        "E.g., 'tidyverse data manipulation', 'xgboost for classification'.",
    )
    potential_improvements: list[str] = Field(
        default_factory=list,
        description="Potential improvements or extensions to the solution. "
        "Useful for creating variant tasks.",
    )

    # Problem classification
    problem_type: str = Field(
        default="unknown",
        description="Type of ML problem: classification, regression, clustering, nlp, etc.",
    )
    evaluation_criteria: str = Field(
        default="",
        description="How the competition evaluates submissions (metric, scoring).",
    )

    # Evaluation
    is_good_task: bool = Field(
        description="Whether this kernel makes a good benchmarking task. "
        "False if the code is too trivial, too complex, or not self-contained.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="If is_good_task is False, explain why.",
    )

    # Grading configuration for benchmark evaluation
    grading: dict[str, Any] = Field(
        default_factory=dict,
        description="Grading configuration for automated evaluation",
    )

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Quality score must be between 1 and 10")
        return v


class KaggleTaskSchema(BaseModel):
    """Schema for a Kaggle benchmark task."""

    task_id: str = Field(description="Unique task identifier")
    source: dict[str, Any] = Field(description="Source information (competition, author, etc.)")

    # Core task (required for grading)
    problem_statement: dict[str, Any] = Field(
        description="Problem description and evaluation criteria"
    )
    grading: dict[str, Any] = Field(description="Grading configuration (metric, method, etc.)")

    # Optional reference material
    reference_solution: dict[str, Any] | None = Field(
        default=None,
        description="Optional reference solution for LLM-judge fallback",
    )

    metadata: dict[str, Any] = Field(default_factory=dict, description="Task metadata")


# =============================================================================
# Data Classes for Kaggle API Data
# =============================================================================


@dataclass
class CompetitionInfo:
    """Container for competition information"""

    ref: str  # Competition slug
    title: str
    description: str
    evaluation_metric: str | None
    competition_type: str | None
    tags: list[str]
    url: str
    deadline: datetime | None = None
    reward: str | None = None


@dataclass
class KernelInfo:
    """Container for kernel information"""

    ref: str  # Full reference (author/kernel-slug)
    title: str
    author: str
    slug: str
    votes: int
    language: str
    kernel_type: str  # 'notebook' or 'script'
    url: str
    competition: str | None = None
    competition_sources: list[str] | None = None  # List of competition slugs from metadata


@dataclass
class KernelContent:
    """Container for kernel code content"""

    kernel_info: KernelInfo
    code: str  # Extracted R code
    metadata: dict[str, Any]  # Additional metadata from the kernel


# =============================================================================
# Kaggle API Client
# =============================================================================


class KaggleAPIClient:
    """Wrapper for Kaggle API with helpful error messages"""

    KAGGLE_CREDS_PATH = Path.home() / ".kaggle" / "kaggle.json"

    def __init__(self, verbose: bool = False):
        """
        Initialize Kaggle API client.

        Raises:
            RuntimeError: If Kaggle credentials are not configured
        """
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

    def list_competitions(
        self,
        page: int = 1,
        search: str | None = None,
        category: str | None = None,
    ) -> list[CompetitionInfo]:
        """
        List Kaggle competitions.

        Args:
            page: Page number for pagination
            search: Search term to filter competitions
            category: Category filter (e.g., 'featured', 'research', 'recruitment', 'gettingStarted')

        Returns:
            List of CompetitionInfo objects
        """
        competitions = []
        try:
            # Kaggle API returns a list of competition objects
            raw_comps = self.api.competitions_list(
                page=page,
                search=search,
                category=category,
            )

            for comp in raw_comps.competitions if hasattr(raw_comps, "competitions") else raw_comps:
                # Build URL - check if ref already contains full path
                if comp.ref.startswith("http"):
                    url = comp.ref
                else:
                    url = f"https://www.kaggle.com/competitions/{comp.ref}"

                comp_info = CompetitionInfo(
                    ref=comp.ref,
                    title=comp.title,
                    description=comp.description or "",
                    evaluation_metric=getattr(comp, "evaluationMetric", None),
                    competition_type=getattr(comp, "competitionType", None),
                    tags=[],  # Tags not directly available in list view
                    url=url,
                    deadline=getattr(comp, "deadline", None),
                    reward=getattr(comp, "reward", None),
                )
                competitions.append(comp_info)

        except Exception as e:
            print(f"Error listing competitions: {e}")
            return []

        return competitions

    def get_competition(self, competition_slug: str) -> CompetitionInfo | None:
        """
        Get detailed information about a specific competition.

        Args:
            competition_slug: Competition reference/slug

        Returns:
            CompetitionInfo or None if not found
        """
        try:
            # competitions_view() doesn't exist; use competitions_list with search/filter
            all_comps = self.api.competitions_list(search=competition_slug, page=1)
            comps = all_comps.competitions if hasattr(all_comps, "competitions") else all_comps

            # Find the specific competition by matching slug in ref
            comp = next(
                (
                    c
                    for c in comps
                    if c.ref.endswith(f"/{competition_slug}") or competition_slug in c.ref
                ),
                None,
            )

            if comp is None:
                print(f"Competition '{competition_slug}' not found in search results")
                return None

            # Extract tags if available (may be list of strings or tag objects)
            tags = []
            if hasattr(comp, "tags") and comp.tags:
                tags = [t if isinstance(t, str) else getattr(t, "name", str(t)) for t in comp.tags]

            # Build URL - check if ref already contains full path
            if comp.ref.startswith("http"):
                url = comp.ref
            else:
                url = f"https://www.kaggle.com/competitions/{comp.ref}"

            return CompetitionInfo(
                ref=comp.ref,
                title=comp.title,
                description=getattr(comp, "description", "") or "",
                evaluation_metric=getattr(comp, "evaluationMetric", None)
                or getattr(comp, "evaluation_metric", None),
                competition_type=getattr(comp, "competitionType", None)
                or getattr(comp, "competition_type", None),
                tags=tags,
                url=url,
                deadline=getattr(comp, "deadline", None),
                reward=getattr(comp, "reward", None),
            )
        except Exception as e:
            print(f"Error getting competition {competition_slug}: {e}")
            return None

    def list_kernels(
        self,
        competition: str,
        language: str = "r",
        sort_by: str = "voteCount",
        page: int = 1,
        page_size: int = 20,
    ) -> list[KernelInfo]:
        """
        List kernels for a competition.

        Args:
            competition: Competition slug
            language: Filter by language ('r', 'python', etc.)
            sort_by: Sort order ('voteCount', 'dateCreated', 'relevance')
            page: Page number
            page_size: Number of results per page

        Returns:
            List of KernelInfo objects
        """
        kernels = []
        try:
            raw_kernels = self.api.kernels_list(
                competition=competition,
                language=language,
                sort_by=sort_by,
                page=page,
                page_size=page_size,
            )

            # kernels_list returns list directly when filtered by competition
            kernel_list = (
                raw_kernels.kernels
                if hasattr(raw_kernels, "kernels")
                else raw_kernels
                if isinstance(raw_kernels, list)
                else []
            )
            for kernel in kernel_list:
                # kernel.ref is in format "author/kernel-slug"
                # Build URL - check if ref already contains full path
                if kernel.ref.startswith("http"):
                    url = kernel.ref
                else:
                    url = f"https://www.kaggle.com/code/{kernel.ref}"

                kernel_info = KernelInfo(
                    ref=kernel.ref,
                    title=kernel.title,
                    author=kernel.author,
                    slug=kernel.ref.split("/")[-1] if "/" in kernel.ref else kernel.ref,
                    votes=getattr(kernel, "total_votes", 0) or 0,
                    language=getattr(kernel, "language", "r"),
                    kernel_type=getattr(kernel, "kernelType", "notebook"),
                    url=url,
                    competition=competition,
                )
                kernels.append(kernel_info)

        except Exception as e:
            print(f"Error listing kernels for {competition}: {e}")
            return []

        return kernels

    def pull_kernel(self, kernel_ref: str) -> KernelContent | None:
        """
        Pull kernel content (code and metadata).

        Args:
            kernel_ref: Full kernel reference (author/kernel-slug)

        Returns:
            KernelContent with extracted R code, or None if failed
        """
        try:
            # Pull kernel - this downloads files to current directory
            # We need to handle .ipynb, .R, .Rmd, and .irnb file formats
            self.api.kernels_pull(kernel_ref, path=".", quiet=True)
            time.sleep(1.0)

            # Find the downloaded file
            # Kaggle downloads as {kernel-slug}.{ipynb|R|Rmd}
            kernel_slug = kernel_ref.split("/")[-1]

            # Check for .ipynb file first (skip these - JSON format)
            ipynb_path = Path(f"{kernel_slug}.ipynb")
            if ipynb_path.exists():
                print("    Skipping .ipynb file (JSON format, not plain text)")
                ipynb_path.unlink()  # Clean up
                return None

            # Try to find the file
            code = None
            metadata = {}

            # Check for .R file
            r_path = Path(f"{kernel_slug}.R")
            if r_path.exists():
                code = r_path.read_text()
                metadata["format"] = "r_script"
                r_path.unlink()  # Clean up

            # Check for .r file (lowercase extension)
            r_path_lower = Path(f"{kernel_slug}.r")
            if r_path_lower.exists():
                code = r_path_lower.read_text()
                metadata["format"] = "r_script"
                r_path_lower.unlink()  # Clean up

            # Check for .Rmd file (R Markdown)
            rmd_path = Path(f"{kernel_slug}.Rmd")
            if rmd_path.exists():
                code = rmd_path.read_text(encoding="utf-8")
                metadata["format"] = "r_markdown"
                rmd_path.unlink()  # Clean up

            # Check for .irnb file (R Notebook)
            irnb_path = Path(f"{kernel_slug}.irnb")
            if irnb_path.exists():
                code = irnb_path.read_text(encoding="utf-8")
                metadata["format"] = "r_notebook"
                irnb_path.unlink()  # Clean up

            if code is None:
                print(
                    f"Warning: Could not find code file (.ipynb, .R, .r, .Rmd, or .irnb) for kernel {kernel_ref}"
                )
                return None

            # Build URL - check if ref already contains full path
            if kernel_ref.startswith("http"):
                url = kernel_ref
            else:
                url = f"https://www.kaggle.com/code/{kernel_ref}"

            # Create a minimal KernelInfo since we already have the ref
            kernel_info = KernelInfo(
                ref=kernel_ref,
                title=kernel_slug,
                author=kernel_ref.split("/")[0],
                slug=kernel_slug,
                votes=0,
                language="r",
                kernel_type="script",  # Only scripts are processed (ipynb skipped)
                url=url,
            )

            return KernelContent(
                kernel_info=kernel_info,
                code=code,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error pulling kernel {kernel_ref}: {e}")
            return None

    def _extract_r_from_ipynb(self, ipynb_path: Path) -> str:
        """
        Extract R code from a Jupyter notebook.

        Args:
            ipynb_path: Path to .ipynb file

        Returns:
            Combined R code from all code cells
        """
        with open(ipynb_path) as f:
            notebook = json.load(f)

        code_cells = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                code = "".join(source) if isinstance(source, list) else source
                code_cells.append(code)

        return "\n\n# --- Cell ---\n\n".join(code_cells)

    def list_top_r_kernels(
        self,
        sort_by: str = "voteCount",
        page: int = 1,
        page_size: int = 20,
        competition_kernels_only: bool = True,
        prefer_scripts: bool = True,
    ) -> list[KernelInfo]:
        """
        List top R kernels across all of Kaggle.

        Args:
            sort_by: Sort order ('voteCount', 'dateCreated', 'relevance')
            page: Page number
            page_size: Number of results per page
            competition_kernels_only: If True, only return kernels with competition_sources
            prefer_scripts: If True, prefer script kernels over notebooks

        Returns:
            List of KernelInfo objects with competition_sources populated
        """
        kernels = []
        try:
            api_kwargs = {
                "language": "r",
                "sort_by": sort_by,
                "page": page,
                "page_size": page_size,
            }
            # Prefer scripts over notebooks when enabled
            if prefer_scripts:
                api_kwargs["kernel_type"] = "script"

            raw_kernels = self.api.kernels_list(**api_kwargs)

            # kernels_list returns a list directly when not filtered by competition
            kernel_list = (
                raw_kernels.kernels
                if hasattr(raw_kernels, "kernels")
                else raw_kernels
                if isinstance(raw_kernels, list)
                else []
            )

            for kernel in kernel_list:
                # Extract competition_sources from kernel metadata (not available in list API)
                competition_sources = []
                try:
                    metadata = self.get_kernel_metadata(kernel.ref)
                    competition_sources = metadata.get("competition_sources", [])
                    time.sleep(0.5)  # Small delay between metadata fetches
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: Could not get metadata for {kernel.ref}: {e}")

                # Filter out non-competition kernels if requested
                if competition_kernels_only and not competition_sources:
                    if self.verbose:
                        print(f"    Skipping {kernel.ref} (no competition sources)")
                    continue

                # Build URL - check if ref already contains full path
                if kernel.ref.startswith("http"):
                    url = kernel.ref
                else:
                    url = f"https://www.kaggle.com/code/{kernel.ref}"

                # Get the primary competition (first one in the list)
                primary_competition = competition_sources[0] if competition_sources else None

                kernel_info = KernelInfo(
                    ref=kernel.ref,
                    title=kernel.title,
                    author=kernel.author,
                    slug=kernel.ref.split("/")[-1] if "/" in kernel.ref else kernel.ref,
                    votes=getattr(kernel, "total_votes", 0) or 0,
                    language=getattr(kernel, "language", "r"),
                    kernel_type=getattr(kernel, "kernelType", "notebook"),
                    url=url,
                    competition=primary_competition,
                    competition_sources=competition_sources,
                )
                kernels.append(kernel_info)

        except Exception as e:
            print(f"Error listing top R kernels: {e}")
            return []

        return kernels

    def get_kernel_metadata(self, kernel_ref: str) -> dict[str, Any]:
        """
        Get metadata for a specific kernel.

        Args:
            kernel_ref: Full kernel reference (author/kernel-slug)

        Returns:
            Dict with kernel metadata including competition_sources
        """
        import json
        import os
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self.api.kernels_pull(kernel_ref, path=tmpdir, metadata=True)

                # Find and read the metadata JSON file
                for f in os.listdir(tmpdir):
                    if f.endswith(".json"):
                        with open(os.path.join(tmpdir, f)) as jf:
                            return json.load(jf)

                return {}
        except Exception as e:
            if self.verbose:
                print(f"Error getting kernel metadata for {kernel_ref}: {e}")
            return {}


# =============================================================================
# LLM Task Judge
# =============================================================================


class LLMTaskJudge:
    """Use LLM to evaluate kernel quality and extract structured task data"""

    SYSTEM_PROMPT = """You are an expert R programmer and machine learning practitioner.
Your task is to analyze Kaggle competition kernels and evaluate their suitability as
benchmarking tasks for training AI coding assistants.

Evaluate each kernel on:
1. **Completeness**: Does the code provide a complete solution (data loading, preprocessing, modeling, evaluation)?
2. **Clarity**: Is the code well-structured and readable?
3. **Educational value**: Can someone learn from this approach?
4. **Self-contained**: Can the task be understood and replicated without extensive external context?
5. **Difficulty appropriateness**: Is it neither trivial (just loading data) nor impossibly complex?

## Response Format (JSON Schema)

You MUST include these required fields in your JSON response:

**task_type** (required): One of "bug_fix", "feature_impl", "test_writing", "refactor", "documentation", "performance", "security", "dependency"

For Kaggle tasks, typically use:
- "feature_impl" for data preprocessing/feature engineering
- "performance" for model optimization
- "refactor" for code improvement tasks

**difficulty** (required): One of "easy" (simple, <50 lines of core logic), "medium" (moderate complexity, 50-200 lines), "hard" (complex pipeline, >200 lines)

**quality_score** (required): Integer 1-10. 7+ is good. Consider: completeness, clarity, educational value.

**instruction** (required): Clear task description describing what the ML problem is and what the solution should achieve.

**is_good_task** (required): Boolean. True if this kernel would make a good benchmarking task.

Optional fields:
- **context**: Competition description and evaluation criteria
- **key_techniques**: Array of key techniques/packages used (e.g., "tidyverse", "xgboost", "cross-validation")
- **potential_improvements**: Array of ways the solution could be improved
- **problem_type**: Type of ML problem (classification, regression, clustering, nlp, etc.)
- **evaluation_criteria**: How the competition evaluates submissions
- **rejection_reason**: If is_good_task is false, explain why"""

    EVALUATION_PROMPT = """Analyze this Kaggle competition kernel and evaluate its quality as a benchmarking task.

## Competition Information
**Title**: {competition_title}
**Description**: {competition_description}
**Evaluation Metric**: {evaluation_metric}

## Kernel Information
**Title**: {kernel_title}
**Author**: {author}
**Votes**: {votes}

## R Code Solution
```r
{code}
```

Provide a structured evaluation of this kernel as a potential benchmarking task.
Consider: Is the code complete? Is it educational? Does it represent a realistic ML task?
"""

    def __init__(self, model: str | None = None, api_key: str | None = None, verbose: bool = False):
        """
        Initialize LLM judge.
        Uses unified config from configs/llm.yaml by default.
        """
        self.verbose = verbose
        config = get_llm_config()

        # Get model name
        model_name = model or config.get_mining_model()
        self.model = model_name

        # Get full model config
        try:
            model_cfg = config.get_model_config(model_name)
            self.provider = model_cfg.get("provider", "")

            # Get API key from config
            self.api_key = api_key or config.get_model_api_key(model_name)

            # Store model_id for inference gateway
            self._model_id = model_cfg.get("id") or model_cfg.get("model_id", model_name)
            self._base_url = config.get_model_base_url(model_name)
            self._json_mode = model_cfg.get("capabilities", {}).get("json_mode", "native")

        except ValueError:
            # Model not in llm.yaml - treat as raw model ID
            self._model_id = model_name
            if model_name.startswith("opencode/") or model_name.startswith("zai/"):
                self._model_id = model_name.split("/", 1)[1]
                self.provider = model_name.split("/", 1)[0]
            elif model_name.startswith("openrouter/"):
                self._model_id = model_name.split("/", 1)[1]
                self.provider = "openrouter"
            else:
                self.provider = "unknown"

            self.api_key = api_key or self._resolve_api_key_for_model(model_name, config)
            self._base_url = config.get_base_url(model_name)
            self._json_mode = "native"

        if not self.api_key:
            print(f"Warning: No API key found for model '{model_name}'. Check llm.yaml config.")

    def _resolve_api_key_for_model(self, model_name: str, config: Any) -> str | None:
        """Resolve API key for a model using central resolver with fallback."""
        try:
            from bench.provider import resolve_api_key_env

            if "/" in model_name:
                provider = model_name.split("/", 1)[0]
                key_name = resolve_api_key_env(provider)
                from bench.provider import get_env_var

                key = get_env_var(key_name)
                if key:
                    return key
        except (ImportError, KeyError):
            pass

        warnings.warn(
            f"Using fallback API key detection in LLMTaskJudge for '{model_name}'.",
            DeprecationWarning,
            stacklevel=3,
        )

        if model_name.startswith("opencode/"):
            from bench.provider import get_env_var

            return get_env_var("OPENCODE_API_KEY")
        elif model_name.startswith("zai/"):
            from bench.provider import get_env_var

            return get_env_var("Z_AI_API_KEY")
        elif model_name.startswith("openrouter/"):
            from bench.provider import get_env_var

            return get_env_var("OPENROUTER_API_KEY")
        else:
            return config.get_api_key()

    def _call_inference_gateway(
        self, messages: list[dict[str, Any]], response_format: type[BaseModel]
    ) -> BaseModel:
        """Call inference gateway for structured output."""
        if self._json_mode == "prompt":
            messages = embed_schema_in_messages(messages, response_format)

        if self.verbose:
            print("\n" + "=" * 60)
            print("LLM INPUT:")
            print(f"Model: {self.model}")
            print(f"Model ID: {self._model_id}")
            print(f"Provider: {self.provider}")
            print(f"API Key: {self.api_key[:15]}..." if self.api_key else "No API key")

        try:
            result = generate_structured(
                messages=messages,
                response_schema=response_format,
                model_name=self.model,
                api_key=self.api_key,
                verbose=self.verbose,
            )
        except Exception as e:
            if self.verbose:
                print(f"\n[LLM ERROR] {type(e).__name__}: {e}")
            raise

        if self.verbose:
            print("\nLLM OUTPUT:")
            content_preview = (
                result.content[:1000] if len(result.content) > 1000 else result.content
            )
            print(content_preview)
            print("=" * 60 + "\n")

        return response_format.model_validate_json(result.content)

    def evaluate_kernel(
        self,
        kernel_content: KernelContent,
        competition_info: CompetitionInfo,
    ) -> KaggleKernelAnalysis:
        """
        Evaluate kernel quality and extract structured task data.

        Args:
            kernel_content: KernelContent with code and metadata
            competition_info: CompetitionInfo with competition details

        Returns:
            KaggleKernelAnalysis with evaluation and extracted task data
        """
        prompt = self.EVALUATION_PROMPT.format(
            competition_title=competition_info.title,
            competition_description=competition_info.description[:2000],
            evaluation_metric=competition_info.evaluation_metric or "Not specified",
            kernel_title=kernel_content.kernel_info.title,
            author=kernel_content.kernel_info.author,
            votes=kernel_content.kernel_info.votes,
            code=kernel_content.code[:30000],  # Truncate to avoid token limits
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            return cast(
                "KaggleKernelAnalysis",
                self._call_inference_gateway(messages, KaggleKernelAnalysis),
            )
        except Exception as e:
            if self.verbose:
                print(f"\n[LLM ERROR] {type(e).__name__}: {e}")
            comp_slug = (
                competition_info.ref.split("/")[-1]
                if competition_info.ref and "/" in competition_info.ref
                else competition_info.ref or ""
            )
            return KaggleKernelAnalysis(
                task_type=TaskType.FEATURE_IMPL,
                difficulty=Difficulty.MEDIUM,
                quality_score=1,
                instruction=f"Error evaluating kernel: {e!s}",
                context="",
                is_good_task=False,
                rejection_reason=f"LLM evaluation failed: {e!s}",
                grading={
                    "metric": competition_info.evaluation_metric or "unknown",
                    "metric_description": competition_info.description[:500]
                    if competition_info.description
                    else "",
                    "problem_type": classify_problem_type(competition_info.evaluation_metric),
                    "competition_slug": comp_slug,
                    "data_available": True,
                    "grading_method": "metric",
                    "baseline_score": None,
                },
            )


# =============================================================================
# Task Creation Functions
# =============================================================================


def classify_problem_type(evaluation_metric: str | None) -> str:
    """
    Classify the problem type based on evaluation metric.

    Args:
        evaluation_metric: The competition's evaluation metric string

    Returns:
        Problem type string (classification, regression, clustering, etc.)
    """
    if not evaluation_metric:
        return "unknown"

    metric_lower = evaluation_metric.lower().replace("_", " ").replace("-", " ")

    # Classification metrics
    classification_keywords = [
        "accuracy",
        "auc",
        "f1",
        "precision",
        "recall",
        "logloss",
        "log loss",
        "roc",
        "balanced",
        "matthews",
        "kappa",
        "true positive",
        "false positive",
    ]
    if any(kw in metric_lower for kw in classification_keywords):
        return "classification"

    # Regression metrics
    regression_keywords = [
        "rmse",
        "mse",
        "mae",
        "r2",
        "rmsle",
        "mape",
        "smape",
        "root mean",
        "mean absolute",
        "mean squared",
        "error",
    ]
    if any(kw in metric_lower for kw in regression_keywords):
        return "regression"

    # NLP/Text metrics
    nlp_keywords = ["bleu", "rouge", "perplexity", "exact match", "word error"]
    if any(kw in metric_lower for kw in nlp_keywords):
        return "nlp"

    # Object detection/Image metrics
    image_keywords = ["iou", "map", "dice", "pixel", "intersection over union"]
    if any(kw in metric_lower for kw in image_keywords):
        return "computer_vision"

    return "unknown"

    metric_lower = evaluation_metric.lower()

    # Classification metrics
    classification_metrics = [
        "accuracy",
        "auc",
        "f1",
        "precision",
        "recall",
        "logloss",
        "log_loss",
        "roc_auc",
        "balanced_accuracy",
        "matthews",
        "cohen_kappa",
    ]
    if any(m in metric_lower for m in classification_metrics):
        return "classification"

    # Regression metrics
    regression_metrics = [
        "rmse",
        "mse",
        "mae",
        "r2",
        "rmsle",
        "mape",
        "smape",
        "root_mean_squared",
        "mean_absolute",
        "mean_squared",
    ]
    if any(m in metric_lower for m in regression_metrics):
        return "regression"

    # NLP/Text metrics
    nlp_metrics = ["bleu", "rouge", "perplexity", "f1_score", "exact_match"]
    if any(m in metric_lower for m in nlp_metrics):
        return "nlp"

    # Object detection/Image metrics
    image_metrics = ["iou", "map", "dice", "pixel"]
    if any(m in metric_lower for m in image_metrics):
        return "computer_vision"

    return "unknown"


def create_task_from_kernel(
    kernel_content: KernelContent,
    competition_info: CompetitionInfo,
    analysis: KaggleKernelAnalysis,
) -> dict:
    """
    Create final task JSON from kernel and LLM evaluation.

    Args:
        kernel_content: KernelContent with code
        competition_info: CompetitionInfo with competition details
        analysis: KaggleKernelAnalysis from LLM evaluation

    Returns:
        Dict in Kaggle task format
    """
    kernel_info = kernel_content.kernel_info

    # Create task_id - extract slug from ref (handles both full URLs and simple refs)
    comp_slug = (
        competition_info.ref.split("/")[-1] if "/" in competition_info.ref else competition_info.ref
    )
    task_id = f"kaggle_{comp_slug}_{kernel_info.slug}"

    task = {
        "task_id": task_id,
        "source": {
            "type": "kaggle_kernel",
            "competition": competition_info.ref,
            "competition_title": competition_info.title,
            "kernel_slug": kernel_info.slug,
            "kernel_ref": kernel_info.ref,
            "author": kernel_info.author,
            "votes": kernel_info.votes,
            "url": kernel_info.url,
            "language": kernel_info.language,
            "kernel_type": kernel_info.kernel_type,
        },
        "problem_statement": {
            "title": competition_info.title,
            "description": competition_info.description[:2000],
            "evaluation_metric": competition_info.evaluation_metric,
            "problem_type": analysis.problem_type,
            "instruction": analysis.instruction,
            "context": analysis.context,
        },
        "grading": analysis.grading
        if analysis.grading
        else {
            "metric": competition_info.evaluation_metric or "unknown",
            "metric_description": competition_info.description[:500]
            if competition_info.description
            else "",
            "problem_type": classify_problem_type(competition_info.evaluation_metric),
            "competition_slug": comp_slug,
            "data_available": True,
            "grading_method": "metric",
            "baseline_score": None,
        },
        "reference_solution": {
            "code": kernel_content.code,
            "format": kernel_content.metadata.get("format", "unknown"),
            "key_techniques": analysis.key_techniques,
        },
        "metadata": {
            "task_type": analysis.task_type.value,
            "difficulty": analysis.difficulty.value,
            "quality_score": analysis.quality_score,
            "mined_at": datetime.now(timezone.utc).isoformat(),
            "competition_type": competition_info.competition_type,
            "tags": competition_info.tags,
            "evaluation_criteria": analysis.evaluation_criteria,
            "potential_improvements": analysis.potential_improvements,
        },
    }

    return task


def get_mined_task_ids(output_dir: Path) -> tuple[set[str], dict[str, dict]]:
    """
    Get set of task IDs that have already been mined and rejected kernels.

    Returns:
        Tuple of (mined_task_ids, rejected_kernels_info)
    """
    task_ids = set()
    rejected_kernels = {}

    if output_dir.exists():
        for task_file in output_dir.glob("*.json"):
            task_id = task_file.stem
            task_ids.add(task_id)

    rejected_file = output_dir / "rejected_kernels.json"
    if rejected_file.exists():
        rejected_kernels = json.loads(rejected_file.read_text())

    return task_ids, rejected_kernels


def save_rejected_kernel(output_dir: Path, task_id: str, details: dict) -> None:
    """Save rejected kernel details to cache."""
    rejected_file = output_dir / "rejected_kernels.json"
    rejected_kernels = json.loads(rejected_file.read_text()) if rejected_file.exists() else {}

    rejected_kernels[task_id] = {
        **details,
        "rejected_at": datetime.now().isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_file.write_text(json.dumps(rejected_kernels, indent=2))


# =============================================================================
# Main CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Mine Kaggle R kernels for benchmarking tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available competitions
  python scripts/mine_kaggle.py --list-competitions

  # Mine R kernels from a specific competition
  python scripts/mine_kaggle.py --competition titanic --max-kernels 10

  # Mine multiple competitions at once
  python scripts/mine_kaggle.py --competitions titanic,house-prices-advanced-regression-techniques --max-kernels 5

  # Bulk collection without quality filtering (faster)
  python scripts/mine_kaggle.py --competition titanic --skip-quality-check --max-kernels 50

  # Mine top R kernels across ALL competitions
  python scripts/mine_kaggle.py --max-kernels 50 --min-votes 100

  # Mine top R kernels including tutorials (not just competition solutions)
  python scripts/mine_kaggle.py --max-kernels 20 --competition-kernels-only False

  # Dry run to see what would be mined
  python scripts/mine_kaggle.py --competition titanic --dry-run

  # Search for competitions with specific terms
  python scripts/mine_kaggle.py --list-competitions --search "house prices"
""",
    )

    # Competition selection
    parser.add_argument(
        "--competition",
        help="Competition slug to mine (e.g., 'titanic', 'house-prices-advanced-regression-techniques'). "
        "If not provided, searches for top R kernels across ALL competitions.",
    )
    parser.add_argument(
        "--competitions",
        type=str,
        help="Comma-separated list of competitions to mine (e.g., 'titanic,house-prices,spaceship-titanic')",
    )
    parser.add_argument(
        "--list-competitions",
        action="store_true",
        help="List available competitions and exit",
    )
    parser.add_argument(
        "--search",
        help="Search term for listing competitions",
    )
    parser.add_argument(
        "--category",
        help="Filter competitions by category (featured, research, recruitment, gettingStarted)",
    )

    # Kernel selection
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=10,
        help="Maximum number of kernels to mine (default: 10)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=5,
        help="Minimum votes for a kernel to be considered (default: 5)",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=7,
        help="Minimum LLM quality score (1-10, default: 7)",
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Skip LLM quality evaluation (faster, collect all)",
    )
    parser.add_argument(
        "--competition-kernels-only",
        type=lambda x: x.lower() not in ("false", "0", "no", "n"),
        default=True,
        help="When searching all kernels, only include those with competition_sources "
        "(actual competition solutions, not tutorials). Default: True. "
        "Set to False to include tutorials and non-competition kernels.",
    )
    parser.add_argument(
        "--prefer-scripts",
        type=lambda x: x.lower() not in ("false", "0", "no", "n"),
        default=True,
        help="Prefer script kernels over notebooks. Default: True. "
        "Scripts (.R, .Rmd) are preferred over notebooks (.ipynb) for cleaner code extraction.",
    )

    # Output options
    parser.add_argument(
        "--output",
        default="tasks/kaggle",
        help="Output directory for task JSON files (default: tasks/kaggle)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for evaluation (default: from configs/llm.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed debug info including LLM inputs/outputs",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between API calls (seconds)",
    )

    args = parser.parse_args()

    # Initialize Kaggle API client
    try:
        kaggle_client = KaggleAPIClient(verbose=args.verbose)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # List competitions mode
    if args.list_competitions:
        print("Fetching Kaggle competitions...")
        competitions = kaggle_client.list_competitions(
            search=args.search,
            category=args.category,
        )

        if not competitions:
            print("No competitions found.")
            sys.exit(0)

        print(f"\nFound {len(competitions)} competitions:\n")
        for comp in competitions:
            print(f"  {comp.ref}")
            print(f"    Title: {comp.title}")
            print(f"    Type: {comp.competition_type or 'Unknown'}")
            print(f"    Reward: {comp.reward or 'Knowledge'}")
            print(f"    URL: {comp.url}")
            print()

        sys.exit(0)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get already-mined task IDs
    mined_task_ids, rejected_kernels = get_mined_task_ids(output_dir)

    if mined_task_ids and not args.dry_run:
        print(f"Found {len(mined_task_ids)} previously mined tasks (will skip)")
    if rejected_kernels and not args.dry_run:
        print(f"Found {len(rejected_kernels)} previously rejected kernels (will skip)")

    # Statistics
    stats = {
        "kernels_found": 0,
        "already_mined": 0,
        "previously_rejected": 0,
        "filtered_low_votes": 0,
        "pull_errors": 0,
        "would_evaluate": 0,
        "tasks_created": 0,
        "tasks_rejected": 0,
    }

    # Initialize LLM judge
    judge = LLMTaskJudge(model=args.model, verbose=args.verbose)

    # Determine mode: specific competition(s) vs cross-competition
    if args.competitions:
        # Multiple competitions mode
        competition_list = [c.strip() for c in args.competitions.split(",")]
        print(f"Mining {len(competition_list)} competitions: {', '.join(competition_list)}")
        for competition_slug in competition_list:
            _mine_single_competition(
                kaggle_client=kaggle_client,
                judge=judge,
                competition_slug=competition_slug,
                args=args,
                output_dir=output_dir,
                mined_task_ids=mined_task_ids,
                rejected_kernels=rejected_kernels,
                stats=stats,
            )
    elif args.competition:
        # Single competition mode
        _mine_single_competition(
            kaggle_client=kaggle_client,
            judge=judge,
            competition_slug=args.competition,
            args=args,
            output_dir=output_dir,
            mined_task_ids=mined_task_ids,
            rejected_kernels=rejected_kernels,
            stats=stats,
        )
    else:
        # Cross-competition mode
        _mine_cross_competition(
            kaggle_client=kaggle_client,
            judge=judge,
            args=args,
            output_dir=output_dir,
            mined_task_ids=mined_task_ids,
            rejected_kernels=rejected_kernels,
            stats=stats,
        )


def _mine_single_competition(
    kaggle_client: KaggleAPIClient,
    judge: LLMTaskJudge,
    competition_slug: str,
    args: argparse.Namespace,
    output_dir: Path,
    mined_task_ids: set[str],
    rejected_kernels: dict[str, dict],
    stats: dict[str, int],
) -> None:
    """Mine kernels from a specific competition."""
    # Get competition info
    print(f"Fetching competition: {competition_slug}")
    competition_info = kaggle_client.get_competition(competition_slug)
    if not competition_info:
        print(f"Error: Competition '{competition_slug}' not found")
        sys.exit(1)

    assert competition_info is not None

    print(f"  Title: {competition_info.title}")
    print(f"  Evaluation: {competition_info.evaluation_metric or 'Not specified'}")
    print()

    # List R kernels for the competition
    print(f"Fetching R kernels for {competition_slug}...")
    kernels = kaggle_client.list_kernels(
        competition=competition_slug,
        language="r",
        sort_by="voteCount",
        page_size=args.max_kernels * 2,  # Fetch extra to account for filtering
    )

    print(f"Found {len(kernels)} R kernels")
    stats["kernels_found"] = len(kernels)

    print("\n" + "=" * 60)
    print("MINING KAGGLE KERNELS (Single Competition Mode)")
    print("=" * 60)

    processed = 0
    for kernel in kernels:
        if processed >= args.max_kernels:
            break

        print(f"\n  Kernel: {kernel.title[:60]}...")
        print(f"    Author: {kernel.author}, Votes: {kernel.votes}")

        # Generate task ID - sanitize competition slug
        comp_slug = competition_slug.split("/")[-1] if "/" in competition_slug else competition_slug
        task_id = f"kaggle_{comp_slug}_{kernel.slug}"

        # Process the kernel
        should_continue = _process_kernel(
            kernel=kernel,
            task_id=task_id,
            competition_info=competition_info,
            kaggle_client=kaggle_client,
            judge=judge,
            args=args,
            output_dir=output_dir,
            mined_task_ids=mined_task_ids,
            rejected_kernels=rejected_kernels,
            stats=stats,
        )

        if should_continue:
            processed += 1

    # Print summary
    _print_summary(args, stats, output_dir)


def _mine_cross_competition(
    kaggle_client: KaggleAPIClient,
    judge: LLMTaskJudge,
    args: argparse.Namespace,
    output_dir: Path,
    mined_task_ids: set[str],
    rejected_kernels: dict[str, dict],
    stats: dict[str, int],
) -> None:
    """Mine top R kernels across all competitions."""
    print("Fetching top R kernels across all of Kaggle...")
    if args.competition_kernels_only:
        print("(Filtering to competition solutions only)")
    else:
        print("(Including tutorials and non-competition kernels)")

    # Fetch kernels in batches until we have enough
    all_kernels = []
    page = 1
    batch_size = 50  # Fetch more per page for efficiency

    while len(all_kernels) < args.max_kernels * 3:  # Fetch extra to account for filtering
        print(f"  Fetching page {page}...")
        kernels = kaggle_client.list_top_r_kernels(
            sort_by="voteCount",
            page=page,
            page_size=batch_size,
            competition_kernels_only=args.competition_kernels_only,
            prefer_scripts=args.prefer_scripts,
        )

        if not kernels:
            print("  No more kernels found")
            break

        all_kernels.extend(kernels)
        page += 1

        # Safety limit
        if page > 20:
            print("  Reached page limit (20)")
            break

    print(f"\nFound {len(all_kernels)} R kernels")
    stats["kernels_found"] = len(all_kernels)

    print("\n" + "=" * 60)
    print("MINING KAGGLE KERNELS (Cross-Competition Mode)")
    print("=" * 60)

    # Cache for competition info to avoid repeated API calls
    competition_cache: dict[str, CompetitionInfo] = {}

    processed = 0
    for kernel in all_kernels:
        if processed >= args.max_kernels:
            break

        print(f"\n  Kernel: {kernel.title[:60]}...")
        print(f"    Author: {kernel.author}, Votes: {kernel.votes}")

        # Determine competition from kernel's competition_sources
        if not kernel.competition_sources:
            print("    -> Skipping (no competition sources)")
            continue

        # Use the first competition source for task_id
        primary_competition = kernel.competition_sources[0]

        # Generate task ID using competition from metadata
        comp_slug = (
            primary_competition.split("/")[-1]
            if "/" in primary_competition
            else primary_competition
        )
        task_id = f"kaggle_{comp_slug}_{kernel.slug}"

        # Get or fetch competition info
        if primary_competition not in competition_cache:
            print(f"    -> Fetching competition info for {primary_competition}...")
            comp_info = kaggle_client.get_competition(primary_competition)
            if comp_info:
                competition_cache[primary_competition] = comp_info
            else:
                # Create a minimal CompetitionInfo if fetch fails
                competition_cache[primary_competition] = CompetitionInfo(
                    ref=primary_competition,
                    title=primary_competition.replace("-", " ").title(),
                    description="",
                    evaluation_metric=None,
                    competition_type=None,
                    tags=[],
                    url=f"https://www.kaggle.com/competitions/{primary_competition}",
                )

        competition_info = competition_cache[primary_competition]

        # Process the kernel
        should_continue = _process_kernel(
            kernel=kernel,
            task_id=task_id,
            competition_info=competition_info,
            kaggle_client=kaggle_client,
            judge=judge,
            args=args,
            output_dir=output_dir,
            mined_task_ids=mined_task_ids,
            rejected_kernels=rejected_kernels,
            stats=stats,
        )

        if should_continue:
            processed += 1

    # Print summary
    _print_summary(args, stats, output_dir)


def _process_kernel(
    kernel: KernelInfo,
    task_id: str,
    competition_info: CompetitionInfo,
    kaggle_client: KaggleAPIClient,
    judge: LLMTaskJudge,
    args: argparse.Namespace,
    output_dir: Path,
    mined_task_ids: set[str],
    rejected_kernels: dict[str, dict],
    stats: dict[str, int],
) -> bool:
    """
    Process a single kernel: check filters, pull, evaluate, and save.

    Returns:
        True if the kernel was processed (regardless of outcome), False if skipped early.
    """
    # Skip if already mined
    if task_id in mined_task_ids:
        print("    -> Skipping (already mined)")
        stats["already_mined"] += 1
        return True

    # Skip if previously rejected
    if task_id in rejected_kernels:
        prev_rejection = rejected_kernels[task_id]
        print(
            f"    -> Skipping (previously rejected: {prev_rejection.get('rejection_reason', 'unknown')})"
        )
        stats["previously_rejected"] += 1
        return True

    # Check minimum votes
    if kernel.votes < args.min_votes:
        print(f"    -> Skipping (votes {kernel.votes} < {args.min_votes})")
        stats["filtered_low_votes"] += 1
        return True

    # Pull kernel content
    if args.dry_run:
        print("    -> Would pull and evaluate kernel")
        stats["would_evaluate"] += 1
        return True

    print("    -> Pulling kernel content...")
    kernel_content = kaggle_client.pull_kernel(kernel.ref)
    time.sleep(args.delay)

    if kernel_content is None:
        print("    -> Error pulling kernel")
        stats["pull_errors"] += 1
        return True

    # Update kernel_info with actual data from list
    kernel_content.kernel_info.votes = kernel.votes
    kernel_content.kernel_info.title = kernel.title

    # Evaluate with LLM (skip if --skip-quality-check is set)
    if not args.skip_quality_check:
        print("    -> Evaluating with LLM...")
        llm_schema = judge.evaluate_kernel(kernel_content, competition_info)

        # Check quality threshold
        if not llm_schema.is_good_task or llm_schema.quality_score < args.min_quality:
            rejection_reason = (
                llm_schema.rejection_reason
                or f"Quality score {llm_schema.quality_score} < {args.min_quality}"
            )
            print(f"    -> Rejected: {rejection_reason}")
            stats["tasks_rejected"] += 1

            save_rejected_kernel(
                output_dir,
                task_id,
                {
                    "quality_score": llm_schema.quality_score,
                    "task_type": str(llm_schema.task_type.value) if llm_schema.task_type else None,
                    "difficulty": str(llm_schema.difficulty.value)
                    if llm_schema.difficulty
                    else None,
                    "rejection_reason": rejection_reason,
                    "kernel_title": kernel.title,
                    "kernel_url": kernel.url,
                    "votes": kernel.votes,
                },
            )
            return True
    else:
        # Create a default schema when skipping quality check
        print("    -> Skipping LLM evaluation (--skip-quality-check)")
        comp_slug = (
            competition_info.ref.split("/")[-1]
            if competition_info.ref and "/" in competition_info.ref
            else competition_info.ref or ""
        )
        llm_schema = KaggleKernelAnalysis(
            task_type=TaskType.FEATURE_IMPL,
            difficulty=Difficulty.MEDIUM,
            quality_score=5,  # Default score when skipping evaluation
            instruction=f"Complete the {competition_info.title} competition task.",
            context=competition_info.description[:1000] if competition_info.description else "",
            is_good_task=True,
            problem_type=classify_problem_type(competition_info.evaluation_metric),
            evaluation_criteria=competition_info.evaluation_metric or "Not specified",
            grading={
                "metric": competition_info.evaluation_metric or "unknown",
                "metric_description": competition_info.description[:500]
                if competition_info.description
                else "",
                "problem_type": classify_problem_type(competition_info.evaluation_metric),
                "competition_slug": comp_slug,
                "data_available": True,
                "grading_method": "metric",
                "baseline_score": None,
            },
        )

    # Create task
    task = create_task_from_kernel(kernel_content, competition_info, llm_schema)

    # Save task
    task_file = output_dir / f"{task['task_id']}.json"
    with open(task_file, "w") as f:
        json.dump(task, f, indent=2)

    print(
        f"    -> Created task: {llm_schema.task_type.value} "
        f"(difficulty: {llm_schema.difficulty.value}, "
        f"quality: {llm_schema.quality_score}/10)"
    )
    stats["tasks_created"] += 1
    return True


def _print_summary(args: argparse.Namespace, stats: dict[str, int], output_dir: Path) -> None:
    """Print mining summary."""
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("MINING COMPLETE")
    print("=" * 60)
    print(f"Kernels found: {stats['kernels_found']}")
    if stats["already_mined"] > 0:
        print(f"Kernels skipped (already mined): {stats['already_mined']}")
    if stats["previously_rejected"] > 0:
        print(f"Kernels skipped (previously rejected): {stats['previously_rejected']}")
    if args.dry_run:
        print(f"Kernels filtered (low votes): {stats['filtered_low_votes']}")
        print(f"Kernels would be evaluated: {stats['would_evaluate']}")
        print("Tasks might be created: unknown (requires LLM evaluation)")
    else:
        print(f"Kernels filtered (low votes): {stats['filtered_low_votes']}")
        print(f"Kernel pull errors: {stats['pull_errors']}")
        print(f"Tasks created: {stats['tasks_created']}")
        print(f"Tasks rejected: {stats['tasks_rejected']}")
        print(f"\nOutput directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
