"""Skill selection policy module for policy-driven selective skill access.

This module provides:
- SkillSelectionPolicy base class for implementing selection strategies
- HeuristicPolicy for rule-based skill selection
- KeywordMatchPolicy for keyword-based skill matching
- SelectionResult for tracking selected skills and rationale
"""

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import yaml


class PolicyType(str, Enum):
    """Types of skill selection policies."""

    HEURISTIC = "heuristic"
    KEYWORD = "keyword"
    EMBEDDING = "embedding"


@dataclass
class SkillMetadata:
    """Metadata about a skill for selection purposes."""

    skill_id: str
    name: str
    path: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    priority: int = 0
    enabled: bool = True

    @classmethod
    def from_skill_file(cls, path: Path) -> "SkillMetadata":
        """Parse skill metadata from a skill file.

        Args:
            path: Path to the skill markdown file

        Returns:
            SkillMetadata instance
        """
        content = path.read_text()
        skill_id = path.stem

        # Parse YAML front matter if present
        metadata: dict[str, Any] = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                with contextlib.suppress(yaml.YAMLError):
                    metadata = yaml.safe_load(parts[1]) or {}

        # Extract description from first paragraph after front matter
        description = metadata.get("description", "")
        if not description:
            # Try to extract from content
            body = content.split("---", 2)[-1].strip() if "---" in content else content
            lines = [
                line.strip()
                for line in body.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            if lines:
                description = lines[0][:200]

        return cls(
            skill_id=skill_id,
            name=metadata.get("name", skill_id),
            path=str(path),
            description=description,
            tags=metadata.get("tags", []),
            keywords=metadata.get("keywords", []),
            domains=metadata.get("domains", []),
            priority=metadata.get("priority", 0),
            enabled=metadata.get("enabled", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "tags": self.tags,
            "keywords": self.keywords,
            "domains": self.domains,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class SelectionRationale:
    """Rationale for selecting a single skill."""

    skill_id: str
    score: float
    reasons: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    matched_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "skill_id": self.skill_id,
            "score": self.score,
            "reasons": self.reasons,
            "matched_keywords": self.matched_keywords,
            "matched_tags": self.matched_tags,
        }


@dataclass
class SelectionResult:
    """Result of skill selection process."""

    selected_skills: list[str]
    selection_policy: str
    selection_rationale: list[SelectionRationale] = field(default_factory=list)
    config_hash: str = ""
    seed: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "selected_skills": self.selected_skills,
            "selection_policy": self.selection_policy,
            "selection_rationale": [r.to_dict() for r in self.selection_rationale],
            "config_hash": self.config_hash,
            "seed": self.seed,
            "created_at": self.created_at,
        }

    def get_ordered_skills(self) -> list[str]:
        """Get skills in deterministic order (sorted alphabetically)."""
        return sorted(self.selected_skills)


class SkillSelectionPolicy(ABC):
    """Base class for skill selection policies.

    Skill selection policies determine which skills from a collection
    are relevant for a given task based on various criteria.
    """

    policy_type: PolicyType

    def __init__(
        self,
        seed: int | None = None,
        max_skills: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Initialize the policy.

        Args:
            seed: Random seed for reproducibility
            max_skills: Maximum number of skills to select (None = no limit)
            min_score: Minimum score threshold for selection
        """
        self.seed = seed
        self.max_skills = max_skills
        self.min_score = min_score

    @abstractmethod
    def select(
        self,
        task_context: dict[str, Any],
        available_skills: list[SkillMetadata],
    ) -> SelectionResult:
        """Select relevant skills based on task context.

        Args:
            task_context: Task information including instruction, context, etc.
            available_skills: List of available skill metadata

        Returns:
            SelectionResult with selected skills and rationale
        """
        pass

    def _apply_limits(
        self,
        rationales: list[SelectionRationale],
    ) -> list[SelectionRationale]:
        """Apply min_score and max_skills limits to selection.

        Args:
            rationales: List of selection rationales

        Returns:
            Filtered and limited rationales
        """
        # Filter by minimum score
        filtered = [r for r in rationales if r.score >= self.min_score]

        # Sort by score (descending), then by skill_id for determinism
        filtered.sort(key=lambda r: (-r.score, r.skill_id))

        # Apply max skills limit
        if self.max_skills is not None:
            filtered = filtered[: self.max_skills]

        return filtered

    def _compute_config_hash(self, task_context: dict[str, Any]) -> str:
        """Compute a deterministic hash for the configuration.

        Args:
            task_context: Task context dictionary

        Returns:
            Hash string
        """
        from hashlib import sha256

        # Create deterministic string from relevant config
        components = [
            self.policy_type.value,
            str(self.seed),
            str(self.max_skills),
            str(self.min_score),
        ]

        # Add task identifiers
        if "task_id" in task_context:
            components.append(str(task_context["task_id"]))
        if "instruction" in task_context:
            # Hash instruction to keep it bounded
            instruction_hash = sha256(str(task_context["instruction"]).encode()).hexdigest()[:8]
            components.append(instruction_hash)

        combined = "|".join(components)
        return sha256(combined.encode()).hexdigest()[:16]


class HeuristicPolicy(SkillSelectionPolicy):
    """Heuristic-based skill selection policy.

    Uses rule-based matching to select skills based on task characteristics
    like domain, task type, and content patterns.
    """

    policy_type = PolicyType.HEURISTIC

    # Domain keyword mappings
    DOMAIN_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "testing": ["test", "testthat", "expect", "mock", "snapshot", "fixture"],
        "tidyverse": ["dplyr", "tidyr", "ggplot", "tibble", "purrr", "stringr"],
        "metaprogramming": ["rlang", "quote", "expr", "enquo", "tidyeval", "data-masking"],
        "cli": ["cli", "console", "message", "warning", "output"],
        "package": ["package", "description", "namespace", "install", "cran"],
        "performance": ["profile", "benchmark", "optimize", "vectorize", "parallel"],
    }

    # Task type patterns
    TASK_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "test_writing": ["write test", "add test", "test for", "testing"],
        "bug_fix": ["fix", "bug", "error", "issue", "broken"],
        "feature": ["implement", "add", "create", "new feature"],
        "refactor": ["refactor", "clean", "restructure", "improve"],
    }

    def __init__(
        self,
        seed: int | None = None,
        max_skills: int | None = None,
        min_score: float = 0.1,
        domain_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize heuristic policy.

        Args:
            seed: Random seed for reproducibility
            max_skills: Maximum skills to select
            min_score: Minimum score threshold
            domain_weights: Custom domain weight overrides
        """
        super().__init__(seed=seed, max_skills=max_skills, min_score=min_score)
        self.domain_weights = domain_weights or {}

    def select(
        self,
        task_context: dict[str, Any],
        available_skills: list[SkillMetadata],
    ) -> SelectionResult:
        """Select skills using heuristic rules.

        Args:
            task_context: Task information
            available_skills: Available skills

        Returns:
            SelectionResult with selected skills
        """
        rationales: list[SelectionRationale] = []

        # Combine task text for analysis
        instruction = str(task_context.get("instruction", "")).lower()
        context = str(task_context.get("context", "")).lower()
        source_package = str(task_context.get("source_package", "")).lower()
        combined_text = f"{instruction} {context} {source_package}"

        # Detect domains from task
        detected_domains = self._detect_domains(combined_text)

        # Detect task types
        detected_task_types = self._detect_task_types(combined_text)

        for skill in available_skills:
            if not skill.enabled:
                continue

            score = 0.0
            reasons: list[str] = []
            matched_keywords: list[str] = []
            matched_tags: list[str] = []

            # Score based on domain match
            for domain in skill.domains:
                if domain in detected_domains:
                    weight = self.domain_weights.get(domain, 1.0)
                    score += 0.3 * weight
                    reasons.append(f"Domain match: {domain}")

            # Score based on tag match
            for tag in skill.tags:
                tag_lower = tag.lower()
                if tag_lower in combined_text:
                    score += 0.2
                    matched_tags.append(tag)
                    reasons.append(f"Tag found in task: {tag}")

            # Score based on keyword match
            for keyword in skill.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in combined_text:
                    score += 0.15
                    matched_keywords.append(keyword)

            # Score based on skill name in task
            if skill.name.lower() in combined_text:
                score += 0.4
                reasons.append(f"Skill name mentioned: {skill.name}")

            # Score based on task type alignment
            if "testing" in detected_task_types and "testing" in skill.tags:
                score += 0.25
                reasons.append("Testing task with testing skill")

            # Boost for source package match
            if source_package and source_package in skill.tags:
                score += 0.3
                reasons.append(f"Source package in skill tags: {source_package}")

            if score > 0:
                rationales.append(
                    SelectionRationale(
                        skill_id=skill.skill_id,
                        score=min(score, 1.0),  # Cap at 1.0
                        reasons=reasons,
                        matched_keywords=matched_keywords,
                        matched_tags=matched_tags,
                    )
                )

        # Apply limits
        rationales = self._apply_limits(rationales)

        # Get ordered skill IDs
        selected_skills = [r.skill_id for r in rationales]

        return SelectionResult(
            selected_skills=selected_skills,
            selection_policy=self.policy_type.value,
            selection_rationale=rationales,
            config_hash=self._compute_config_hash(task_context),
            seed=self.seed,
        )

    def _detect_domains(self, text: str) -> set[str]:
        """Detect domains from text.

        Args:
            text: Text to analyze

        Returns:
            Set of detected domains
        """
        detected: set[str] = set()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    detected.add(domain)
                    break
        return detected

    def _detect_task_types(self, text: str) -> set[str]:
        """Detect task types from text.

        Args:
            text: Text to analyze

        Returns:
            Set of detected task types
        """
        detected: set[str] = set()
        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    detected.add(task_type)
                    break
        return detected


class KeywordMatchPolicy(SkillSelectionPolicy):
    """Keyword-based skill selection policy.

    Matches task content against skill keywords using exact and fuzzy matching.
    """

    policy_type = PolicyType.KEYWORD

    def __init__(
        self,
        seed: int | None = None,
        max_skills: int | None = None,
        min_score: float = 0.1,
        keyword_weight: float = 0.3,
        tag_weight: float = 0.3,
        name_weight: float = 0.4,
    ) -> None:
        """Initialize keyword policy.

        Args:
            seed: Random seed for reproducibility
            max_skills: Maximum skills to select
            min_score: Minimum score threshold
            keyword_weight: Weight for keyword matches
            tag_weight: Weight for tag matches
            name_weight: Weight for name matches
        """
        super().__init__(seed=seed, max_skills=max_skills, min_score=min_score)
        self.keyword_weight = keyword_weight
        self.tag_weight = tag_weight
        self.name_weight = name_weight

    def select(
        self,
        task_context: dict[str, Any],
        available_skills: list[SkillMetadata],
    ) -> SelectionResult:
        """Select skills using keyword matching.

        Args:
            task_context: Task information
            available_skills: Available skills

        Returns:
            SelectionResult with selected skills
        """
        rationales: list[SelectionRationale] = []

        # Extract searchable text from task
        instruction = str(task_context.get("instruction", ""))
        context = str(task_context.get("context", ""))
        source_package = str(task_context.get("source_package", ""))

        # Tokenize task text
        task_tokens = self._tokenize(f"{instruction} {context} {source_package}")

        for skill in available_skills:
            if not skill.enabled:
                continue

            score = 0.0
            reasons: list[str] = []
            matched_keywords: list[str] = []
            matched_tags: list[str] = []

            # Score keyword matches
            for keyword in skill.keywords:
                keyword_tokens = self._tokenize(keyword)
                for kw_token in keyword_tokens:
                    if kw_token in task_tokens:
                        score += self.keyword_weight / max(len(skill.keywords), 1)
                        matched_keywords.append(keyword)
                        reasons.append(f"Keyword match: {keyword}")
                        break

            # Score tag matches
            for tag in skill.tags:
                tag_tokens = self._tokenize(tag)
                for tag_token in tag_tokens:
                    if tag_token in task_tokens:
                        score += self.tag_weight / max(len(skill.tags), 1)
                        matched_tags.append(tag)
                        reasons.append(f"Tag match: {tag}")
                        break

            # Score name match
            name_tokens = self._tokenize(skill.name)
            name_matches = sum(1 for t in name_tokens if t in task_tokens)
            if name_matches > 0:
                name_score = self.name_weight * (name_matches / max(len(name_tokens), 1))
                score += name_score
                reasons.append(f"Name match ({name_matches}/{len(name_tokens)} tokens)")

            # Deduplicate matched lists
            matched_keywords = list(set(matched_keywords))
            matched_tags = list(set(matched_tags))

            if score > 0:
                rationales.append(
                    SelectionRationale(
                        skill_id=skill.skill_id,
                        score=min(score, 1.0),
                        reasons=reasons,
                        matched_keywords=matched_keywords,
                        matched_tags=matched_tags,
                    )
                )

        # Apply limits
        rationales = self._apply_limits(rationales)

        selected_skills = [r.skill_id for r in rationales]

        return SelectionResult(
            selected_skills=selected_skills,
            selection_policy=self.policy_type.value,
            selection_rationale=rationales,
            config_hash=self._compute_config_hash(task_context),
            seed=self.seed,
        )

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for matching.

        Args:
            text: Text to tokenize

        Returns:
            Set of lowercase tokens
        """
        import re

        # Split on non-alphanumeric, lowercase
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        return {t for t in tokens if len(t) >= 2}


def load_policy_from_config(config: dict[str, Any]) -> SkillSelectionPolicy:
    """Load a skill selection policy from configuration.

    Args:
        config: Policy configuration dictionary

    Returns:
        Configured SkillSelectionPolicy instance

    Raises:
        ValueError: If policy type is unknown
    """
    policy_type = config.get("type", "heuristic")

    seed_raw = config.get("seed")
    seed = seed_raw if isinstance(seed_raw, int) else None

    max_skills_raw = config.get("max_skills")
    max_skills = max_skills_raw if isinstance(max_skills_raw, int) else None

    min_score_raw = config.get("min_score", 0.0)
    min_score = float(min_score_raw) if isinstance(min_score_raw, (int, float)) else 0.0

    if policy_type == "heuristic":
        return HeuristicPolicy(
            seed=seed,
            max_skills=max_skills,
            min_score=min_score,
            domain_weights=config.get("domain_weights"),
        )
    elif policy_type == "keyword":
        keyword_weight_raw = config.get("keyword_weight", 0.3)
        tag_weight_raw = config.get("tag_weight", 0.3)
        name_weight_raw = config.get("name_weight", 0.4)
        return KeywordMatchPolicy(
            seed=seed,
            max_skills=max_skills,
            min_score=min_score,
            keyword_weight=float(keyword_weight_raw)
            if isinstance(keyword_weight_raw, (int, float))
            else 0.3,
            tag_weight=float(tag_weight_raw) if isinstance(tag_weight_raw, (int, float)) else 0.3,
            name_weight=float(name_weight_raw) if isinstance(name_weight_raw, (int, float)) else 0.4,
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def load_policy_from_yaml(path: str | Path, policy_name: str) -> SkillSelectionPolicy:
    """Load a named policy from a YAML configuration file.

    Args:
        path: Path to YAML configuration file
        policy_name: Name of the policy to load

    Returns:
        Configured SkillSelectionPolicy instance

    Raises:
        FileNotFoundError: If file not found
        KeyError: If policy name not in file
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    policies = config.get("policies", {})
    if policy_name not in policies:
        available = list(policies.keys())
        raise KeyError(f"Policy '{policy_name}' not found. Available: {available}")

    return load_policy_from_config(policies[policy_name])


def discover_skills(skills_dir: str | Path) -> list[SkillMetadata]:
    """Discover skills from a directory.

    Args:
        skills_dir: Directory containing skill files

    Returns:
        List of SkillMetadata for discovered skills
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []

    skills: list[SkillMetadata] = []
    for skill_file in skills_dir.glob("**/*.md"):
        try:
            metadata = SkillMetadata.from_skill_file(skill_file)
            if metadata.enabled:
                skills.append(metadata)
        except Exception:
            # Skip files that can't be parsed
            continue

    return skills
