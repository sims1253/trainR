#!/usr/bin/env python3
"""
Config Contract Validator

Validates cross-references between configuration files to catch model/config drift early.
Returns non-zero exit code on validation failures with actionable error messages.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ValidationError:
    """Represents a single validation error with context."""

    config_file: str
    field_path: str
    issue: str
    expected: str | None = None
    actual: str | None = None

    def __str__(self) -> str:
        msg = f"  [{self.config_file}] {self.field_path}: {self.issue}"
        if self.expected:
            msg += f"\n    Expected: {self.expected}"
        if self.actual:
            msg += f"\n    Actual: {self.actual}"
        return msg


@dataclass
class ValidationResult:
    """Aggregates validation errors."""

    errors: list[ValidationError] = field(default_factory=list)

    def add_error(self, error: ValidationError) -> None:
        self.errors.append(error)

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_report(self) -> None:
        if self.is_valid():
            print("All config contracts validated successfully.")
            return

        print(f"\n{'=' * 60}")
        print("CONFIG CONTRACT VALIDATION FAILED")
        print(f"{'=' * 60}")
        print(f"\nFound {len(self.errors)} error(s):\n")

        # Group errors by config file
        by_file: dict[str, list[ValidationError]] = {}
        for error in self.errors:
            by_file.setdefault(error.config_file, []).append(error)

        for config_file, errors in by_file.items():
            print(f"\n{config_file}:")
            for error in errors:
                print(str(error))

        print(f"\n{'=' * 60}")
        print("Fix the above issues before proceeding.")
        print(f"{'=' * 60}\n")


class ConfigValidator:
    """Validates configuration file contracts."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.result = ValidationResult()
        self._llm_config: dict[str, Any] | None = None
        self._llm_models: set[str] | None = None
        self._llm_providers: set[str] | None = None

    def load_yaml(self, filename: str) -> dict[str, Any] | None:
        """Load a YAML config file."""
        path = self.config_dir / filename
        if not path.exists():
            self.result.add_error(
                ValidationError(
                    config_file=filename,
                    field_path="file",
                    issue="Config file does not exist",
                    expected=str(path),
                )
            )
            return None

        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.result.add_error(
                ValidationError(
                    config_file=filename,
                    field_path="file",
                    issue=f"Invalid YAML syntax: {e}",
                )
            )
            return None

    def load_llm_config(self) -> dict[str, Any]:
        """Load and cache the LLM config."""
        if self._llm_config is None:
            self._llm_config = self.load_yaml("llm.yaml") or {}
        return self._llm_config

    def get_llm_models(self) -> set[str]:
        """Get set of valid model names from llm.yaml."""
        if self._llm_models is None:
            config = self.load_llm_config()
            models = config.get("models", {})
            self._llm_models = set(models.keys())
        return self._llm_models or set()

    def get_llm_providers(self) -> set[str]:
        """Get set of valid provider names from llm.yaml."""
        if self._llm_providers is None:
            config = self.load_llm_config()
            providers = config.get("providers", {})
            self._llm_providers = set(providers.keys())
        return self._llm_providers or set()

    def validate_llm_config(self) -> None:
        """Validate llm.yaml structure and internal references."""
        config = self.load_yaml("llm.yaml")
        if not config:
            return

        # Check required top-level sections
        required_sections = ["providers", "models", "defaults"]
        for section in required_sections:
            if section not in config:
                self.result.add_error(
                    ValidationError(
                        config_file="llm.yaml",
                        field_path=section,
                        issue="Required section missing",
                    )
                )

        # Validate models reference valid providers
        models = config.get("models", {})
        providers = self.get_llm_providers()

        for model_name, model_def in models.items():
            if not isinstance(model_def, dict):
                continue

            # Check single provider reference
            if "provider" in model_def:
                provider = model_def["provider"]
                if provider not in providers:
                    self.result.add_error(
                        ValidationError(
                            config_file="llm.yaml",
                            field_path=f"models.{model_name}.provider",
                            issue=f"Unknown provider '{provider}'",
                            expected=f"One of: {sorted(providers)}",
                            actual=provider,
                        )
                    )

            # Check multi-provider references
            if "providers" in model_def:
                for i, pdef in enumerate(model_def["providers"]):
                    if isinstance(pdef, dict) and "provider" in pdef:
                        provider = pdef["provider"]
                        if provider not in providers:
                            self.result.add_error(
                                ValidationError(
                                    config_file="llm.yaml",
                                    field_path=f"models.{model_name}.providers[{i}].provider",
                                    issue=f"Unknown provider '{provider}'",
                                    expected=f"One of: {sorted(providers)}",
                                    actual=provider,
                                )
                            )

        # Validate defaults reference valid models
        valid_models = self.get_llm_models()
        defaults = config.get("defaults", {})
        for purpose, model_name in defaults.items():
            if model_name not in valid_models:
                self.result.add_error(
                    ValidationError(
                        config_file="llm.yaml",
                        field_path=f"defaults.{purpose}",
                        issue=f"References unknown model '{model_name}'",
                        expected=f"One of: {sorted(valid_models)}",
                        actual=model_name,
                    )
                )

    def validate_benchmark_config(self) -> None:
        """Validate benchmark.yaml references to llm.yaml."""
        config = self.load_yaml("benchmark.yaml")
        if not config:
            return

        valid_models = self.get_llm_models()

        # Validate models list
        models = config.get("models", [])
        for i, model_name in enumerate(models):
            if model_name not in valid_models:
                self.result.add_error(
                    ValidationError(
                        config_file="benchmark.yaml",
                        field_path=f"models[{i}]",
                        issue=f"Model '{model_name}' not defined in llm.yaml",
                        expected=f"One of: {sorted(valid_models)}",
                        actual=model_name,
                    )
                )

        # Validate judge configuration
        judge = config.get("judge", {})

        # Single judge
        if "single" in judge:
            judge_model = judge["single"]
            if judge_model not in valid_models:
                self.result.add_error(
                    ValidationError(
                        config_file="benchmark.yaml",
                        field_path="judge.single",
                        issue=f"Judge model '{judge_model}' not defined in llm.yaml",
                        expected=f"One of: {sorted(valid_models)}",
                        actual=judge_model,
                    )
                )

        # Ensemble judges
        ensemble = judge.get("ensemble", {})
        judges = ensemble.get("judges", [])
        for i, judge_model in enumerate(judges):
            if judge_model not in valid_models:
                self.result.add_error(
                    ValidationError(
                        config_file="benchmark.yaml",
                        field_path=f"judge.ensemble.judges[{i}]",
                        issue=f"Judge model '{judge_model}' not defined in llm.yaml",
                        expected=f"One of: {sorted(valid_models)}",
                        actual=judge_model,
                    )
                )

        # Validate skill file reference
        skill = config.get("skill")
        if skill and isinstance(skill, str):
            self._validate_file_reference("benchmark.yaml", "skill", skill)

    def validate_evaluation_config(self, filename: str = "evaluation.yaml") -> None:
        """Validate evaluation.yaml or smoke.yaml structure."""
        config = self.load_yaml(filename)
        if not config:
            return

        # Validate required sections
        required_sections = ["model", "workers", "tasks", "execution", "output"]
        for section in required_sections:
            if section not in config:
                self.result.add_error(
                    ValidationError(
                        config_file=filename,
                        field_path=section,
                        issue="Required section missing",
                    )
                )

        # Validate model section
        model = config.get("model", {})
        for field in ["task", "reflection"]:
            if field not in model:
                self.result.add_error(
                    ValidationError(
                        config_file=filename,
                        field_path=f"model.{field}",
                        issue="Required field missing",
                    )
                )

        # Validate skill file reference
        skill = config.get("skill", {})
        if skill and isinstance(skill, dict):
            skill_path = skill.get("path")
            if skill_path and isinstance(skill_path, str):
                self._validate_file_reference(filename, "skill.path", skill_path)

    def validate_mining_config(self) -> None:
        """Validate mining.yaml structure."""
        config = self.load_yaml("mining.yaml")
        if not config:
            return

        # Validate required sections
        required_sections = ["repos", "settings", "quality_criteria"]
        for section in required_sections:
            if section not in config:
                self.result.add_error(
                    ValidationError(
                        config_file="mining.yaml",
                        field_path=section,
                        issue="Required section missing",
                    )
                )

        # Validate repos structure
        repos = config.get("repos", [])
        required_repo_fields = ["owner", "repo"]
        for i, repo in enumerate(repos):
            if not isinstance(repo, dict):
                continue
            for field in required_repo_fields:
                if field not in repo:
                    self.result.add_error(
                        ValidationError(
                            config_file="mining.yaml",
                            field_path=f"repos[{i}].{field}",
                            issue="Required field missing",
                        )
                    )

    def _validate_file_reference(self, config_file: str, field_path: str, file_path: str) -> None:
        """Validate that a referenced file exists."""
        # Resolve relative to project root
        full_path = self.config_dir.parent / file_path
        if not full_path.exists():
            self.result.add_error(
                ValidationError(
                    config_file=config_file,
                    field_path=field_path,
                    issue=f"Referenced file does not exist",
                    expected=str(full_path),
                    actual=file_path,
                )
            )

    def validate_all(self) -> ValidationResult:
        """Run all validations."""
        print("Validating config contracts...")
        print(f"  Config directory: {self.config_dir}")
        print()

        # Validate in order of dependencies
        print("  [1/5] Validating llm.yaml...")
        self.validate_llm_config()

        print("  [2/5] Validating benchmark.yaml...")
        self.validate_benchmark_config()

        print("  [3/5] Validating evaluation.yaml...")
        self.validate_evaluation_config("evaluation.yaml")

        print("  [4/5] Validating smoke.yaml...")
        self.validate_evaluation_config("smoke.yaml")

        print("  [5/5] Validating mining.yaml...")
        self.validate_mining_config()

        print()
        return self.result


def main() -> int:
    """Main entry point."""
    # Find config directory
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}", file=sys.stderr)
        return 1

    validator = ConfigValidator(config_dir)
    result = validator.validate_all()
    result.print_report()

    return 0 if result.is_valid() else 1


if __name__ == "__main__":
    sys.exit(main())
