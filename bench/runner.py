"""Canonical benchmark execution API.

This module provides the single canonical entry point for all benchmark execution
as specified in ARCH_REMEDIATION_PLAN.md Section 4.1.

The `run()` function is the library API that all other entrypoints (CLI, tests,
notebooks, web UI) should delegate to. This ensures:
- Consistent execution guarantees
- Proper artifact generation
- Unified telemetry and logging

Usage:
    from bench.runner import run

    # From config path
    manifest = run("configs/experiments/smoke.yaml")

    # From ExperimentConfig object
    config = ExperimentConfig.from_yaml("config.yaml")
    manifest = run(config)

    # With overrides
    manifest = run("config.yaml", output_dir="results/custom", seed=42)

The scripts/run_experiment.py CLI is a thin wrapper that delegates to this module.
"""

import logging
from pathlib import Path
from typing import Any

from bench.experiments import ExperimentConfig, ExperimentRunner, load_experiment_config
from bench.experiments.matrix import generate_matrix
from bench.schema.v1 import ManifestV1

logger = logging.getLogger(__name__)


def run(
    config: str | Path | ExperimentConfig,
    *,
    output_dir: str | None = None,
    seed: int | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    progress_callback: Any = None,
    **kwargs: Any,
) -> ManifestV1:
    """
    Canonical entry point for benchmark execution.

    This is the single supported runtime path for running benchmarks.
    All other entrypoints (CLI, tests, notebooks) should delegate to this function.

    Args:
        config: Path to experiment config YAML, or ExperimentConfig object.
            If a string or Path is provided, it will be loaded as YAML.
        output_dir: Override output directory for results. If not provided,
            uses the directory specified in the config.
        seed: Override random seed for reproducibility. If not provided,
            uses the seed from the config (if any).
        workers: Override number of parallel workers. If not provided,
            uses the value from the config.
        dry_run: If True, validate config and show experiment matrix without
            executing any tasks. Returns a minimal manifest with setup info.
        validate_only: If True, only validate the configuration without
            setting up or running anything. Returns an empty manifest.
        progress_callback: Optional callback function that receives progress updates
            during experiment execution. Called with a dict containing:
            - type: "start" or "result"
            - For "start": total_runs, models, task_count
            - For "result": model, task_id, passed, tokens, latency_s
        **kwargs: Additional override options. Supported keys:
            - timeout: Override per-task timeout in seconds
            - repeats: Override number of task repeats
            - save_trajectories: Override trajectory saving setting
            - save_container_logs: Persist raw container stdout/stderr logs
            - provider_parallel_limits: Override per-provider concurrency caps
            - provider_min_interval_s: Override per-provider run-start spacing
            - provider_max_requests_per_second: Override per-provider run-start rate cap

    Returns:
        ManifestV1 with run results and metadata. Contains:
        - Run identification (run_id, run_name)
        - Configuration fingerprints
        - Environment details
        - Per-model and overall summaries
        - References to output artifacts

    Raises:
        FileNotFoundError: If config path does not exist.
        ValueError: If configuration is invalid or missing required fields.
        RuntimeError: If experiment execution fails.

    Examples:
        >>> # Run from config path
        >>> manifest = run("configs/experiments/smoke.yaml")
        >>> print(f"Pass rate: {manifest.summary.pass_rate:.1%}")

        >>> # Run with ExperimentConfig object
        >>> config = ExperimentConfig.from_yaml("config.yaml")
        >>> manifest = run(config, seed=42, workers=2)

        >>> # Dry run to preview experiment matrix
        >>> manifest = run("config.yaml", dry_run=True)
        >>> print(f"Total runs: {manifest.task_count}")
    """
    # Type enforcement
    if not isinstance(config, (str, Path, ExperimentConfig)):
        raise TypeError(
            f"config must be str, Path, or ExperimentConfig, got {type(config).__name__}"
        )

    # Load or use provided config
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        experiment_config = load_experiment_config(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        experiment_config = config
        logger.info("Using provided ExperimentConfig object")

    # Apply overrides
    experiment_config = _apply_overrides(
        experiment_config,
        output_dir=output_dir,
        seed=seed,
        workers=workers,
        **kwargs,
    )

    # Validate only mode - return empty manifest
    if validate_only:
        logger.info("Validation only mode - skipping execution")
        return _create_validation_manifest(experiment_config)

    # Dry run mode - show matrix without executing
    if dry_run:
        logger.info("Dry run mode - showing experiment matrix")
        return _create_dry_run_manifest(experiment_config)

    # Create runner and execute
    runner = ExperimentRunner(experiment_config, progress_callback=progress_callback)

    # Setup phase
    logger.info("Setting up experiment...")
    runner.setup()

    # Execute phase
    logger.info("Starting experiment execution...")
    manifest = runner.run()

    logger.info(f"Experiment complete: {manifest.summary.completed} tasks")
    logger.info(f"Pass rate: {manifest.summary.pass_rate:.1%}")

    return manifest


def _apply_overrides(
    config: ExperimentConfig,
    *,
    output_dir: str | None = None,
    seed: int | None = None,
    workers: int | None = None,
    **kwargs: Any,
) -> ExperimentConfig:
    """
    Apply CLI/runtime overrides to the configuration.

    Creates a modified copy of the config with overrides applied.
    This ensures the original config object is not mutated.

    Args:
        config: Base configuration to modify
        output_dir: Override output directory
        seed: Override random seed
        workers: Override parallel worker count
        **kwargs: Additional overrides (timeout, repeats, etc.)

    Returns:
        Modified ExperimentConfig with overrides applied

    Raises:
        ValueError: If unknown override keys are provided
    """
    # Validate kwargs - only known override keys are allowed
    known_overrides = {
        "timeout",
        "repeats",
        "save_trajectories",
        "save_container_logs",
        "provider_parallel_limits",
        "provider_min_interval_s",
        "provider_max_requests_per_second",
    }
    unknown = set(kwargs.keys()) - known_overrides
    if unknown:
        raise ValueError(
            "Unknown override keys: "
            f"{unknown}. Supported: timeout, repeats, save_trajectories, "
            "save_container_logs, provider_parallel_limits, "
            "provider_min_interval_s, provider_max_requests_per_second"
        )

    # Create a copy to avoid mutating the original
    # Using model_dump and reconstruction for clean copy
    config_data = config.model_dump()

    if output_dir is not None:
        config_data["output"]["dir"] = output_dir
        logger.debug(f"Override: output_dir = {output_dir}")

    if seed is not None:
        config_data["determinism"]["seed"] = seed
        logger.debug(f"Override: seed = {seed}")

    if workers is not None:
        config_data["execution"]["parallel_workers"] = workers
        logger.debug(f"Override: workers = {workers}")

    # Handle additional kwargs
    if "timeout" in kwargs:
        config_data["execution"]["timeout"] = kwargs["timeout"]
        logger.debug(f"Override: timeout = {kwargs['timeout']}")

    if "repeats" in kwargs:
        config_data["execution"]["repeats"] = kwargs["repeats"]
        logger.debug(f"Override: repeats = {kwargs['repeats']}")

    if "save_trajectories" in kwargs:
        config_data["execution"]["save_trajectories"] = kwargs["save_trajectories"]
        logger.debug(f"Override: save_trajectories = {kwargs['save_trajectories']}")

    if "save_container_logs" in kwargs:
        config_data["execution"]["save_container_logs"] = kwargs["save_container_logs"]
        logger.debug(f"Override: save_container_logs = {kwargs['save_container_logs']}")

    if "provider_parallel_limits" in kwargs:
        config_data["execution"]["provider_parallel_limits"] = kwargs["provider_parallel_limits"]
        logger.debug(
            "Override: provider_parallel_limits = %s",
            kwargs["provider_parallel_limits"],
        )

    if "provider_min_interval_s" in kwargs:
        config_data["execution"]["provider_min_interval_s"] = kwargs["provider_min_interval_s"]
        logger.debug(
            "Override: provider_min_interval_s = %s",
            kwargs["provider_min_interval_s"],
        )

    if "provider_max_requests_per_second" in kwargs:
        config_data["execution"]["provider_max_requests_per_second"] = kwargs[
            "provider_max_requests_per_second"
        ]
        logger.debug(
            "Override: provider_max_requests_per_second = %s",
            kwargs["provider_max_requests_per_second"],
        )

    return ExperimentConfig.from_dict(config_data)


def _create_validation_manifest(config: ExperimentConfig) -> ManifestV1:
    """
    Create a minimal manifest for validation-only mode.

    Args:
        config: Validated configuration

    Returns:
        ManifestV1 with validation metadata but no results
    """
    return ManifestV1(
        run_id=config.generate_run_id(),
        run_name=config.name,
        models=config.models.names,
        task_count=0,
        skill_version=config.skill.get_name(),
        config={"validation_only": True, "name": config.name},
    )


def _create_dry_run_manifest(config: ExperimentConfig) -> ManifestV1:
    """
    Create a manifest for dry-run mode with experiment matrix info.

    Args:
        config: Configuration to analyze

    Returns:
        ManifestV1 with matrix metadata but no execution results
    """
    # Generate matrix to get task/model counts
    matrix = generate_matrix(config)

    return ManifestV1(
        run_id=config.generate_run_id(),
        run_name=config.name,
        models=[m.name for m in matrix.models],
        task_count=len(matrix.tasks),
        skill_version=config.skill.get_name(),
        config={
            "dry_run": True,
            "name": config.name,
            "total_runs": len(matrix.runs),
            "repeats": config.execution.repeats,
            "task_ids": [t.task_id for t in matrix.tasks[:10]],  # First 10 task IDs
        },
    )


# Convenience exports for common patterns
__all__ = [
    "run",
]
