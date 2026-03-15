"""Shared pytest fixtures for grist-mill tests."""

from __future__ import annotations

# Legacy tests that import from pre-rewrite modules (bench/, evaluation/, scripts/,
# task_generator/, optimization/) — these depend on code outside the grist-mill
# package and cannot run in the grist-mill test suite.
collect_ignore = [
    "test_canonical_runner.py",
    "test_container_logs.py",
    "test_eval_prompt_builder.py",
    "test_evaluation_config.py",
    "test_experiment_config_behavior.py",
    "test_experiment_kaggle_scoring.py",
    "test_harness_integration.py",
    "test_mine_prs_llm_judge.py",
    "test_mined_task.py",
    "test_optimization.py",
    "test_pi_runner_auth_policy.py",
    "test_provider_env.py",
    "test_provider_inference.py",
    "test_provider_parallel_limits.py",
    "test_provider_preflight.py",
    "test_provider_resolver.py",
    "test_sandbox_docker.py",
    "test_schema_v1.py",
    "test_skill_policy.py",
    "test_task_generator.py",
    "test_telemetry_cutover.py",
    "test_config_import.py",
    "integration",
]
