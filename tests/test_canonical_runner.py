"""Regression tests verifying all supported entrypoints delegate to the canonical runner API.

This test module ensures:
1. bench.runner.run() is the canonical API for benchmark execution
2. scripts/run_experiment.py delegates to bench.runner.run()
3. Legacy scripts hard-fail with helpful migration messages
4. posit_gskill evaluate uses the canonical runner

These tests are fast unit tests that mock actual benchmark execution.
"""

import contextlib
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCanonicalRunnerAPI:
    """Tests verifying bench.runner.run() is the canonical API."""

    def test_run_function_exists(self):
        """Test that bench.runner.run is importable and callable."""
        from bench.runner import run

        assert callable(run)

    def test_run_function_signature_accepts_str(self):
        """Test that run() accepts a string path argument."""
        from bench.runner import run

        # Create a mock config file path
        with patch("bench.runner.load_experiment_config") as mock_load:
            with patch("bench.runner._apply_overrides") as mock_apply:
                with patch("bench.runner._create_dry_run_manifest") as mock_dry_run:
                    mock_config = MagicMock()
                    mock_config.name = "test"
                    mock_config.generate_run_id.return_value = "test-run"
                    mock_config.skill.get_name.return_value = "test-skill"
                    mock_load.return_value = mock_config
                    mock_apply.return_value = mock_config
                    mock_dry_run.return_value = MagicMock()

                    # Should not raise - string path accepted
                    with patch.object(Path, "exists", return_value=True):
                        run("configs/test.yaml", dry_run=True)

                    mock_load.assert_called_once()

    def test_run_function_signature_accepts_path(self):
        """Test that run() accepts a pathlib.Path argument."""
        from bench.runner import run

        with patch("bench.runner.load_experiment_config") as mock_load:
            with patch("bench.runner._apply_overrides") as mock_apply:
                with patch("bench.runner._create_dry_run_manifest") as mock_dry_run:
                    mock_config = MagicMock()
                    mock_config.name = "test"
                    mock_config.generate_run_id.return_value = "test-run"
                    mock_config.skill.get_name.return_value = "test-skill"
                    mock_load.return_value = mock_config
                    mock_apply.return_value = mock_config
                    mock_dry_run.return_value = MagicMock()

                    # Should not raise - Path accepted
                    with patch.object(Path, "exists", return_value=True):
                        run(Path("configs/test.yaml"), dry_run=True)

                    mock_load.assert_called_once()

    def test_run_function_signature_accepts_config_object(self):
        """Test that run() accepts an ExperimentConfig object."""
        from bench.experiments import (
            DeterminismConfig,
            ExecutionConfig,
            ExperimentConfig,
            ModelsConfig,
            OutputConfig,
            SkillConfig,
            TasksConfig,
            TaskSelectionMode,
        )
        from bench.runner import run

        # Create a minimal valid config to pass the isinstance check
        config = ExperimentConfig(
            name="test-experiment",
            tasks=TasksConfig(
                selection=TaskSelectionMode.SPLITS,
                splits=["dev"],
            ),
            models=ModelsConfig(names=["test-model"]),
            skill=SkillConfig(no_skill=True),
            execution=ExecutionConfig(timeout=60),
            output=OutputConfig(dir="results/test"),
            determinism=DeterminismConfig(seed=42),
        )

        with patch("bench.runner._apply_overrides") as mock_apply:
            with patch("bench.runner._create_dry_run_manifest") as mock_dry_run:
                mock_apply.return_value = config
                mock_dry_run.return_value = MagicMock()

                # Should not raise - ExperimentConfig accepted
                run(config, dry_run=True)

    def test_run_returns_manifest_v1(self):
        """Test that run() returns a ManifestV1 instance."""
        from bench.experiments import ExperimentConfig
        from bench.runner import run
        from bench.schema.v1 import ManifestV1

        with patch("bench.runner._apply_overrides") as mock_apply:
            with patch("bench.runner._create_validation_manifest") as mock_validate:
                mock_config = MagicMock(spec=ExperimentConfig)
                mock_config.name = "test"
                mock_apply.return_value = mock_config

                # Create a real ManifestV1 for the return
                expected_manifest = ManifestV1(
                    run_id="test-run",
                    run_name="test",
                    models=["test-model"],
                    task_count=0,
                    skill_version="test-skill",
                )
                mock_validate.return_value = expected_manifest

                result = run(mock_config, validate_only=True)

                assert isinstance(result, ManifestV1)
                assert result.run_id == "test-run"

    def test_run_raises_filenotfound_for_missing_config(self):
        """Test that run() raises FileNotFoundError for missing config file."""
        from bench.runner import run

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            run("nonexistent_config.yaml")

    def test_run_supports_dry_run_mode(self):
        """Test that run() supports dry_run mode."""
        from bench.experiments import ExperimentConfig
        from bench.runner import run
        from bench.schema.v1 import ManifestV1

        with patch("bench.runner._apply_overrides") as mock_apply:
            with patch("bench.runner._create_dry_run_manifest") as mock_dry_run:
                mock_config = MagicMock(spec=ExperimentConfig)
                mock_apply.return_value = mock_config

                expected_manifest = ManifestV1(
                    run_id="test-run",
                    models=["test-model"],
                    task_count=0,
                )
                mock_dry_run.return_value = expected_manifest

                result = run(mock_config, dry_run=True)

                mock_dry_run.assert_called_once_with(mock_config)
                assert isinstance(result, ManifestV1)

    def test_run_supports_validate_only_mode(self):
        """Test that run() supports validate_only mode."""
        from bench.experiments import ExperimentConfig
        from bench.runner import run
        from bench.schema.v1 import ManifestV1

        with patch("bench.runner._apply_overrides") as mock_apply:
            with patch("bench.runner._create_validation_manifest") as mock_validate:
                mock_config = MagicMock(spec=ExperimentConfig)
                mock_apply.return_value = mock_config

                expected_manifest = ManifestV1(
                    run_id="test-run",
                    models=[],
                    task_count=0,
                )
                mock_validate.return_value = expected_manifest

                result = run(mock_config, validate_only=True)

                mock_validate.assert_called_once_with(mock_config)
                assert isinstance(result, ManifestV1)


class TestRunExperimentScriptDelegation:
    """Tests verifying scripts/run_experiment.py delegates to bench.runner.run()."""

    @patch("bench.runner.run")
    @patch("scripts.run_experiment.validate_config")
    @patch("scripts.run_experiment.print_config_summary")
    def test_main_passes_overrides_to_runner(self, mock_print_summary, mock_validate, mock_run):
        """Test that CLI overrides are passed to bench.runner.run()."""
        from bench.schema.v1 import ManifestV1

        mock_config = MagicMock()
        mock_config.output.dir = "default"
        mock_config.determinism.seed = None
        mock_config.execution.parallel_workers = 1
        mock_validate.return_value = mock_config

        mock_manifest = MagicMock(spec=ManifestV1)
        mock_manifest.results_path = None
        mock_run.return_value = mock_manifest

        with (
            patch(
                "sys.argv",
                [
                    "run_experiment.py",
                    "--config",
                    "test.yaml",
                    "--output-dir",
                    "custom/output",
                    "--seed",
                    "42",
                    "--workers",
                    "4",
                ],
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            from scripts.run_experiment import main

            with contextlib.suppress(SystemExit):
                main()

        # Verify run was called with the overrides
        mock_run.assert_called()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("output_dir") == "custom/output"
        assert call_kwargs.get("seed") == 42
        assert call_kwargs.get("workers") == 4
        assert "verbose" not in call_kwargs

    @patch("bench.runner.run")
    @patch("scripts.run_experiment.validate_config")
    def test_validate_flag_calls_runner_with_validate_only(self, mock_validate, mock_run):
        """Test that --validate flag calls run() with validate_only=True."""
        mock_config = MagicMock()
        mock_config.output.dir = "default"
        mock_config.determinism.seed = None
        mock_config.execution.parallel_workers = 1
        mock_validate.return_value = mock_config

        with (
            patch(
                "sys.argv",
                ["run_experiment.py", "--config", "test.yaml", "--validate"],
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            from scripts.run_experiment import main

            with contextlib.suppress(SystemExit):
                main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("validate_only") is True

    @patch("bench.runner.run")
    @patch("scripts.run_experiment.validate_config")
    def test_dry_run_flag_calls_runner_with_dry_run(self, mock_validate, mock_run):
        """Test that --dry-run flag calls run() with dry_run=True."""
        mock_config = MagicMock()
        mock_config.output.dir = "default"
        mock_config.determinism.seed = None
        mock_config.execution.parallel_workers = 1
        mock_validate.return_value = mock_config

        mock_manifest = MagicMock()
        mock_manifest.task_count = 5
        mock_manifest.models = ["model1"]
        mock_manifest.config = {"total_runs": 5, "task_ids": ["t1", "t2"]}
        mock_run.return_value = mock_manifest

        with (
            patch(
                "sys.argv",
                ["run_experiment.py", "--config", "test.yaml", "--dry-run"],
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            from scripts.run_experiment import main

            with contextlib.suppress(SystemExit):
                main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("dry_run") is True


class TestLegacyScriptsHardFail:
    """Tests verifying legacy scripts exit with code 1 and helpful messages."""

    def test_run_benchmark_hard_fails(self):
        """Test that run_benchmark.py exits with code 1 and mentions canonical path."""
        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_benchmark.py"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 1
        # Check stderr contains migration message
        assert "deprecated" in result.stderr.lower() or "error" in result.stderr.lower()
        assert "run_experiment.py" in result.stderr

    def test_mini_benchmark_hard_fails(self):
        """Test that mini_benchmark.py exits with code 1 and mentions canonical path."""
        result = subprocess.run(
            ["uv", "run", "python", "scripts/mini_benchmark.py"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 1
        assert "deprecated" in result.stderr.lower() or "error" in result.stderr.lower()
        assert "run_experiment.py" in result.stderr

    def test_run_parallel_benchmark_hard_fails(self):
        """Test that run_parallel_benchmark.py exits with code 1 and mentions canonical path."""
        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_parallel_benchmark.py"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode == 1
        assert "deprecated" in result.stderr.lower() or "error" in result.stderr.lower()
        assert "run_experiment.py" in result.stderr

    def test_run_benchmark_message_mentions_config(self):
        """Test that run_benchmark.py error message mentions config file usage."""
        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_benchmark.py"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        # The migration message should mention using --config
        assert "--config" in result.stderr

    def test_all_legacy_scripts_consistent_messaging(self):
        """Test that all legacy scripts have consistent migration messaging."""
        legacy_scripts = [
            "scripts/run_benchmark.py",
            "scripts/mini_benchmark.py",
            "scripts/run_parallel_benchmark.py",
        ]

        for script in legacy_scripts:
            result = subprocess.run(
                ["uv", "run", "python", script],
                capture_output=True,
                text=True,
                cwd=project_root,
            )

            # All should exit with code 1
            assert result.returncode == 1, f"{script} should exit with code 1"

            # All should mention the canonical runner
            assert "run_experiment.py" in result.stderr, (
                f"{script} should mention run_experiment.py"
            )

            # All should mention --config flag
            assert "--config" in result.stderr, f"{script} should mention --config flag"


class TestPositGskillEvaluateUsesCanonicalRunner:
    """Tests verifying posit_gskill evaluate command uses bench.runner.run()."""

    @patch("bench.runner.run")
    @patch("config.get_llm_config")
    @patch.object(Path, "exists")
    def test_evaluate_calls_canonical_runner(self, mock_exists, mock_get_llm_config, mock_run):
        """Test that run_evaluate() calls bench.runner.run()."""
        from bench.schema.v1 import ResultSummaryV1

        # Setup mocks
        mock_exists.return_value = True

        mock_llm_config = MagicMock()
        mock_llm_config.get_default_model.return_value = "test-model"
        mock_get_llm_config.return_value = mock_llm_config

        mock_manifest = MagicMock()
        mock_manifest.summary = ResultSummaryV1(
            total_tasks=1,
            completed=1,
            passed=1,
            avg_score=1.0,
            avg_latency_s=1.0,
        )
        mock_manifest.results = []
        mock_manifest.results_path = "/results/test"
        mock_run.return_value = mock_manifest

        # Create args mock
        args = MagicMock()
        args.skill = "skills/test.md"
        args.tasks_dir = "tasks"

        from posit_gskill.commands import run_evaluate

        run_evaluate(args)

        # Verify bench.runner.run was called
        mock_run.assert_called_once()

        # Verify it was called with an ExperimentConfig
        call_args = mock_run.call_args[0]
        from bench.experiments import ExperimentConfig

        assert isinstance(call_args[0], ExperimentConfig)

    @patch("bench.runner.run")
    @patch("config.get_llm_config")
    @patch.object(Path, "exists")
    def test_evaluate_uses_skill_path_from_args(self, mock_exists, mock_get_llm_config, mock_run):
        """Test that run_evaluate() passes skill path to config."""
        from bench.schema.v1 import ResultSummaryV1

        mock_exists.return_value = True

        mock_llm_config = MagicMock()
        mock_llm_config.get_default_model.return_value = "test-model"
        mock_get_llm_config.return_value = mock_llm_config

        mock_manifest = MagicMock()
        mock_manifest.summary = ResultSummaryV1()
        mock_manifest.results = []
        mock_manifest.results_path = None
        mock_run.return_value = mock_manifest

        args = MagicMock()
        args.skill = "skills/custom_skill.md"
        args.tasks_dir = "custom_tasks"

        from posit_gskill.commands import run_evaluate

        run_evaluate(args)

        # Verify the config passed to run has the correct skill path
        call_args = mock_run.call_args[0]
        config = call_args[0]
        assert config.skill.path == "skills/custom_skill.md"
        assert config.tasks.dir == "custom_tasks"

    @patch("bench.runner.run")
    @patch("config.get_llm_config")
    @patch.object(Path, "exists")
    def test_evaluate_exits_on_missing_skill_file(self, mock_exists, mock_get_llm_config, mock_run):
        """Test that run_evaluate() exits if skill file doesn't exist."""
        # Skill file doesn't exist
        mock_exists.return_value = False

        args = MagicMock()
        args.skill = "skills/nonexistent.md"
        args.tasks_dir = "tasks"

        from posit_gskill.commands import run_evaluate

        with pytest.raises(SystemExit) as exc_info:
            run_evaluate(args)

        assert exc_info.value.code == 1
        # Should not have called the runner
        mock_run.assert_not_called()

    @patch("bench.runner.run")
    @patch("config.get_llm_config")
    @patch.object(Path, "exists")
    def test_evaluate_exits_on_missing_tasks_dir(self, mock_exists, mock_get_llm_config, mock_run):
        """Test that run_evaluate() exits if tasks directory doesn't exist."""
        # First call (skill path) returns True, second call (tasks_dir) returns False
        mock_exists.side_effect = [True, False]

        args = MagicMock()
        args.skill = "skills/test.md"
        args.tasks_dir = "nonexistent_tasks"

        from posit_gskill.commands import run_evaluate

        with pytest.raises(SystemExit) as exc_info:
            run_evaluate(args)

        assert exc_info.value.code == 1
        mock_run.assert_not_called()

    @patch("bench.runner.run")
    @patch("config.get_llm_config")
    @patch.object(Path, "exists")
    def test_evaluate_propagates_runner_errors(self, mock_exists, mock_get_llm_config, mock_run):
        """Test that run_evaluate() propagates errors from bench.runner.run()."""
        mock_exists.return_value = True

        mock_llm_config = MagicMock()
        mock_llm_config.get_default_model.return_value = "test-model"
        mock_get_llm_config.return_value = mock_llm_config

        # Simulate runner raising an error
        mock_run.side_effect = RuntimeError("Benchmark execution failed")

        args = MagicMock()
        args.skill = "skills/test.md"
        args.tasks_dir = "tasks"
        args.verbose = False

        from posit_gskill.commands import run_evaluate

        with pytest.raises(SystemExit) as exc_info:
            run_evaluate(args)

        assert exc_info.value.code == 1


class TestCanonicalRunnerExports:
    """Tests verifying the canonical runner exports are correct."""

    def test_runner_module_exports_run(self):
        """Test that bench.runner exports run function."""
        import bench.runner

        assert hasattr(bench.runner, "run")
        assert callable(bench.runner.run)

    def test_runner_all_exports(self):
        """Test that __all__ contains expected exports."""
        import bench.runner

        assert "run" in bench.runner.__all__

    def test_runner_module_docstring(self):
        """Test that runner module has documentation about canonical API."""
        import bench.runner

        assert bench.runner.__doc__ is not None
        assert "canonical" in bench.runner.__doc__.lower()


class TestRunnerIntegration:
    """Integration tests for the canonical runner API."""

    def test_runner_with_mock_experiment_config(self):
        """Test runner with a fully mocked ExperimentConfig."""
        from bench.experiments import (
            DeterminismConfig,
            ExecutionConfig,
            ExperimentConfig,
            ModelsConfig,
            OutputConfig,
            SkillConfig,
            TasksConfig,
            TaskSelectionMode,
        )
        from bench.runner import run
        from bench.schema.v1 import ManifestV1

        # Create a minimal valid config
        config = ExperimentConfig(
            name="test-experiment",
            tasks=TasksConfig(
                selection=TaskSelectionMode.SPLITS,
                splits=["dev"],
            ),
            models=ModelsConfig(names=["test-model"]),
            skill=SkillConfig(no_skill=True),
            execution=ExecutionConfig(timeout=60),
            output=OutputConfig(dir="results/test"),
            determinism=DeterminismConfig(seed=42),
        )

        with patch("bench.runner._apply_overrides") as mock_apply:
            with patch("bench.runner._create_validation_manifest") as mock_validate:
                mock_apply.return_value = config

                expected_manifest = ManifestV1(
                    run_id="test-run",
                    run_name="test-experiment",
                    models=["test-model"],
                    task_count=0,
                    skill_version="no_skill",
                )
                mock_validate.return_value = expected_manifest

                result = run(config, validate_only=True)

                assert isinstance(result, ManifestV1)
                mock_validate.assert_called_once()


class TestGuardNoBypassPaths:
    """Guard tests verifying no bypass paths exist for direct Docker/provider logic.

    These are static analysis tests that scan source files to ensure:
    1. No direct DockerTestRunner imports in production code
    2. No direct provider logic bypass in scripts
    3. No ExperimentRunner direct usage in CLI
    4. Canonical path enforcement through bench.runner.run()
    """

    # Directories to exclude from scanning
    EXCLUDED_DIRS = {
        ".venv",
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "build",
        "dist",
    }

    # Files that are allowed to use low-level components directly
    # These are test utilities, development scripts, or the canonical runner itself
    ALLOWED_DIRECT_IMPORTS = {
        # Test files can import anything
        "tests/",
        # The canonical runner itself uses ExperimentRunner internally
        "bench/runner.py",
        "bench/experiments/runner.py",
        "bench/experiments/__init__.py",
        # Evaluation sandbox is the abstraction layer
        "evaluation/__init__.py",
        "evaluation/sandbox.py",
        "evaluation/pi_runner.py",
        # Development/utility scripts for testing the runner directly
        "scripts/test_docker_pi_runner.py",
        # Optimization uses sandbox directly for single-task evaluation
        "optimization/adapter.py",
        # Legacy run_evaluation.py is a development utility, not canonical path
        "scripts/run_evaluation.py",
        # Tests mocking the sandbox
        "tests/test_optimization.py",
        "tests/test_evaluation_config.py",
        # GEPA adapter for optimization
        "bench/optimize/gepa_adapter.py",
    }

    def _get_python_files(self, base_path: Path) -> list[Path]:
        """Get all Python files in the project, excluding virtual environments."""
        python_files = []
        for path in base_path.rglob("*.py"):
            # Check if any excluded dir is in the path
            relative = path.relative_to(base_path)
            if any(part in self.EXCLUDED_DIRS for part in relative.parts):
                continue
            python_files.append(path)
        return sorted(python_files)

    def _is_allowed_path(self, file_path: Path, base_path: Path) -> bool:
        """Check if the file path is allowed to have direct imports."""
        relative = str(file_path.relative_to(base_path))
        for allowed in self.ALLOWED_DIRECT_IMPORTS:
            if relative.startswith(allowed) or relative == allowed.rstrip("/"):
                return True
        return False

    def test_no_docker_test_runner_in_production(self):
        """Guard: No DockerTestRunner imports in production code.

        DockerTestRunner was the old name for the test runner. It should
        not exist anywhere in the codebase as it has been replaced by
        DockerPiRunner and the canonical bench.runner.run() path.
        """
        python_files = self._get_python_files(project_root)

        for file_path in python_files:
            # Skip test files - they may reference DockerTestRunner in comments/assertions
            if self._is_allowed_path(file_path, project_root):
                continue

            content = file_path.read_text()
            # Check for DockerTestRunner (old name)
            assert "DockerTestRunner" not in content, (
                f"Found 'DockerTestRunner' in {file_path.relative_to(project_root)}. "
                "This old class name should not exist. Use bench.runner.run() instead."
            )

    def test_no_direct_pi_runner_in_run_script(self):
        """Guard: run_experiment.py doesn't bypass canonical runner.

        scripts/run_experiment.py must use bench.runner.run() for all
        execution, not directly import DockerPiRunner or DockerPiRunnerConfig.
        """
        run_script = project_root / "scripts" / "run_experiment.py"
        assert run_script.exists(), "scripts/run_experiment.py should exist"

        content = run_script.read_text()

        # Should not import DockerPiRunner directly
        assert "from evaluation" not in content or "DockerPiRunner" not in content, (
            "scripts/run_experiment.py should not import DockerPiRunner directly. "
            "Use bench.runner.run() instead."
        )

        # Should import bench.runner
        assert "import bench.runner" in content or "from bench.runner import" in content, (
            "scripts/run_experiment.py should import bench.runner for execution."
        )

    def test_no_experiment_runner_direct_in_cli(self):
        """Guard: CLI uses bench.runner.run() not ExperimentRunner directly.

        scripts/run_experiment.py must delegate through bench.runner.run(),
        not directly instantiate ExperimentRunner.
        """
        run_script = project_root / "scripts" / "run_experiment.py"
        content = run_script.read_text()

        # Should not import ExperimentRunner directly
        assert "ExperimentRunner" not in content, (
            "scripts/run_experiment.py should not import ExperimentRunner directly. "
            "Use bench.runner.run() which internally manages ExperimentRunner."
        )

    def test_canonical_runner_is_single_entrypoint(self):
        """Guard: All benchmark execution goes through bench.runner.run().

        Verify that bench.runner exports 'run' as the canonical entrypoint
        and that it's properly documented.
        """
        import bench.runner

        # Verify 'run' is exported
        assert hasattr(bench.runner, "run"), "bench.runner must export 'run' function"
        assert callable(bench.runner.run), "bench.runner.run must be callable"

        # Verify __all__ contains 'run'
        assert hasattr(bench.runner, "__all__"), "bench.runner must define __all__"
        assert "run" in bench.runner.__all__, "bench.runner.__all__ must contain 'run'"

        # Verify 'run' is the only public function in __all__ (except internal ones)
        assert bench.runner.__all__ == ["run"], (
            f"bench.runner.__all__ should only contain ['run'], got {bench.runner.__all__}"
        )

    def test_no_direct_sandbox_in_run_script(self):
        """Guard: run_experiment.py doesn't use EvaluationSandbox directly.

        scripts/run_experiment.py must use bench.runner.run() which
        internally handles sandbox creation, not create EvaluationSandbox directly.
        """
        run_script = project_root / "scripts" / "run_experiment.py"
        content = run_script.read_text()

        assert "EvaluationSandbox" not in content, (
            "scripts/run_experiment.py should not use EvaluationSandbox directly. "
            "Use bench.runner.run() which manages sandbox creation internally."
        )

    def test_bench_experiments_runner_not_imported_in_scripts(self):
        """Guard: scripts don't import bench.experiments.runner directly.

        All scripts should use bench.runner.run() instead of importing
        from bench.experiments.runner.
        """
        scripts_dir = project_root / "scripts"
        if not scripts_dir.exists():
            return

        for script in scripts_dir.glob("*.py"):
            if script.name in ("test_docker_pi_runner.py",):
                # These are test utilities, skip them
                continue

            content = script.read_text()
            # Check for direct imports from bench.experiments.runner
            if "from bench.experiments.runner import" in content:
                # Allow imports that don't include ExperimentRunner
                if "ExperimentRunner" in content:
                    raise AssertionError(
                        f"{script.name} should not import ExperimentRunner from "
                        "bench.experiments.runner. Use bench.runner.run() instead."
                    )

    def test_posit_gskill_uses_canonical_runner(self):
        """Guard: posit_gskill evaluate command uses bench.runner.run()."""
        commands_file = project_root / "posit_gskill" / "commands.py"
        if not commands_file.exists():
            return

        content = commands_file.read_text()

        # Should import bench.runner.run
        assert "from bench.runner import run" in content or "bench.runner.run" in content, (
            "posit_gskill/commands.py should import bench.runner.run for evaluation."
        )

    def test_no_legacy_import_patterns(self):
        """Guard: No legacy import patterns exist in production code.

        Check for old patterns that should have been migrated:
        - Direct imports of test_runner module
        - Old cc_mirror imports
        """
        python_files = self._get_python_files(project_root)

        for file_path in python_files:
            if self._is_allowed_path(file_path, project_root):
                continue

            content = file_path.read_text()
            relative = str(file_path.relative_to(project_root))

            # Check for old test_runner module imports
            assert "from evaluation.test_runner" not in content, (
                f"Found legacy 'from evaluation.test_runner' in {relative}. "
                "This module has been removed. Use bench.runner.run() instead."
            )

            # Check for cc_mirror imports (old variant creation system).
            # The underscore form "cc_mirror" is the legacy Python import pattern.
            # The hyphenated "cc-mirror" won't match since "-" != "_".
            assert "cc_mirror" not in content.lower(), (
                f"Found 'cc_mirror' in {relative}. "
                "The cc-mirror system has been replaced with DockerPiRunner. "
                "Use bench.runner.run() instead."
            )

    def test_no_direct_experiment_runner_instantiation(self):
        """Guard: No direct ExperimentRunner instantiation outside allowlist."""
        # Direct instantiation of ExperimentRunner bypasses bench.runner.run()
        # which means no canonical guarantees (telemetry, logging, etc.)

        ALLOWED_FILES = {
            "bench/runner.py",  # The canonical runner itself
            "bench/experiments/runner.py",  # The ExperimentRunner implementation
            "bench/optimize/",  # GEPA adapter uses ExperimentRunner for optimization
            "tests/",  # Test files can use it for testing
        }

        for py_file in Path(".").rglob("*.py"):
            # Skip excluded directories
            if any(
                part in str(py_file)
                for part in [".venv", "node_modules", "__pycache__", "visualizer"]
            ):
                continue

            # Skip allowed files
            if any(allowed in str(py_file) for allowed in ALLOWED_FILES):
                continue

            content = py_file.read_text()
            # Check for direct ExperimentRunner instantiation
            if "ExperimentRunner(" in content and "from bench.experiments" in content:
                # Also check it's not just an import but an actual instantiation
                if "ExperimentRunner(" in content:
                    pytest.fail(
                        f"Direct ExperimentRunner instantiation in {py_file}. "
                        "Use bench.runner.run() instead."
                    )

    def test_no_run_experiment_function_calls(self):
        """Guard: No direct run_experiment() calls outside allowlist."""
        ALLOWED_FILES = {
            "bench/runner.py",
            "bench/experiments/runner.py",
            "tests/",
        }

        for py_file in Path(".").rglob("*.py"):
            if any(
                part in str(py_file)
                for part in [".venv", "node_modules", "__pycache__", "visualizer"]
            ):
                continue

            if any(allowed in str(py_file) for allowed in ALLOWED_FILES):
                continue

            content = py_file.read_text()
            # Check for run_experiment function call (not method)
            if re.search(r"\brun_experiment\s*\(", content):
                pytest.fail(
                    f"Direct run_experiment() call in {py_file}. Use bench.runner.run() instead."
                )
