"""Tests for PiRunner configuration."""

from evaluation.pi_runner import DockerPiRunnerConfig, PiEvaluationResult, PiRunnerConfig


def test_pi_runner_config_defaults():
    """Test PiRunnerConfig default values."""
    config = PiRunnerConfig()
    assert config.model == "openrouter/openai/gpt-oss-120b:free"
    assert config.max_turns == 50
    assert config.timeout == 600
    assert config.cwd is None


def test_pi_runner_config_custom():
    """Test PiRunnerConfig with custom values."""
    config = PiRunnerConfig(
        model="custom-model",
        max_turns=100,
        timeout=300,
        cwd="/custom/path",
    )
    assert config.model == "custom-model"
    assert config.max_turns == 100
    assert config.timeout == 300
    assert config.cwd == "/custom/path"


def test_pi_runner_config_post_init_sets_pi_binary():
    """Test that __post_init__ sets pi_binary."""
    config = PiRunnerConfig()
    # Should be set to something (either a path, or 'bun')
    assert config.pi_binary != ""
    assert config.pi_binary is not None


def test_docker_pi_runner_config_defaults():
    """Test DockerPiRunnerConfig default values."""
    config = DockerPiRunnerConfig()
    assert config.model == "openrouter/openai/gpt-oss-120b:free"
    assert config.docker_image == "posit-gskill-eval:latest"
    assert config.max_turns == 50
    assert config.timeout == 600


def test_docker_pi_runner_config_custom():
    """Test DockerPiRunnerConfig with custom values."""
    config = DockerPiRunnerConfig(
        model="custom-model",
        timeout=300,
        docker_image="custom:latest",
        max_turns=25,
    )
    assert config.model == "custom-model"
    assert config.timeout == 300
    assert config.docker_image == "custom:latest"
    assert config.max_turns == 25


def test_docker_pi_runner_config_api_keys_default():
    """Test DockerPiRunnerConfig sets default API keys list."""
    config = DockerPiRunnerConfig()
    assert config.api_keys is not None
    assert "OPENROUTER_API_KEY" in config.api_keys
    assert "ZAI_API_KEY" in config.api_keys


def test_docker_pi_runner_config_api_keys_custom():
    """Test DockerPiRunnerConfig with custom API keys."""
    config = DockerPiRunnerConfig(api_keys=["CUSTOM_KEY"])
    assert config.api_keys == ["CUSTOM_KEY"]


def test_pi_evaluation_result_defaults():
    """Test PiEvaluationResult default values."""
    result = PiEvaluationResult(success=True, score=1.0, output="test output")
    assert result.success is True
    assert result.score == 1.0
    assert result.output == "test output"
    assert result.error is None
    assert result.test_results is None
    assert result.execution_time == 0.0
    assert result.model == ""
    assert result.turns_used == 0


def test_pi_evaluation_result_custom():
    """Test PiEvaluationResult with custom values."""
    result = PiEvaluationResult(
        success=False,
        score=0.0,
        output="",
        error="Timeout",
        test_results={"passed": False, "num_failed": 3},
        execution_time=120.5,
        model="test-model",
        turns_used=10,
    )
    assert result.success is False
    assert result.score == 0.0
    assert result.error == "Timeout"
    assert result.test_results == {"passed": False, "num_failed": 3}
    assert result.execution_time == 120.5
    assert result.model == "test-model"
    assert result.turns_used == 10
