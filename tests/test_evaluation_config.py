"""Tests for evaluation configuration."""

import os
from unittest.mock import patch

from evaluation.config import EvaluationConfig, ModelConfig, SkillConfig, TasksConfig
from evaluation.models import FailureCategory
from evaluation.sandbox import get_required_api_key


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()
    assert config.task == "glm-4.5"
    assert config.reflection == "glm-5"
    assert config.tasks is None


def test_model_config_custom():
    """Test ModelConfig with custom values."""
    config = ModelConfig(task="custom-model", reflection="reflection-model")
    assert config.task == "custom-model"
    assert config.reflection == "reflection-model"


def test_model_config_get_models_single():
    """Test get_models returns single model in list."""
    config = ModelConfig(task="test-model")
    assert config.get_models() == ["test-model"]


def test_model_config_get_models_multiple():
    """Test get_models returns all tasks when tasks is set."""
    config = ModelConfig(task="default", tasks=["model1", "model2", "model3"])
    assert config.get_models() == ["model1", "model2", "model3"]


def test_skill_config_defaults():
    """Test SkillConfig default values."""
    config = SkillConfig()
    assert config.path is None
    assert config.no_skill is False


def test_skill_config_no_skill():
    """Test SkillConfig with no_skill=True."""
    config = SkillConfig(no_skill=True)
    assert config.no_skill is True
    assert config.path is None


def test_skill_config_with_path():
    """Test SkillConfig with path."""
    config = SkillConfig(path="skills/test.md")
    assert config.path == "skills/test.md"
    assert config.no_skill is False


def test_tasks_config_defaults():
    """Test TasksConfig default values."""
    config = TasksConfig()
    assert config.dir == "tasks"
    assert config.splits == ["dev", "held_out"]
    assert config.difficulty is None
    assert config.packages is None


def test_tasks_config_custom():
    """Test TasksConfig with custom values."""
    config = TasksConfig(
        dir="custom_tasks",
        splits=["train", "test"],
        difficulty=["easy", "medium"],
        packages=["dplyr", "ggplot2"],
    )
    assert config.dir == "custom_tasks"
    assert config.splits == ["train", "test"]
    assert config.difficulty == ["easy", "medium"]
    assert config.packages == ["dplyr", "ggplot2"]


def test_evaluation_config_defaults():
    """Test EvaluationConfig default values."""
    config = EvaluationConfig()
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.skill, SkillConfig)
    assert isinstance(config.tasks, TasksConfig)


def test_evaluation_config_from_dict():
    """Test EvaluationConfig creation from dict."""
    data = {
        "model": {"task": "test-model", "reflection": "test-reflection"},
        "skill": {"no_skill": True},
        "tasks": {"splits": ["train"]},
    }
    config = EvaluationConfig.from_dict(data)
    assert config.model.task == "test-model"
    assert config.model.reflection == "test-reflection"
    assert config.skill.no_skill is True
    assert config.tasks.splits == ["train"]


def test_evaluation_config_from_dict_empty():
    """Test EvaluationConfig from empty dict uses defaults."""
    config = EvaluationConfig.from_dict({})
    assert config.model.task == "glm-4.5"
    assert config.skill.no_skill is False
    assert config.tasks.splits == ["dev", "held_out"]


def test_evaluation_config_get_skill_name_no_skill():
    """Test get_skill_name with no_skill."""
    config = EvaluationConfig(skill=SkillConfig(no_skill=True))
    assert config.get_skill_name() == "no_skill"


def test_evaluation_config_get_skill_name_with_path():
    """Test get_skill_name extracts stem from path."""
    config = EvaluationConfig(skill=SkillConfig(path="skills/test-skill.md"))
    assert config.get_skill_name() == "test-skill"


def test_evaluation_config_get_skill_name_unknown():
    """Test get_skill_name when no path and no_skill is False."""
    config = EvaluationConfig(skill=SkillConfig())
    assert config.get_skill_name() == "unknown"


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_infrastructure_categories_exist(self):
        """Test that infrastructure error categories exist."""
        assert FailureCategory.CONFIG_ERROR.value == "CONFIG_ERROR"
        assert FailureCategory.ENVIRONMENT_ERROR.value == "ENVIRONMENT_ERROR"
        assert FailureCategory.PACKAGE_NOT_FOUND.value == "PACKAGE_NOT_FOUND"

    def test_runtime_categories_exist(self):
        """Test that runtime error categories exist."""
        assert FailureCategory.TIMEOUT.value == "TIMEOUT"

    def test_code_generation_categories_exist(self):
        """Test that code generation error categories exist."""
        assert FailureCategory.SYNTAX_ERROR.value == "SYNTAX_ERROR"
        assert FailureCategory.MISSING_IMPORT.value == "MISSING_IMPORT"

    def test_test_failure_categories_exist(self):
        """Test that test failure categories exist."""
        assert FailureCategory.TEST_FAILURE.value == "TEST_FAILURE"
        assert FailureCategory.SNAPSHOT_MISMATCH.value == "SNAPSHOT_MISMATCH"
        assert FailureCategory.INCOMPLETE_SOLUTION.value == "INCOMPLETE_SOLUTION"


class TestProviderAwareKeyCheck:
    """Tests for provider-aware API key checking."""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_required_api_key_openrouter_missing(self):
        """Test that missing OPENROUTER_API_KEY is detected for openrouter model."""
        env_var, api_key = get_required_api_key("openrouter/some-model")
        assert env_var == "OPENROUTER_API_KEY"
        assert api_key is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_get_required_api_key_openrouter_present(self):
        """Test that present OPENROUTER_API_KEY is returned for openrouter model."""
        env_var, api_key = get_required_api_key("openrouter/some-model")
        assert env_var == "OPENROUTER_API_KEY"
        assert api_key == "test-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_required_api_key_opencode_missing(self):
        """Test that missing OPENCODE_API_KEY is detected for opencode model."""
        env_var, api_key = get_required_api_key("opencode/some-model")
        assert env_var == "OPENCODE_API_KEY"
        assert api_key is None

    @patch.dict(os.environ, {"OPENCODE_API_KEY": "test-key"}, clear=True)
    def test_get_required_api_key_opencode_present(self):
        """Test that present OPENCODE_API_KEY is returned for opencode model."""
        env_var, api_key = get_required_api_key("opencode/some-model")
        assert env_var == "OPENCODE_API_KEY"
        assert api_key == "test-key"

    @patch.dict(os.environ, {"Z_AI_API_KEY": "zai-key"}, clear=True)
    def test_get_required_api_key_zai_present(self):
        """Test that Z_AI_API_KEY is detected for zai model."""
        env_var, api_key = get_required_api_key("zai/some-model")
        assert env_var == "Z_AI_API_KEY"
        assert api_key == "zai-key"

    def test_get_required_api_key_unknown_prefix(self):
        """Test that unknown prefix returns (None, None)."""
        env_var, api_key = get_required_api_key("unknown-provider/some-model")
        assert env_var is None
        assert api_key is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True)
    def test_get_required_api_key_openai(self):
        """Test that OPENAI_API_KEY is detected for openai model."""
        env_var, api_key = get_required_api_key("openai/gpt-4")
        assert env_var == "OPENAI_API_KEY"
        assert api_key == "openai-key"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-key"}, clear=True)
    def test_get_required_api_key_anthropic(self):
        """Test that ANTHROPIC_API_KEY is detected for anthropic model."""
        env_var, api_key = get_required_api_key("anthropic/claude-3")
        assert env_var == "ANTHROPIC_API_KEY"
        assert api_key == "anthropic-key"
