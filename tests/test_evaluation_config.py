"""Tests for evaluation configuration."""

from evaluation.config import EvaluationConfig, ModelConfig, SkillConfig, TasksConfig


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
