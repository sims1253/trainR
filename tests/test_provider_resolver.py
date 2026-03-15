"""Regression tests for provider resolver compatibility fields."""

from bench.provider.resolver import ProviderResolver
from config import LLMConfig


def test_resolver_builds_litellm_model_with_separator() -> None:
    """Deprecated litellm_model should keep provider/model separator."""
    resolver = ProviderResolver("configs/llm.yaml")

    assert resolver.get_litellm_model("zai/glm-5") == "openai/glm-5"
    assert (
        resolver.get_litellm_model("qwen3-next-80b-a3b")
        == "openrouter/qwen/qwen3-next-80b-a3b-instruct:free"
    )


def test_llm_config_defaults_preserve_provider_prefixes() -> None:
    """Default task/reflection models should keep compatibility prefixes."""
    llm_config = LLMConfig.from_yaml("configs/llm.yaml")

    assert llm_config.task_agent == "k2p5"
    assert llm_config.reflection == "openai/glm-5"
