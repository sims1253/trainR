"""Tests for LLMTaskJudge in mine_prs.py."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from scripts.mine_prs import LLMTaskJudge


class TestLLMTaskJudge:
    """Tests for LLMTaskJudge class."""

    @patch("scripts.mine_prs.get_llm_config")
    def test_init_resolves_model_from_config(self, mock_get_config):
        """Judge should resolve model name from config."""
        mock_config = MagicMock()
        mock_config.get_mining_model.return_value = "glm-5-plan"
        mock_config.get_model_config.return_value = {
            "provider": "zai",
            "id": "glm-5",
            "capabilities": {"json_mode": "native"},
        }
        mock_config.get_model_api_key.return_value = "test-api-key"
        mock_config.get_model_base_url.return_value = "https://api.zukijourney.com/v1"
        mock_get_config.return_value = mock_config

        judge = LLMTaskJudge()

        assert judge.model == "glm-5-plan"
        assert judge.provider == "zai"
        assert judge.api_key == "test-api-key"

    @patch("scripts.mine_prs.get_llm_config")
    def test_init_with_explicit_model(self, mock_get_config):
        """Judge should use explicitly provided model."""
        mock_config = MagicMock()
        mock_config.get_model_config.return_value = {
            "provider": "opencode",
            "id": "glm-5-free",
            "capabilities": {"json_mode": "native"},
        }
        mock_config.get_model_api_key.return_value = "test-key"
        mock_config.get_model_base_url.return_value = "https://opencode.ai/zen/v1"
        mock_get_config.return_value = mock_config

        judge = LLMTaskJudge(model="glm-5-free")

        assert judge.model == "glm-5-free"

    @patch("scripts.mine_prs.get_llm_config")
    def test_call_inference_gateway_success(self, mock_get_config):
        """_call_inference_gateway should return validated schema."""
        mock_config = MagicMock()
        mock_config.get_mining_model.return_value = "glm-5-plan"
        mock_config.get_model_config.return_value = {
            "provider": "zai",
            "id": "glm-5",
            "capabilities": {"json_mode": "native"},
        }
        mock_config.get_model_api_key.return_value = "test-key"
        mock_config.get_model_base_url.return_value = "https://api.zukijourney.com/v1"
        mock_get_config.return_value = mock_config

        judge = LLMTaskJudge()

        # Mock the inference gateway
        class TestSchema(BaseModel):
            result: str

        from bench.provider.inference import InferenceResult

        mock_result = InferenceResult(content='{"result": "success"}')

        with patch("scripts.mine_prs.generate_structured", return_value=mock_result):
            result = judge._call_inference_gateway(
                [{"role": "user", "content": "test"}],
                TestSchema,
            )

        assert isinstance(result, TestSchema)
        assert result.result == "success"

    @patch("scripts.mine_prs.get_llm_config")
    def test_no_litellm_import(self, mock_get_config):
        """Verify that litellm is not imported anywhere in the module."""
        import scripts.mine_prs as mine_module

        source = inspect.getsource(mine_module)
        assert "import litellm" not in source, "litellm should not be imported"
        assert "litellm." not in source, "litellm module should not be used"

    @patch("scripts.mine_prs.get_llm_config")
    def test_call_inference_gateway_uses_embed_schema_for_prompt_mode(self, mock_get_config):
        """_call_inference_gateway should embed schema when json_mode is 'prompt'."""
        mock_config = MagicMock()
        mock_config.get_mining_model.return_value = "test-model"
        mock_config.get_model_config.return_value = {
            "provider": "test-provider",
            "id": "test-id",
            "capabilities": {"json_mode": "prompt"},  # Requires prompt embedding
        }
        mock_config.get_model_api_key.return_value = "test-key"
        mock_config.get_model_base_url.return_value = "https://api.test.com/v1"
        mock_get_config.return_value = mock_config

        judge = LLMTaskJudge()

        class TestSchema(BaseModel):
            value: int

        from bench.provider.inference import InferenceResult

        mock_result = InferenceResult(content='{"value": 42}')

        with (
            patch("scripts.mine_prs.generate_structured", return_value=mock_result),
            patch("scripts.mine_prs.embed_schema_in_messages") as mock_embed,
        ):
            mock_embed.return_value = [{"role": "user", "content": "test with schema"}]

            judge._call_inference_gateway(
                [{"role": "user", "content": "test"}],
                TestSchema,
            )

            # Verify embed_schema_in_messages was called for prompt mode
            mock_embed.assert_called_once()

    @patch("scripts.mine_prs.get_llm_config")
    def test_call_inference_gateway_handles_errors(self, mock_get_config):
        """_call_inference_gateway should propagate exceptions from generate_structured."""
        mock_config = MagicMock()
        mock_config.get_mining_model.return_value = "test-model"
        mock_config.get_model_config.return_value = {
            "provider": "test-provider",
            "id": "test-id",
            "capabilities": {"json_mode": "native"},
        }
        mock_config.get_model_api_key.return_value = "test-key"
        mock_config.get_model_base_url.return_value = "https://api.test.com/v1"
        mock_get_config.return_value = mock_config

        judge = LLMTaskJudge()

        class TestSchema(BaseModel):
            result: str

        with (
            patch(
                "scripts.mine_prs.generate_structured",
                side_effect=ValueError("API error"),
            ),
            pytest.raises(ValueError, match="API error"),
        ):
            judge._call_inference_gateway(
                [{"role": "user", "content": "test"}],
                TestSchema,
            )
