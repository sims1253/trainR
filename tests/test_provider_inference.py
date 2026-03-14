"""Tests for provider inference gateway."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from bench.provider.adapters.openai_compat import InferenceResult, OpenAICompatAdapter
from bench.provider.inference import embed_schema_in_messages, generate_structured


class SampleSchema(BaseModel):
    """Sample schema for structured output testing."""

    result: str
    score: int


class TestEmbedSchemaInMessages:
    """Tests for embed_schema_in_messages function."""

    def test_adds_schema_to_user_message(self):
        """Schema should be appended to last user message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = embed_schema_in_messages(messages, SampleSchema)

        assert len(result) == 2
        assert result[0]["content"] == "You are helpful."
        assert "Respond with valid JSON" in result[1]["content"]
        assert "result" in result[1]["content"]
        assert "score" in result[1]["content"]

    def test_original_messages_not_modified(self):
        """Original messages list should not be mutated."""
        messages = [
            {"role": "user", "content": "Original"},
        ]
        original_content = messages[0]["content"]
        embed_schema_in_messages(messages, SampleSchema)

        assert messages[0]["content"] == original_content

    def test_no_user_message_returns_same_messages(self):
        """If no user message, return messages unchanged (schema not embedded)."""
        messages = [
            {"role": "system", "content": "System only"},
        ]
        result = embed_schema_in_messages(messages, SampleSchema)

        assert result[0]["content"] == "System only"
        assert "schema" not in result[0]["content"].lower()


class TestOpenAICompatAdapter:
    """Tests for OpenAI-compatible adapter."""

    def _create_mock_response(self, content: str) -> MagicMock:
        """Create a mock OpenAI response."""
        response = MagicMock()
        message = MagicMock()
        message.content = content
        message.reasoning_content = None
        response.choices = [MagicMock(message=message)]
        return response

    def test_generate_structured_parses_json(self):
        """Adapter should parse JSON response."""
        adapter = OpenAICompatAdapter(
            base_url="https://api.example.com/v1",
            api_key="test-key",
        )

        mock_response = self._create_mock_response('{"result": "success", "score": 42}')

        with patch.object(adapter.client.chat.completions, "create", return_value=mock_response):
            result = adapter.generate_structured(
                model_id="test-model",
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
            )

        # Result content should be valid JSON
        parsed = json.loads(result.content)
        assert parsed["result"] == "success"
        assert parsed["score"] == 42

    def test_generate_structured_handles_reasoning_content(self):
        """Adapter should extract content from reasoning_content field."""
        adapter = OpenAICompatAdapter(
            base_url="https://api.example.com/v1",
            api_key="test-key",
        )

        response = MagicMock()
        message = MagicMock()
        message.content = None
        message.reasoning_content = '{"result": "from-reasoning", "score": 100}'
        response.choices = [MagicMock(message=message)]

        with patch.object(adapter.client.chat.completions, "create", return_value=response):
            result = adapter.generate_structured(
                model_id="test-model",
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
            )

        parsed = json.loads(result.content)
        assert parsed["result"] == "from-reasoning"

    def test_generate_structured_extracts_wrapped_json(self):
        """Adapter should extract JSON from wrapped content."""
        adapter = OpenAICompatAdapter(
            base_url="https://api.example.com/v1",
            api_key="test-key",
        )

        wrapped_content = """Here's my response:
```json
{"result": "extracted", "score": 50}
```
Hope that helps!"""

        mock_response = self._create_mock_response(wrapped_content)

        with patch.object(adapter.client.chat.completions, "create", return_value=mock_response):
            result = adapter.generate_structured(
                model_id="test-model",
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
            )

        parsed = json.loads(result.content)
        assert parsed["result"] == "extracted"

    def test_generate_structured_raises_on_invalid_json(self):
        """Adapter should raise ValueError for invalid JSON."""
        adapter = OpenAICompatAdapter(
            base_url="https://api.example.com/v1",
            api_key="test-key",
        )

        mock_response = self._create_mock_response("not valid json")

        with (
            patch.object(adapter.client.chat.completions, "create", return_value=mock_response),
            pytest.raises(ValueError, match="Failed to parse response"),
        ):
            adapter.generate_structured(
                model_id="test-model",
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
            )


class TestGenerateStructured:
    """Tests for the main generate_structured gateway function."""

    @patch("bench.provider.inference.OpenAICompatAdapter")
    def test_routes_to_openai_compat_adapter(self, mock_adapter_class):
        """Should route to OpenAICompatAdapter for openai_compat providers."""
        mock_adapter = MagicMock()
        mock_adapter.generate_structured.return_value = InferenceResult(
            content='{"result": "test", "score": 1}'
        )
        mock_adapter_class.return_value = mock_adapter

        with patch.dict("os.environ", {"Z_AI_API_KEY": "test-key"}):
            result = generate_structured(
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
                model_name="zai/glm-5",  # zai provider in llm.yaml
            )

        assert result.content is not None

    def test_raises_for_unknown_model(self):
        """Should raise KeyError for unknown model."""
        with pytest.raises(KeyError, match="not found"):
            generate_structured(
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
                model_name="nonexistent-model-xyz",
            )

    def test_raises_for_missing_api_key(self):
        """Should raise KeyError when API key is missing."""
        # This test verifies the error path exists
        # We mock the resolver to return a model/provider that has no API key
        from bench.provider.models import ModelInfo, ProviderInfo
        from bench.provider.resolver import ProviderResolver

        mock_resolver = MagicMock(spec=ProviderResolver)
        mock_resolver.get_model_info.return_value = ModelInfo(
            id="test-model",
            provider="test-provider",
            capabilities={"json_mode": "native"},
        )
        mock_resolver.get_provider_info.return_value = ProviderInfo(
            name="test-provider",
            api_style="openai_compat",
            api_key_env="TEST_MISSING_API_KEY",
            base_url="https://api.test.com/v1",
        )

        with (
            patch("bench.provider.env.get_env_var", return_value=None),
            pytest.raises(KeyError, match="API key not found"),
        ):
            generate_structured(
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
                model_name="test-model",
                resolver=mock_resolver,
            )

    @patch("bench.provider.inference.OpenAICompatAdapter")
    def test_openai_compat_provider_uses_canonical_base_url_fallback(self, mock_adapter_class):
        """OpenAI-compatible providers should get canonical base_url defaults."""
        from bench.provider.models import ModelInfo, ProviderInfo
        from bench.provider.resolver import ProviderResolver

        mock_adapter = MagicMock()
        mock_adapter.generate_structured.return_value = InferenceResult(
            content='{"result": "ok", "score": 1}'
        )
        mock_adapter_class.return_value = mock_adapter

        mock_resolver = MagicMock(spec=ProviderResolver)
        mock_resolver.get_model_info.return_value = ModelInfo(
            id="openai/gpt-oss-120b:free",
            provider="openrouter",
            capabilities={"json_mode": "native"},
        )
        mock_resolver.get_provider_info.return_value = ProviderInfo(
            name="openrouter",
            api_style="openai_compat",
            api_key_env="OPENROUTER_API_KEY",
            base_url=None,
        )

        generate_structured(
            messages=[{"role": "user", "content": "test"}],
            response_schema=SampleSchema,
            model_name="gpt-oss-120b",
            resolver=mock_resolver,
            api_key="test-key",
        )

        mock_adapter_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )

    def test_openai_compat_provider_without_base_url_raises_for_unknown_provider(self):
        """Unknown OpenAI-compatible providers must provide explicit base_url."""
        from bench.provider.models import ModelInfo, ProviderInfo
        from bench.provider.resolver import ProviderResolver

        mock_resolver = MagicMock(spec=ProviderResolver)
        mock_resolver.get_model_info.return_value = ModelInfo(
            id="custom-model",
            provider="custom_provider",
            capabilities={"json_mode": "native"},
        )
        mock_resolver.get_provider_info.return_value = ProviderInfo(
            name="custom_provider",
            api_style="openai_compat",
            api_key_env="CUSTOM_API_KEY",
            base_url=None,
        )

        with pytest.raises(ValueError, match="base_url required for openai_compat provider"):
            generate_structured(
                messages=[{"role": "user", "content": "test"}],
                response_schema=SampleSchema,
                model_name="custom-model",
                resolver=mock_resolver,
                api_key="test-key",
            )
