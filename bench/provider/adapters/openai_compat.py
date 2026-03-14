"""OpenAI-compatible adapter for providers using OpenAI SDK with custom base_url."""

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel


@dataclass
class InferenceResult:
    """Result from structured inference."""

    content: str
    reasoning: str | None = None
    raw_response: Any = None  # For debugging


class OpenAICompatAdapter:
    """Adapter for OpenAI-compatible APIs (opencode, openrouter, zai, etc.)."""

    def __init__(self, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_structured(
        self,
        model_id: str,
        messages: list[dict],
        response_schema: type[BaseModel],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        json_mode: str = "native",  # "native", "prompt", or "none"
        drop_unsupported_params: bool = True,
    ) -> InferenceResult:
        """
        Generate structured output matching the given Pydantic schema.

        Args:
            model_id: Raw model ID to send to API
            messages: Chat messages
            response_schema: Pydantic model for response validation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: How to request JSON output
            drop_unsupported_params: Whether to drop params the model may not support

        Returns:
            InferenceResult with validated content
        """
        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        # Handle temperature - some reasoning models don't support it
        if temperature is not None and not drop_unsupported_params:
            kwargs["temperature"] = temperature

        # Handle JSON mode
        if json_mode == "native":
            kwargs["response_format"] = {"type": "json_object"}

        # Make the API call
        response = self.client.chat.completions.create(**kwargs)

        # Extract content - handle reasoning models where content may be in different fields
        message = response.choices[0].message
        raw_content = message.content

        # Handle reasoning models (content may be in reasoning_content)
        reasoning = None
        if not raw_content:
            raw_content = getattr(message, "reasoning_content", None)
            if raw_content:
                reasoning = raw_content  # The actual reasoning
                # Try to get final answer from other fields

        if not raw_content:
            # Check provider-specific fields
            provider_fields = getattr(message, "provider_specific_fields", None)
            if provider_fields:
                raw_content = provider_fields.get("reasoning_content")
                if raw_content:
                    reasoning = raw_content

        if not raw_content:
            raise ValueError("No content in response")

        # Validate JSON against schema
        try:
            validated = response_schema.model_validate_json(raw_content)
            return InferenceResult(
                content=validated.model_dump_json(),
                reasoning=reasoning,
                raw_response=response,
            )
        except Exception as e:
            # Try to extract JSON from potentially wrapped content
            extracted = self._extract_json(raw_content)
            if extracted:
                validated = response_schema.model_validate_json(extracted)
                return InferenceResult(
                    content=validated.model_dump_json(),
                    reasoning=reasoning,
                    raw_response=response,
                )
            raise ValueError(f"Failed to parse response as {response_schema.__name__}: {e}") from e

    def _extract_json(self, content: str) -> str | None:
        """Try to extract JSON from potentially wrapped content."""
        # Try to find JSON object boundaries
        start = content.find("{")
        if start == -1:
            return None
        end = content.rfind("}")
        if end == -1 or end <= start:
            return None
        return content[start : end + 1]
