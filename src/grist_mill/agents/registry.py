"""Agent registry for pluggable agent implementations.

Provides a central ``AgentRegistry`` that maps agent names to factory functions.
New agents can be registered at runtime and selected by name in the config.

Validates VAL-AGENT-05: Agent implementations are pluggable via registry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from grist_mill.agents.api_agent import APIAgent
from grist_mill.agents.provider import MockProvider, ProviderResponse
from grist_mill.interfaces import BaseAgent

if TYPE_CHECKING:
    from grist_mill.schemas import HarnessConfig

logger = logging.getLogger(__name__)

# Type alias for agent factory functions
AgentFactory = Callable[["HarnessConfig"], BaseAgent]


class AgentRegistry:
    """Central registry for pluggable agent implementations.

    Agents are registered by name with a factory function that creates
    an agent instance from a ``HarnessConfig``. The builtin ``api`` agent
    is pre-registered.
    """

    def __init__(self) -> None:
        self._factories: dict[str, AgentFactory] = {}
        # Pre-register the builtin API agent
        self.register("api", _default_api_agent_factory)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        factory: AgentFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an agent factory by name.

        Args:
            name: Unique name for the agent implementation.
            factory: A callable that takes a ``HarnessConfig`` and returns a ``BaseAgent``.
            overwrite: If True, replace an existing registration with the same name.

        Raises:
            ValueError: If an agent with the same name is already registered and
                ``overwrite`` is False.
        """
        if name in self._factories and not overwrite:
            msg = f"Agent '{name}' is already registered. Use overwrite=True to replace it."
            raise ValueError(msg)

        if name in self._factories:
            logger.info("Overwriting agent '%s'", name)
        else:
            logger.info("Registering agent '%s'", name)

        self._factories[name] = factory

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Check if an agent is registered.

        Args:
            name: The agent name to check.

        Returns:
            True if the agent is registered.
        """
        return name in self._factories

    def create(self, name: str, config: HarnessConfig) -> BaseAgent:
        """Create an agent instance by name.

        Args:
            name: The registered agent name.
            config: Harness configuration to pass to the factory.

        Returns:
            A ``BaseAgent`` instance.

        Raises:
            KeyError: If the agent name is not registered.
        """
        if name not in self._factories:
            msg = f"Agent '{name}' is not registered. Available: {self.list_agents()}"
            raise KeyError(msg)

        factory = self._factories[name]
        agent = factory(config)
        logger.debug("Created agent '%s': %s", name, type(agent).__name__)
        return agent

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_agents(self) -> list[str]:
        """List all registered agent names.

        Returns:
            A list of agent name strings.
        """
        return list(self._factories.keys())

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={self.list_agents()})"


# ---------------------------------------------------------------------------
# Default factory for the builtin "api" agent
# ---------------------------------------------------------------------------


def _default_api_agent_factory(config: HarnessConfig) -> BaseAgent:
    """Create the default API agent from a harness config.

    This factory creates an ``APIAgent`` with a ``MockProvider`` by default.
    In production, this would be replaced with a provider resolved from the
    config (e.g., openrouter/openai/anthropic).

    The ``max_turns`` and ``timeout`` default to reasonable values. Production
    users should register a custom factory that reads these from the full
    ``GristMillConfig`` (which includes the ``AgentSection`` with
    ``max_turns`` and ``timeout`` fields).

    Args:
        config: Harness configuration containing agent settings.

    Returns:
        An ``APIAgent`` instance.
    """
    # Default: use MockProvider for the builtin "api" agent.
    # Real providers (openai, openrouter, etc.) will be registered by the
    # provider module (m6-llm-providers milestone).
    provider = MockProvider(
        responses=[
            ProviderResponse(
                content="[Mock] Task completed successfully.",
                tool_calls=[],
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            ),
        ]
    )

    return APIAgent(
        provider=provider,
        max_turns=10,
        timeout=60,
    )


__all__ = [
    "AgentFactory",
    "AgentRegistry",
]
