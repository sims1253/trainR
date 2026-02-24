"""Harness registry for dynamic harness discovery and instantiation.

This module provides a registry pattern for managing harness implementations,
allowing for dynamic registration, discovery, and instantiation of harnesses.

Example:
    from bench.harness import HarnessRegistry, register_harness

    # Register a harness
    @register_harness("my_harness")
    class MyHarness(AgentHarness):
        ...

    # Or manually register
    HarnessRegistry.register("another_harness", AnotherHarness)

    # Get a harness instance
    harness = HarnessRegistry.get("my_harness", config)
"""

from __future__ import annotations

from typing import Callable, ClassVar

from .base import AgentHarness, HarnessConfig


class HarnessRegistry:
    """Registry for agent harness implementations.

    This class provides a central registry for all available harness
    implementations. Harnesses can be registered dynamically and
    retrieved by name.

    Example:
        # List available harnesses
        available = HarnessRegistry.list_available()

        # Get a harness
        harness = HarnessRegistry.get("pi_sdk", config)

        # Register a new harness
        HarnessRegistry.register("custom", CustomHarness)
    """

    _harnesses: ClassVar[dict[str, type[AgentHarness]]] = {}

    @classmethod
    def register(cls, name: str, harness_cls: type[AgentHarness]) -> None:
        """Register a harness class with the given name.

        Args:
            name: Unique name for the harness
            harness_cls: The harness class to register

        Raises:
            ValueError: If name is empty or already registered
            TypeError: If harness_cls is not a valid harness class
        """
        if not name or not name.strip():
            raise ValueError("Harness name cannot be empty")

        name = name.strip().lower()

        if not isinstance(harness_cls, type):
            raise TypeError(f"Expected a class, got {type(harness_cls)}")

        if not issubclass(harness_cls, AgentHarness):
            raise TypeError(
                f"Harness class must be a subclass of AgentHarness, got {harness_cls.__name__}"
            )

        if name in cls._harnesses:
            raise ValueError(
                f"Harness '{name}' is already registered. Use a different name or unregister first."
            )

        cls._harnesses[name] = harness_cls

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a harness by name.

        Args:
            name: Name of the harness to unregister

        Returns:
            True if harness was unregistered, False if not found
        """
        name = name.strip().lower()
        if name in cls._harnesses:
            del cls._harnesses[name]
            return True
        return False

    @classmethod
    def get(cls, name: str, config: HarnessConfig) -> AgentHarness:
        """Get a harness instance by name.

        Args:
            name: Name of the registered harness
            config: Configuration for the harness

        Returns:
            Instantiated harness

        Raises:
            KeyError: If harness is not registered
        """
        name = name.strip().lower()

        if name not in cls._harnesses:
            available = cls.list_available()
            raise KeyError(f"Harness '{name}' is not registered. Available harnesses: {available}")

        harness_cls = cls._harnesses[name]
        return harness_cls(config)

    @classmethod
    def get_class(cls, name: str) -> type[AgentHarness]:
        """Get a harness class by name without instantiating.

        Args:
            name: Name of the registered harness

        Returns:
            The harness class

        Raises:
            KeyError: If harness is not registered
        """
        name = name.strip().lower()

        if name not in cls._harnesses:
            available = cls.list_available()
            raise KeyError(f"Harness '{name}' is not registered. Available harnesses: {available}")

        return cls._harnesses[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered harness names.

        Returns:
            Sorted list of registered harness names
        """
        return sorted(cls._harnesses.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a harness is registered.

        Args:
            name: Name of the harness to check

        Returns:
            True if registered, False otherwise
        """
        name = name.strip().lower()
        return name in cls._harnesses

    @classmethod
    def clear(cls) -> None:
        """Clear all registered harnesses.

        This is primarily useful for testing.
        """
        cls._harnesses.clear()


# Type alias for harness factory functions
HarnessFactory = Callable[[HarnessConfig], AgentHarness]


def register_harness(name: str) -> Callable[[type[AgentHarness]], type[AgentHarness]]:
    """Decorator to register a harness class.

    Args:
        name: Unique name for the harness

    Returns:
        Decorator function

    Example:
        @register_harness("pi_sdk")
        class PiSdkHarness(AgentHarness):
            async def execute(self, request):
                ...
    """

    def decorator(cls: type[AgentHarness]) -> type[AgentHarness]:
        HarnessRegistry.register(name, cls)
        return cls

    return decorator


# Convenience function to create a factory for a registered harness
def create_factory(name: str) -> HarnessFactory:
    """Create a factory function for a registered harness.

    This is useful for dependency injection and testing scenarios
    where you need a callable that creates harness instances.

    Args:
        name: Name of the registered harness

    Returns:
        Factory function that creates harness instances

    Example:
        factory = create_factory("pi_sdk")
        harness = factory(config)
    """

    def factory(config: HarnessConfig) -> AgentHarness:
        return HarnessRegistry.get(name, config)

    return factory
