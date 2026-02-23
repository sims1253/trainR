"""Tool registry for managing tool profiles.

The ToolRegistry provides:
- Registration of available tools
- Resolution of tools by name and version
- Validation of tool configurations
- Discovery of tool profiles from directories
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from bench.profiles.tools import (
    ToolConfig,
    ToolProfile,
    ToolVersion,
    load_tool_profile,
)

logger = logging.getLogger(__name__)


class ToolRegistryError(Exception):
    """Base error for tool registry operations."""

    pass


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not found."""

    pass


class ToolVersionNotFoundError(ToolRegistryError):
    """Raised when a requested tool version is not found."""

    pass


class ToolValidationError(ToolRegistryError):
    """Raised when tool validation fails."""

    pass


class ToolRegistry:
    """
    Registry for managing tool profiles.

    The registry provides:
    - Tool registration and lookup
    - Version resolution
    - Configuration validation
    - Profile discovery from directories

    Example usage:
        registry = ToolRegistry()

        # Register a tool profile
        profile = ToolProfile(tool_id="r-cli", name="R CLI", version="v1")
        registry.register(profile)

        # Resolve a tool
        tool = registry.resolve("r-cli", version="v1")

        # List available tools
        tools = registry.list_tools()
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, dict[str, ToolProfile]] = {}
        # Structure: {tool_id: {full_version_key: ToolProfile}}
        # full_version_key includes variant if present: "v1" or "v1:variant"

    def _get_storage_key(self, profile: ToolProfile) -> str:
        """Get the storage key for a profile (includes variant if present)."""
        version = profile.get_version_string()
        if profile.variant:
            return f"{version}:{profile.variant}"
        return version

    def register(self, profile: ToolProfile) -> None:
        """Register a tool profile.

        Args:
            profile: ToolProfile to register

        Raises:
            ToolValidationError: If profile is invalid
        """
        if not profile.tool_id:
            raise ToolValidationError("Tool profile must have a tool_id")

        tool_id = profile.tool_id
        storage_key = self._get_storage_key(profile)

        if tool_id not in self._tools:
            self._tools[tool_id] = {}

        self._tools[tool_id][storage_key] = profile
        logger.debug(f"Registered tool: {tool_id}@{storage_key}")

    def unregister(self, tool_id: str, version: str | None = None) -> bool:
        """Unregister a tool profile.

        Args:
            tool_id: Tool ID to unregister
            version: Specific version to unregister, or None for all versions

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_id not in self._tools:
            return False

        if version is None:
            del self._tools[tool_id]
            logger.debug(f"Unregistered all versions of: {tool_id}")
            return True

        if version in self._tools[tool_id]:
            del self._tools[tool_id][version]
            logger.debug(f"Unregistered tool: {tool_id}@{version}")

            # Clean up empty tool entries
            if not self._tools[tool_id]:
                del self._tools[tool_id]

            return True

        return False

    def resolve(
        self,
        tool_id: str,
        version: str | ToolVersion | None = None,
        variant: str | None = None,
    ) -> ToolProfile:
        """Resolve a tool profile by ID and version.

        Args:
            tool_id: Tool ID to resolve
            version: Version to resolve (defaults to latest)
            variant: Optional variant name

        Returns:
            Resolved ToolProfile

        Raises:
            ToolNotFoundError: If tool is not found
            ToolVersionNotFoundError: If version is not found
        """
        if tool_id not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {tool_id}")

        # Convert version enum to string if needed
        version_str = None
        if version is not None:
            version_str = version.value if isinstance(version, ToolVersion) else version

        # Get all profiles for this tool
        profiles = self._tools[tool_id]

        if version_str:
            # If variant specified, look for exact match
            if variant:
                storage_key = f"{version_str}:{variant}"
                if storage_key in profiles:
                    return profiles[storage_key]
                # Try to find variant across any version
                for _key, p in profiles.items():
                    if p.variant == variant:
                        return p
                raise ToolVersionNotFoundError(
                    f"Tool variant not found: {tool_id}@{version_str}:{variant}"
                )

            # No variant specified - prefer base version (without variant)
            if version_str in profiles:
                return profiles[version_str]

            # Look for any profile with this base version (has variant)
            for _key, p in profiles.items():
                if p.get_version_string() == version_str and not p.variant:
                    return p

            # Last resort: return first profile with this version
            for key, p in profiles.items():
                if key.startswith(f"{version_str}:") or key == version_str:
                    return p

            available = list(profiles.keys())
            raise ToolVersionNotFoundError(
                f"Tool version not found: {tool_id}@{version_str}. Available versions: {available}"
            )
        else:
            # No version specified - get latest version, preferring base (non-variant)
            # Group profiles by version (without variant)
            version_profiles: dict[str, list[tuple[str, ToolProfile]]] = {}
            for key, p in profiles.items():
                base_ver = p.get_version_string()
                if base_ver not in version_profiles:
                    version_profiles[base_ver] = []
                version_profiles[base_ver].append((key, p))

            # Sort versions to get latest
            sorted_versions = sorted(version_profiles.keys(), reverse=True)
            if not sorted_versions:
                raise ToolVersionNotFoundError(f"No versions found for tool: {tool_id}")

            # Get profiles for latest version, prefer non-variant
            latest_version = sorted_versions[0]
            candidates = version_profiles[latest_version]

            # Return base (non-variant) if available, otherwise first variant
            for _key, p in candidates:
                if not p.variant:
                    return p
            return candidates[0][1]

    def get(self, tool_id: str, version: str | None = None) -> ToolProfile | None:
        """Get a tool profile without raising exceptions.

        Args:
            tool_id: Tool ID to get
            version: Version to get (defaults to latest)

        Returns:
            ToolProfile if found, None otherwise
        """
        try:
            return self.resolve(tool_id, version)
        except ToolRegistryError:
            return None

    def list_tools(self) -> list[str]:
        """List all registered tool IDs.

        Returns:
            List of tool IDs
        """
        return list(self._tools.keys())

    def list_versions(self, tool_id: str) -> list[str]:
        """List all versions of a tool.

        Args:
            tool_id: Tool ID to list versions for

        Returns:
            List of version strings

        Raises:
            ToolNotFoundError: If tool is not found
        """
        if tool_id not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {tool_id}")

        # Return unique base versions (without variant suffix)
        versions = set()
        for _key, p in self._tools[tool_id].items():
            versions.add(p.get_version_string())
        return sorted(versions)

    def list_variants(self, tool_id: str, version: str | None = None) -> list[str]:
        """List all variants of a tool version.

        Args:
            tool_id: Tool ID to list variants for
            version: Version to list variants for (defaults to latest)

        Returns:
            List of variant names
        """
        profile = self.get(tool_id, version)
        if not profile:
            return []

        variants = []
        base_version = profile.get_version_string()

        for _v, p in self._tools.get(tool_id, {}).items():
            if p.variant and p.base_version == base_version:
                variants.append(p.variant)

        return variants

    def validate_config(self, tool_id: str, config: dict[str, Any]) -> ToolConfig:
        """Validate a tool configuration.

        Args:
            tool_id: Tool ID to validate config for
            config: Configuration dictionary to validate

        Returns:
            Validated ToolConfig

        Raises:
            ToolNotFoundError: If tool is not found
            ToolValidationError: If configuration is invalid
        """
        if tool_id not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {tool_id}")

        try:
            return ToolConfig.model_validate(config)
        except Exception as e:
            raise ToolValidationError(f"Invalid configuration for {tool_id}: {e}") from e

    def discover_profiles(self, directory: str | Path) -> int:
        """Discover and register tool profiles from a directory.

        Args:
            directory: Directory to search for profile files

        Returns:
            Number of profiles registered
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Profile directory not found: {directory}")
            return 0

        count = 0
        for profile_file in directory.glob("**/*.yaml"):
            try:
                profile = load_tool_profile(profile_file)
                self.register(profile)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file}: {e}")

        logger.info(f"Discovered {count} tool profiles from {directory}")
        return count

    def get_all_profiles(self) -> list[ToolProfile]:
        """Get all registered tool profiles.

        Returns:
            List of all ToolProfile instances
        """
        profiles = []
        for tool_id in self._tools:
            for version in self._tools[tool_id]:
                profiles.append(self._tools[tool_id][version])
        return profiles

    def get_fingerprints(self) -> dict[str, str]:
        """Get fingerprints for all registered tools.

        Returns:
            Dictionary mapping full tool IDs to fingerprints
        """
        return {profile.get_full_id(): profile.fingerprint for profile in self.get_all_profiles()}

    def export_registry(self) -> dict[str, Any]:
        """Export the registry state as a dictionary.

        Returns:
            Dictionary representation of the registry
        """
        return {
            "tools": {
                tool_id: {version: profile.to_dict() for version, profile in versions.items()}
                for tool_id, versions in self._tools.items()
            },
            "fingerprints": self.get_fingerprints(),
        }

    def save_registry(self, path: str | Path) -> None:
        """Save the registry state to a YAML file.

        Args:
            path: Path to save the registry to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.export_registry(), f, default_flow_style=False)

        logger.info(f"Saved registry to {path}")

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ToolRegistry":
        """Create a registry and populate it from a directory.

        Args:
            directory: Directory containing tool profile files

        Returns:
            Populated ToolRegistry
        """
        registry = cls()
        registry.discover_profiles(directory)
        return registry

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ToolRegistry":
        """Load a registry from a YAML file.

        Args:
            path: Path to the registry YAML file

        Returns:
            Loaded ToolRegistry
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        registry = cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        tools_data = data.get("tools", {})
        for _tool_id, versions in tools_data.items():
            for _version, profile_data in versions.items():
                profile = ToolProfile.from_dict(profile_data)
                registry.register(profile)

        return registry

    def __len__(self) -> int:
        """Return the total number of registered profiles."""
        return sum(len(versions) for versions in self._tools.values())

    def __contains__(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        return tool_id in self._tools

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        tool_count = len(self.list_tools())
        profile_count = len(self)
        return f"ToolRegistry(tools={tool_count}, profiles={profile_count})"


# Global registry instance
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry.

    Returns:
        Global ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global tool registry."""
    global _global_registry
    _global_registry = None
