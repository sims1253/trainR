from dataclasses import dataclass, field
from enum import Enum


class SandboxProfile(str, Enum):
    """Sandbox security profiles."""

    STRICT = "strict"  # Default: non-root, readonly FS, no network
    NETWORKED = "networked"  # Explicit outbound network
    DEVELOPER = "developer"  # Relaxed local-debug


@dataclass
class SandboxPolicy:
    """Sandbox security policy configuration."""

    profile: SandboxProfile = SandboxProfile.STRICT

    # Security settings
    run_as_non_root: bool = True
    read_only_root_fs: bool = True
    drop_capabilities: list[str] = field(default_factory=lambda: ["ALL"])

    # Resource limits
    cpu_limit: str = "2"  # 2 CPUs
    memory_limit: str = "4g"  # 4GB
    pids_limit: int = 256

    # Network settings
    network_enabled: bool = False

    # Writable paths
    writable_paths: list[str] = field(default_factory=lambda: ["/tmp", "/var/tmp"])

    @classmethod
    def from_profile(cls, profile: SandboxProfile) -> "SandboxPolicy":
        """Create a policy from a named profile."""
        if profile == SandboxProfile.STRICT:
            return cls(profile=profile)
        elif profile == SandboxProfile.NETWORKED:
            return cls(
                profile=profile,
                network_enabled=True,
            )
        elif profile == SandboxProfile.DEVELOPER:
            return cls(
                profile=profile,
                run_as_non_root=False,
                read_only_root_fs=False,
                network_enabled=True,
                drop_capabilities=[],
            )
        return cls(profile=profile)


def get_sandbox_policy(profile: SandboxProfile | str | None = None) -> SandboxPolicy:
    """Get a sandbox policy by profile name.

    Args:
        profile: Profile name or SandboxProfile enum. Defaults to STRICT.

    Returns:
        Configured SandboxPolicy instance.
    """
    if profile is None:
        profile = SandboxProfile.STRICT
    elif isinstance(profile, str):
        profile = SandboxProfile(profile)
    return SandboxPolicy.from_profile(profile)
