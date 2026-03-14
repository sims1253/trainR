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
    # Optional user override. If not set, DockerCommandBuilder resolves to host UID:GID.
    run_as_user: str | None = None
    read_only_root_fs: bool = True
    drop_capabilities: list[str] = field(default_factory=lambda: ["ALL"])
    no_new_privileges: bool = True

    # Resource limits
    cpu_limit: str = "2"  # 2 CPUs
    memory_limit: str = "4g"  # 4GB
    # Match memory swap to memory by default to prevent swap abuse.
    memory_swap_limit: str = "4g"
    pids_limit: int = 256
    nofile_ulimit_soft: int = 1024
    nofile_ulimit_hard: int = 4096

    # Network settings
    network_enabled: bool = False
    network_name: str = "bridge"

    # Writable paths
    writable_paths: list[str] = field(default_factory=lambda: ["/tmp", "/var/tmp"])
    tmpfs_size_mb: int = 256

    # Bind mount controls
    read_only_bind_mounts: bool = True
    writable_bind_mount_destinations: list[str] = field(default_factory=lambda: ["/workspace"])
    allow_docker_socket_mount: bool = False

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
                no_new_privileges=False,
                read_only_bind_mounts=False,
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
