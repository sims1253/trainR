import os

from .policy import SandboxPolicy


class DockerCommandBuilder:
    """Builder for Docker run commands with sandbox policy enforcement."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy

    def build_run_command(
        self,
        image: str,
        command: list[str],
        env_vars: dict[str, str] | None = None,
        volumes: list[tuple[str, str, str]] | None = None,  # (src, dest, mode)
    ) -> list[str]:
        """Build a docker run command from policy.

        Args:
            image: Docker image to run.
            command: Command to execute in the container.
            env_vars: Environment variables to set.
            volumes: Volume mounts as (source, destination, mode) tuples.

        Returns:
            Complete docker run command as a list of strings.
        """
        cmd = ["docker", "run", "--rm"]

        # Security options
        if self.policy.run_as_non_root:
            cmd.extend(["--user", self._resolve_user()])

        if self.policy.read_only_root_fs:
            cmd.append("--read-only")

        for cap in self.policy.drop_capabilities:
            cmd.extend(["--cap-drop", cap])

        if self.policy.no_new_privileges:
            cmd.extend(["--security-opt", "no-new-privileges:true"])

        # Resource limits
        cmd.extend(["--cpus", self.policy.cpu_limit])
        cmd.extend(["--memory", self.policy.memory_limit])
        cmd.extend(["--memory-swap", self.policy.memory_swap_limit])
        cmd.extend(["--pids-limit", str(self.policy.pids_limit)])
        cmd.extend(
            [
                "--ulimit",
                f"nofile={self.policy.nofile_ulimit_soft}:{self.policy.nofile_ulimit_hard}",
            ]
        )

        # Network
        if not self.policy.network_enabled:
            cmd.append("--network=none")
        else:
            cmd.append(f"--network={self.policy.network_name}")

        # Writable paths (tmpfs)
        for path in self.policy.writable_paths:
            cmd.extend(["--tmpfs", f"{path}:rw,size={self.policy.tmpfs_size_mb}m"])

        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Volumes
        if volumes:
            for src, dest, mode in volumes:
                normalized_mode = mode.lower()
                self._validate_volume(src=src, dest=dest, mode=normalized_mode)
                cmd.extend(["-v", f"{src}:{dest}:{normalized_mode}"])

        # Image and command
        cmd.append(image)
        cmd.extend(command)

        return cmd

    def _resolve_user(self) -> str:
        """Resolve non-root user for Docker execution."""
        if self.policy.run_as_user:
            return self.policy.run_as_user
        try:
            uid = os.getuid()
            gid = os.getgid()
            if uid == 0:
                # Avoid running as root inside container even when host process is root.
                return "65534:65534"
            return f"{uid}:{gid}"
        except AttributeError:
            # Fallback for non-POSIX environments.
            return "nobody"

    def _validate_volume(self, src: str, dest: str, mode: str) -> None:
        """Validate bind mounts against sandbox policy."""
        if mode not in {"ro", "rw"}:
            raise ValueError(f"Invalid volume mode '{mode}'. Expected 'ro' or 'rw'.")

        if (
            self._is_docker_socket(src) or self._is_docker_socket(dest)
        ) and not self.policy.allow_docker_socket_mount:
            raise ValueError("Docker socket mounts are not allowed by sandbox policy.")

        if (
            mode == "rw"
            and self.policy.read_only_bind_mounts
            and dest not in self.policy.writable_bind_mount_destinations
        ):
            allowed = ", ".join(self.policy.writable_bind_mount_destinations)
            raise ValueError(
                "Writable bind mounts are restricted by sandbox policy. "
                f"Destination '{dest}' is not allowed. Allowed destinations: {allowed}"
            )

    @staticmethod
    def _is_docker_socket(path: str) -> bool:
        docker_sockets = {"/var/run/docker.sock", "/run/docker.sock"}
        return path in docker_sockets


def build_docker_command(
    image: str,
    command: list[str],
    profile: SandboxPolicy | None = None,
    env_vars: dict[str, str] | None = None,
    volumes: list[tuple[str, str, str]] | None = None,
) -> list[str]:
    """Convenience function to build a docker command with default policy.

    Args:
        image: Docker image to run.
        command: Command to execute in the container.
        profile: Sandbox policy to use. Defaults to STRICT.
        env_vars: Environment variables to set.
        volumes: Volume mounts as (source, destination, mode) tuples.

    Returns:
        Complete docker run command as a list of strings.
    """
    if profile is None:
        from .policy import get_sandbox_policy

        profile = get_sandbox_policy()
    builder = DockerCommandBuilder(profile)
    return builder.build_run_command(image, command, env_vars, volumes)
