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
            cmd.extend(["--user", "nobody"])

        if self.policy.read_only_root_fs:
            cmd.append("--read-only")

        for cap in self.policy.drop_capabilities:
            cmd.extend(["--cap-drop", cap])

        # Resource limits
        cmd.extend(["--cpus", self.policy.cpu_limit])
        cmd.extend(["--memory", self.policy.memory_limit])
        cmd.extend(["--pids-limit", str(self.policy.pids_limit)])

        # Network
        if not self.policy.network_enabled:
            cmd.append("--network=none")

        # Writable paths (tmpfs)
        for path in self.policy.writable_paths:
            cmd.extend(["--tmpfs", f"{path}:rw,size=256m"])

        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Volumes
        if volumes:
            for src, dest, mode in volumes:
                cmd.extend(["-v", f"{src}:{dest}:{mode}"])

        # Image and command
        cmd.append(image)
        cmd.extend(command)

        return cmd


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
