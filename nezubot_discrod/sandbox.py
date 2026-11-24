import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

from nezubot_discrod.config import SandboxSettings

CONTAINER_LABEL = f"nezubot-discrod.instance={uuid.uuid4().hex}"


def run_command_enabled() -> bool:
    """run_command is offered only when RUN_COMMAND_ENABLED is truthy (default on)."""
    return os.getenv("RUN_COMMAND_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


class SandboxLifecycle:
    def __init__(self):
        self.condition = threading.Condition()
        self.accepting = True
        self.active = {}

    def start(self, name, arguments):
        with self.condition:
            if not self.accepting:
                raise RuntimeError("Sandbox is stopping for restart")
            process = subprocess.Popen(
                arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self.active[name] = process
            return process

    def finished(self, name):
        with self.condition:
            self.active.pop(name, None)
            self.condition.notify_all()

    def shutdown(self, timeout=30):
        with self.condition:
            self.accepting = False
            self.condition.notify_all()
            self.condition.wait_for(lambda: not self.active, timeout=timeout)
            remaining = list(self.active.values())
        for process in remaining:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        with self.condition:
            if not self.condition.wait_for(lambda: not self.active, timeout=20):
                raise RuntimeError("Sandbox cleanup did not finish; restart cancelled")
        for remove in (True, False):
            result = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            containers = result.stdout.split()
            if not containers:
                return
            if not remove:
                raise RuntimeError("Sandbox containers remain; restart cancelled")
            subprocess.run(
                ["docker", "rm", "--force", *containers],
                capture_output=True, timeout=10, check=False,
            )


_lifecycle = SandboxLifecycle()


def accepting_work():
    return _lifecycle.accepting


def shutdown_sandboxes(timeout=30):
    _lifecycle.shutdown(timeout)


def workspace_user():
    result = subprocess.run(
        ["docker", "info", "--format", "{{json .SecurityOptions}}"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    options = json.loads(result.stdout)
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ValueError("Invalid Docker security options")
    if any(option.split(",", 1)[0] in {"rootless", "name=rootless"} for option in options):
        return "0:0"
    return f"{os.getuid()}:{os.getgid()}"


def docker_arguments(
    settings: SandboxSettings,
    name: str,
    command: str,
    workspace: Path | None = None,
) -> list[str]:
    user = "65534:65534"
    mounts = ["--tmpfs", "/work:rw,nosuid,nodev,size=64m,mode=1777"]
    if workspace is not None:
        path = str(workspace.resolve(strict=True))
        if os.getuid() == 0 or "," in path or not workspace.is_dir():
            raise ValueError("Self-modification requires a non-root host user and a safe directory")
        user = workspace_user()
        mounts = [
            "--mount",
            f"type=bind,src={path},dst=/work",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
        ]
    return [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        name,
        "--label",
        CONTAINER_LABEL,
        "--network",
        "none",
        "--memory",
        settings.memory,
        "--memory-swap",
        settings.memory,
        "--cpus",
        str(settings.cpus),
        "--pids-limit",
        "64",
        "--read-only",
        *mounts,
        "--workdir",
        "/work",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        user,
        "--env",
        "HOME=/work",
        "--log-driver",
        "none",
        settings.image,
        "/bin/sh",
        "-c",
        command,
    ]


def truncate_output(output: str, limit: int) -> str:
    marker = "\n[output truncated]"
    return output if len(output) <= limit else output[: limit - len(marker)] + marker


def run_command(
    command: str,
    settings: SandboxSettings,
    workspace: Path | None = None,
) -> str:
    name = f"discord-ai-{uuid.uuid4().hex}"
    try:
        process = _lifecycle.start(name, docker_arguments(settings, name, command, workspace))
    except OSError:
        return "Sandbox unavailable: could not start Docker."
    try:
        return collect_output(process, name, settings)
    finally:
        _lifecycle.finished(name)


def collect_output(process, name, settings):
    captured = bytearray()

    def drain(stream):
        with stream:
            while chunk := stream.read(4096):
                remaining = settings.output_limit * 4 - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])

    reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=settings.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # Also clean up after CLI failures; no container may outlive a tool call.
        for action in ("kill", "rm"):
            args = ["docker", action]
            if action == "rm":
                args.append("--force")
            try:
                subprocess.run(
                    [*args, name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=5)
    body = captured.decode("utf-8", errors="replace")
    status = "Timed out" if timed_out else f"Exit code: {process.returncode}"
    return truncate_output(f"{status}\n{body}", settings.output_limit)
