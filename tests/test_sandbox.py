import io
import subprocess
import threading
from unittest.mock import Mock, patch

import pytest

from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.sandbox import (
    CONTAINER_LABEL,
    SandboxLifecycle,
    docker_arguments,
    run_command,
    truncate_output,
    workspace_user,
)


def test_arguments():
    args = docker_arguments(SandboxSettings(), "test-name", "echo hi; id")
    for flag, value in {
        "--network": "none",
        "--memory": "128m",
        "--memory-swap": "128m",
        "--cpus": "0.5",
        "--pids-limit": "64",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--user": "65534:65534",
        "--workdir": "/work",
        "--name": "test-name",
        "--pull": "never",
        "--label": CONTAINER_LABEL,
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--rm" in args and "--read-only" in args
    assert args[args.index("--tmpfs") + 1].startswith("/work:rw,nosuid,nodev,size=64m")
    assert args[-4:] == ["discord-ai-sandbox:local", "/bin/sh", "-c", "echo hi; id"]
    assert "--privileged" not in args and "--volume" not in args


@pytest.mark.parametrize("body", ["", "hello", "x" * 500])
def test_truncation(body):
    result = truncate_output(body, 100)
    assert len(result) <= 100
    assert result == body if len(body) <= 100 else result.endswith("[output truncated]")


def fake_process(body=b"hello", timeout=False):
    process = Mock(stdout=io.BytesIO(body), returncode=0)
    process.wait.side_effect = [subprocess.TimeoutExpired("docker", 1), 0] if timeout else [0, 0]
    process.poll.return_value = None if timeout else 0
    return process


def test_timeout_kills_named_container():
    process = fake_process(timeout=True)
    with (
        patch("nezubot_discrod.sandbox.subprocess.Popen", return_value=process) as start,
        patch("nezubot_discrod.sandbox.subprocess.run") as cleanup,
    ):
        result = run_command("sleep 99", SandboxSettings(timeout=0.01))
    name = start.call_args.args[0][start.call_args.args[0].index("--name") + 1]
    assert cleanup.call_args_list[0].args[0] == ["docker", "kill", name]
    assert cleanup.call_args_list[1].args[0] == ["docker", "rm", "--force", name]
    process.kill.assert_called_once()
    assert result.startswith("Timed out")
    assert "shell" not in start.call_args.kwargs


def test_capture_is_bounded():
    with (
        patch("nezubot_discrod.sandbox.subprocess.Popen", return_value=fake_process(b"x" * 100000)),
        patch("nezubot_discrod.sandbox.subprocess.run"),
    ):
        result = run_command("yes", SandboxSettings(output_limit=100))
    assert len(result) == 100
    assert result.endswith("[output truncated]")


def test_docker_missing():
    with patch("nezubot_discrod.sandbox.subprocess.Popen", side_effect=FileNotFoundError):
        assert "unavailable" in run_command("id", SandboxSettings())


@pytest.mark.parametrize("options", ['["rootless"]', '["name=rootless,other=value"]'])
def test_rootless_workspace_user(options):
    with patch("nezubot_discrod.sandbox.subprocess.run", return_value=Mock(stdout=options)):
        assert workspace_user() == "0:0"


@pytest.mark.parametrize("options", ["", "null", "{}", "[123]"])
def test_workspace_user_rejects_invalid_docker_info(options):
    with patch("nezubot_discrod.sandbox.subprocess.run", return_value=Mock(stdout=options)):
        with pytest.raises(ValueError):
            workspace_user()


@pytest.mark.parametrize(
    "failure",
    [
        OSError(),
        subprocess.TimeoutExpired("docker", 10),
        subprocess.CalledProcessError(1, "docker"),
    ],
)
def test_workspace_user_does_not_guess_after_probe_failure(failure):
    with patch("nezubot_discrod.sandbox.subprocess.run", side_effect=failure):
        with pytest.raises(type(failure)):
            workspace_user()


def test_shutdown_drains_inflight_and_refuses_new_commands():
    lifecycle = SandboxLifecycle()
    process = Mock()
    with patch("nezubot_discrod.sandbox.subprocess.Popen", return_value=process) as start:
        lifecycle.start("inflight", ["docker", "run"])

        def finish():
            with lifecycle.condition:
                assert lifecycle.condition.wait_for(lambda: not lifecycle.accepting, timeout=2)
            with pytest.raises(RuntimeError, match="stopping"):
                lifecycle.start("late", ["docker", "run"])
            lifecycle.finished("inflight")

        worker = threading.Thread(target=finish)
        worker.start()
        with patch("nezubot_discrod.sandbox.subprocess.run", return_value=Mock(stdout="")):
            lifecycle.shutdown(timeout=2)
        worker.join(timeout=2)
        assert not worker.is_alive()
    start.assert_called_once()
    process.kill.assert_not_called()
    assert not lifecycle.active


def test_shutdown_force_removes_labelled_containers_after_bounded_wait():
    lifecycle = SandboxLifecycle()
    process = Mock()
    process.poll.return_value = None
    lifecycle.active["inflight"] = process
    process.wait.side_effect = lambda **kwargs: lifecycle.finished("inflight")
    with patch(
        "nezubot_discrod.sandbox.subprocess.run",
        side_effect=[Mock(stdout="abc\ndef\n"), Mock(), Mock(stdout="")],
    ) as docker:
        lifecycle.shutdown(timeout=0)
    process.kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)
    assert docker.call_args_list[0].args[0] == [
        "docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}"
    ]
    assert docker.call_args_list[1].args[0] == ["docker", "rm", "--force", "abc", "def"]
    assert not lifecycle.accepting


@pytest.mark.parametrize("failure", [OSError("docker unavailable"), Mock(stdout="remaining")])
def test_shutdown_fails_closed_when_cleanup_cannot_be_verified(failure):
    lifecycle = SandboxLifecycle()
    with patch(
        "nezubot_discrod.sandbox.subprocess.run", side_effect=[Mock(stdout="abc"), Mock(), failure]
    ), pytest.raises((OSError, RuntimeError)):
        lifecycle.shutdown(timeout=0)
    assert not lifecycle.accepting


@pytest.mark.integration
def test_real_container_offline_and_hardened():
    try:
        available = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker unavailable")
    if available.returncode:
        pytest.skip("Docker daemon unavailable")
    command = """python3 - <<'PY'
import os, socket
assert os.getuid() == 65534
assert os.listdir('/sys/class/net') == ['lo']
try:
    socket.create_connection(('1.1.1.1', 443), timeout=1)
except OSError:
    print('NETWORK_UNREACHABLE')
else:
    raise AssertionError('external network reachable')
try:
    open('/root-write-test', 'w')
except OSError:
    pass
else:
    raise AssertionError('root filesystem writable')
open('/work/test', 'w').write('ok')
PY"""
    result = run_command(command, SandboxSettings())
    assert result.startswith("Exit code: 0"), result
    assert "NETWORK_UNREACHABLE" in result
    assert "Timed out" in run_command("sleep 30", SandboxSettings(timeout=0.3))
    check = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=discord-ai-"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not check.stdout.strip()
