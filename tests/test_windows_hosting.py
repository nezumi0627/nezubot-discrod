import io
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "windows"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="WSL/Linux hosting scripts")


@pytest.fixture
def hosting_home():
    if not str(Path.home()).startswith("/home/") or os.getuid() == 0:
        pytest.skip("Requires a non-root /home user, as used by the WSL deployment")
    with tempfile.TemporaryDirectory(prefix="hosting-test-", dir=Path.home()) as directory:
        home = Path(directory)
        venv = home / "nezubot-discrod" / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").symlink_to(sys.executable)
        yield home


def archive(home, entries):
    path = home / "data.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        for name, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                output.addfile(member)
            else:
                body = b"test database content"
                member.size = len(body)
                output.addfile(member, io.BytesIO(body))
    return path


def run_import(home, path):
    return subprocess.run(
        ["bash", str(SCRIPTS / "import-data.sh"), str(path)],
        env={**os.environ, "HOME": str(home)}, capture_output=True, text=True, check=False,
    )


def test_import_private_files_and_refuse_overwrite(hosting_home):
    home = hosting_home
    path = archive(home, [(".nezubot-discrod/wiki.db", "file")])
    result = run_import(home, path)
    assert result.returncode == 0, result.stderr
    data = home / ".nezubot-discrod" / "wiki.db"
    assert data.read_bytes() == b"test database content"
    assert data.stat().st_mode & 0o777 == 0o600
    assert data.parent.stat().st_mode & 0o777 == 0o700
    assert run_import(home, path).returncode != 0


@pytest.mark.parametrize("entries", [
    [],
    [(".nezubot-discrod/../escape", "file")],
    [("/tmp/escape", "file")],
    [(".nezubot-discrod/wiki.db", "link")],
    [(".nezubot-discrod/.env", "file")],
    [(".nezubot-discrod/wiki.db", "file")] * 2,
])
def test_import_rejects_empty_and_unsafe_archives(hosting_home, entries):
    result = run_import(hosting_home, archive(hosting_home, entries))
    assert result.returncode != 0
    assert not (hosting_home / ".nezubot-discrod").exists()
    assert not list(hosting_home.glob(".data-import-*"))


def test_shell_scripts_parse():
    for script in SCRIPTS.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_run_script_refuses_non_home_path(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run-wsl.sh")],
        env={**os.environ, "HOME": str(tmp_path)}, capture_output=True, check=False,
    )
    assert result.returncode != 0


@pytest.mark.integration
@pytest.mark.parametrize("behavior", ["graceful", "force", "empty"])
def test_stop_cleans_only_its_instance(hosting_home, behavior):
    from nezubot_discrod.config import SandboxSettings
    from nezubot_discrod.sandbox import docker_arguments

    try:
        available = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker unavailable")
    if available.returncode:
        pytest.skip("Docker daemon unavailable")
    home = hosting_home
    repo = home / "nezubot-discrod"
    scripts = repo / "scripts" / "windows"
    scripts.mkdir(parents=True)
    for name in ("run-wsl.sh", "run-bot.py"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    package = repo / "nezubot_discrod"
    package.mkdir()
    (package / "__init__.py").write_text("")
    label = f"nezubot-discrod.instance={uuid.uuid4().hex}"
    name = f"discord-ai-{uuid.uuid4().hex}"
    unrelated = f"discord-ai-{uuid.uuid4().hex}"
    args = docker_arguments(SandboxSettings(), name, "sleep 300")
    args.insert(2, "-d")
    args[args.index("--label") + 1] = label
    other_args = args.copy()
    other_args[other_args.index("--name") + 1] = unrelated
    other_args[other_args.index("--label") + 1] = (
        f"nezubot-discrod.instance={uuid.uuid4().hex}"
    )
    (package / "sandbox.py").write_text(
        f"from pathlib import Path\nCONTAINER_LABEL = {label!r}\n"
        "def shutdown_sandboxes(timeout):\n"
        "    Path.home().joinpath('cleanup-called').touch()\n"
    )
    (package / "bot.py").write_text(
        "import signal, subprocess, time\nfrom pathlib import Path\n"
        "def main():\n"
        + ("    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
           "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if behavior == "force" else "")
        + (f"    subprocess.run({args!r}, check=True, capture_output=True)\n"
           if behavior != "empty" else "")
        + "    Path.home().joinpath('ready').touch()\n"
        "    while True: time.sleep(1)\n"
    )
    process = None
    try:
        subprocess.run(other_args, capture_output=True, check=True, timeout=20)
        process = subprocess.Popen(
            ["bash", str(scripts / "run-wsl.sh")],
            env={**os.environ, "HOME": str(home), "PYTHONPATH": str(repo)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 20
        while not (home / "ready").exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.communicate()[0].decode())
            time.sleep(0.05)
        assert (home / "ready").exists()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=40)
        assert process.returncode == 0, output.decode()
        assert (home / "cleanup-called").exists() == (behavior != "force")
        remaining = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={label}"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        assert not remaining.stdout.strip()
        subprocess.run(
            ["docker", "inspect", unrelated], capture_output=True, check=True, timeout=10,
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.communicate(timeout=40)
        subprocess.run(
            ["docker", "rm", "--force", name, unrelated], capture_output=True, timeout=10,
        )
