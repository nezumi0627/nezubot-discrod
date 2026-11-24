import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.policy import (
    PROTECTED_PATHS,
    PolicyViolation,
    guard_host_write,
    host_write_forbidden,
    protected_path,
    validate_workspace,
)
from nezubot_discrod.sandbox import docker_arguments
from nezubot_discrod.selfmod import host_git


@pytest.mark.parametrize(
    "path",
    sorted(PROTECTED_PATHS)
    + [
        "new/conftest.py",
        "new/pyproject.toml",
        ".github/workflows/new.yml",
        ".gitattributes",
        "nested/.gitmodules",
        "nested/hooks/pre-commit",
        "tests/__init__.py",
    ],
)
def test_protected_paths(path):
    assert protected_path(path)


@pytest.mark.parametrize("workspace,rootless", [(False, False), (True, False), (True, True)])
def test_docker_policy(tmp_path, workspace, rootless):
    options = '["name=rootless"]' if rootless else "[]"
    with (
        patch("nezubot_discrod.sandbox.os.getuid", return_value=1000),
        patch("nezubot_discrod.sandbox.os.getgid", return_value=1000),
        patch("nezubot_discrod.sandbox.subprocess.run", return_value=Mock(stdout=options)),
    ):
        args = docker_arguments(
            SandboxSettings(), "policy-test", "id", tmp_path if workspace else None
        )
    for flag, value in {
        "--network": "none",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--read-only" in args
    user = args[args.index("--user") + 1]
    assert user == ("0:0" if rootless else "1000:1000" if workspace else "65534:65534")
    # Rootless UID 0 maps to the non-root daemon owner on the host.
    if not rootless:
        assert user.split(":")[0] != "0"
    mounts = [args[i + 1] for i, item in enumerate(args) if item == "--mount"]
    assert mounts == ([f"type=bind,src={tmp_path},dst=/work"] if workspace else [])
    assert not {"-v", "--volume", "--volumes-from", "--privileged"}.intersection(args)
    assert not any("docker.sock" in arg for arg in args)


def test_host_write_guard_resolves_symlinks_and_parent_paths(tmp_path):
    checkout, data = tmp_path / "bot", tmp_path / "data"
    checkout.mkdir()
    data.mkdir()
    (data / "alias").symlink_to(checkout, target_is_directory=True)
    (checkout / "escape").symlink_to(data, target_is_directory=True)
    for target in (
        checkout / "README.md",
        data / "alias" / "anything",
        data / ".." / "bot" / "file",
        checkout / "escape" / "file",
    ):
        assert host_write_forbidden(target, checkout=checkout)
        with pytest.raises(PermissionError):
            guard_host_write(target, checkout=checkout, allowed_root=data)
    with pytest.raises(PermissionError):
        guard_host_write(data / ".." / "outside", checkout=checkout, allowed_root=data)
    assert (
        guard_host_write(data / "new.md", checkout=checkout, allowed_root=data) == data / "new.md"
    )
    actual_repo = Path(__file__).resolve().parent.parent
    assert host_write_forbidden(actual_repo / "README.md", checkout=checkout)


def test_host_write_guard_fails_closed_on_symlink_loop(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    assert host_write_forbidden(loop, checkout=tmp_path / "bot")
    with pytest.raises(PermissionError):
        guard_host_write(loop, checkout=tmp_path / "bot", allowed_root=tmp_path)


@pytest.fixture
def repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git unavailable")
    work = tmp_path / "work"
    work.mkdir()
    host_git(["init", "--initial-branch=main"], cwd=work)
    (work / "nezubot_discrod").mkdir()
    (work / "nezubot_discrod" / "sandbox.py").write_text("original")
    (work / "README.md").write_text("original")
    host_git(["add", "--", "nezubot_discrod/sandbox.py", "README.md"], cwd=work)
    host_git(["commit", "-m", "base"], cwd=work)
    metadata = tmp_path / "metadata"
    shutil.move(work / ".git", metadata)

    def git(args):
        return host_git(args, cwd=work, git_dir=metadata)

    return work, git


def check_repository(work, git):
    raw = git(
        [
            "diff",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            "HEAD",
        ]
    )
    others = git(["ls-files", "--others", "-z"])
    validate_workspace(work, raw, others)


@pytest.mark.parametrize("name", [
    "server_log.py", "server_log_dense.py", "server_log_discord.py", "server_log_tools.py",
])
def test_rag_workspace_changes_pass_without_feature_ast_check(repository, name):
    work, git = repository
    (work / "nezubot_discrod" / name).write_text("import os\n")
    check_repository(work, git)


@pytest.mark.parametrize(
    "change",
    ["modify", "delete", "rename", "add", "ignored", "symlink", "submodule", "attributes", "hooks"],
)
def test_real_git_security_changes_rejected(repository, change):
    work, git = repository
    protected = work / "nezubot_discrod" / "sandbox.py"
    expected = "nezubot_discrod/sandbox.py"
    if change == "modify":
        protected.write_text("changed")
    elif change == "delete":
        protected.unlink()
    elif change == "rename":
        protected.rename(work / "moved.py")
    elif change in {"add", "ignored"}:
        expected = "conftest.py"
        (work / expected).write_text("malicious plugin")
        if change == "ignored":
            (work / ".gitignore").write_text("conftest.py\n")
    elif change == "symlink":
        expected = "link"
        (work / expected).symlink_to("/etc/passwd")
    elif change == "submodule":
        expected = "vendor"
        (work / expected).mkdir()
        sha = git(["rev-parse", "HEAD"])
        git(["update-index", "--add", "--cacheinfo", f"160000,{sha},vendor"])
    elif change == "attributes":
        expected = ".gitattributes"
        (work / expected).write_text("* filter=custom\n")
    else:
        expected = "hooks"
        (work / expected).mkdir()
        (work / expected / "pre-commit").write_text("malicious")
    with pytest.raises(PolicyViolation) as error:
        check_repository(work, git)
    assert expected in error.value.paths


def test_empty_and_normal_changes_allowed(repository):
    work, git = repository
    check_repository(work, git)
    (work / "README.md").write_text("documentation update")
    (work / "docs").mkdir()
    (work / "docs/new.md").write_text("new documentation")
    check_repository(work, git)


@pytest.mark.parametrize(
    "path",
    [
        "nezubot_discrod/prompts/__init__.pyc",
        "nezubot_discrod/prompts/payload.py",
        "nezubot_discrod/sandbox.so",
        "nezubot_discrod/policy.pyc",
        "nezubot_discrod/prompts.pyo",
        "nezubot_discrod/features/ping.pyc",
        "nezubot_discrod/features/loader.pth",
        "nezubot_discrod/features/native.pyd",
        "nezubot_discrod/features/native.abi3.so",
        "tests/features/__pycache__/test_ping.pyc",
        "docs/import.pth",
        "sitecustomize.py",
        "usercustomize/loader.py",
        "json.py",
        "discord/helpers.py",
        "asyncio/__init__.pyc",
        "new_package/helper.py",
        "docs/module.py",
        "tests/test_other.py",
    ],
)
def test_module_shadowing_and_nonallowlisted_paths_rejected(repository, path):
    work, git = repository
    target = work / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("malicious loader")
    assert protected_path(path)
    with pytest.raises(PolicyViolation):
        check_repository(work, git)


def test_empty_shadow_package_directory_is_rejected(repository):
    work, git = repository
    (work / "nezubot_discrod/prompts").mkdir()
    with pytest.raises(PolicyViolation) as error:
        check_repository(work, git)
    assert "nezubot_discrod/prompts" in error.value.paths


def test_plain_feature_and_feature_test_files_remain_editable(repository):
    work, git = repository
    for path in [
        "nezubot_discrod/features/hello.py",
        "tests/features/test_hello.py",
        "tests/test_feature_hello.py",
        "docs/hello.md",
    ]:
        target = work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# plain source or documentation\n")
        assert not protected_path(path)
    check_repository(work, git)


def test_invalid_diff_fails_closed(tmp_path):
    with pytest.raises(PolicyViolation):
        validate_workspace(tmp_path, "malformed", "")
