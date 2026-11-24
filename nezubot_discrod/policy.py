"""Trusted host-side selfmod policy; never import policy from an AI-edited clone.

Every future host-side write tool (including any separately implemented wiki
writer) must call guard_host_write before writing. Its allowed root must be
trusted, and it must refuse all writes inside this checkout or SELF_CHECKOUT.
The resolved path check is defense in depth, not a race-free filesystem sandbox;
writers must also prevent concurrent symlink replacement (e.g. dirfd/O_NOFOLLOW).
Feature AST validation is defense in depth, not Python isolation. The owner must
review every PR: approved feature code is imported into the bot's host process.
"""

import ast
import json
import os
import stat
from pathlib import Path, PurePosixPath

PROTECTED_PATHS = frozenset(
    {
        "nezubot_discrod/__init__.py",
        "nezubot_discrod/assistant.py",
        "nezubot_discrod/bot.py",
        "nezubot_discrod/codex.py",
        "nezubot_discrod/config.py",
        "nezubot_discrod/improvement_ui.py",
        "nezubot_discrod/feature_loader.py",
        "nezubot_discrod/features/__init__.py",
        "nezubot_discrod/messages.py",
        "nezubot_discrod/policy.py",
        "nezubot_discrod/prompts.py",
        "nezubot_discrod/sandbox.py",
        "nezubot_discrod/selfmod.py",
        "Dockerfile",
        ".dockerignore",
        "pyproject.toml",
        "tests/test_assistant.py",
        "tests/test_bot.py",
        "tests/test_codex.py",
        "tests/test_policy.py",
        "tests/test_prompts.py",
        "tests/test_sandbox.py",
        "tests/test_selfmod.py",
        "tests/test_feature_registry_security.py",
    }
)

ALLOWED_FEATURE_IMPORTS = frozenset(
    {
        "discord",
        "asyncio",
        "datetime",
        "random",
        "re",
        "math",
        "json",
        "typing",
        "dataclasses",
    }
)
BLOCKED_FEATURE_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
    }
)
BLOCKED_FEATURE_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "importlib",
        "pathlib",
        "io",
        "builtins",
        "multiprocessing",
        "threading",
        "signal",
        "pty",
        "pickle",
        "marshal",
        "nezubot_discrod",
    }
)
BLOCKED_FEATURE_ATTRIBUTES = BLOCKED_FEATURE_IMPORTS | frozenset(
    {
        "create_subprocess_shell",
        "create_subprocess_exec",
        "subprocess",
        "subprocess_shell",
        "subprocess_exec",
        "run_in_executor",
        "to_thread",
        "open_connection",
        "open_unix_connection",
        "start_server",
        "start_unix_server",
        "create_connection",
        "create_server",
        "create_datagram_endpoint",
        "http",
        "_http",
        "loop",
        "get_event_loop",
        "get_running_loop",
    }
)


def feature_module(path):
    parts = PurePosixPath(path).parts
    return (
        len(parts) == 3
        and parts[:2] == ("nezubot_discrod", "features")
        and parts[2].endswith(".py")
        and parts[2] != "__init__.py"
    )


def validate_feature_source(path, source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise PolicyViolation([f"{path}:{getattr(error, 'lineno', 1)}: invalid Python"]) from error
    violations = []
    for node in ast.walk(tree):
        reason = None
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_FEATURE_IMPORTS for alias in node.names):
                reason = "forbidden import"
        elif isinstance(node, ast.ImportFrom):
            if (
                node.level
                or (node.module or "").split(".")[0] not in ALLOWED_FEATURE_IMPORTS
                or any(alias.name.lstrip("_") in BLOCKED_FEATURE_IMPORTS for alias in node.names)
                or any(alias.name in BLOCKED_FEATURE_ATTRIBUTES for alias in node.names)
            ):
                reason = "forbidden import"
        elif isinstance(node, ast.Name) and node.id in BLOCKED_FEATURE_NAMES:
            reason = f"forbidden name {node.id}"
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__")
            or node.attr.endswith("__")
            or node.attr in BLOCKED_FEATURE_NAMES
            or node.attr in BLOCKED_FEATURE_ATTRIBUTES
        ):
            reason = f"forbidden attribute {node.attr}"
        if isinstance(node, ast.alias) and (node.asname or node.name) in BLOCKED_FEATURE_NAMES:
            reason = "forbidden binding"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.arg)):
            if (node.arg if isinstance(node, ast.arg) else node.name) in BLOCKED_FEATURE_NAMES:
                reason = "forbidden binding"
        if reason:
            violations.append(f"{path}:{getattr(node, 'lineno', 1)}: {reason}")
    if violations:
        raise PolicyViolation(violations)


PROTECTED_NAMES = frozenset(
    {
        "conftest.py",
        "pyproject.toml",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "setup.py",
        "ruff.toml",
        ".ruff.toml",
        "sitecustomize.py",
        "usercustomize.py",
        "__init__.py",
        ".gitattributes",
        ".gitmodules",
        ".git",
        "hooks",
        ".githooks",
    }
)


class PolicyViolation(RuntimeError):
    def __init__(self, paths):
        self.paths = sorted(set(paths))
        super().__init__(
            "保護ポリシーにより変更を拒否しました。PR は作成しません。\n"
            + "\n".join(json.dumps(p, ensure_ascii=False) for p in self.paths)
        )


def protected_path(path):
    relative = PurePosixPath(path)
    return (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() in PROTECTED_PATHS
        or relative.parts[:1] == (".github",)
        or bool(PROTECTED_NAMES.intersection(relative.parts))
        or unsafe_loader_path(path)
        or shadowed_module(path)
        or not editable_path(path)
    )


def unsafe_loader_path(path):
    relative = PurePosixPath(path)
    return (
        relative.suffix.lower() in {".pyc", ".pyo", ".so", ".pyd", ".pth"}
        or "__pycache__" in relative.parts
        or any(
            PurePosixPath(part).stem in {"sitecustomize", "usercustomize"}
            for part in relative.parts
        )
    )


def shadowed_module(path):
    parts = PurePosixPath(path).parts
    for protected in PROTECTED_PATHS:
        module = PurePosixPath(protected)
        parent = module.parent.parts
        if module.suffix != ".py" or parts[: len(parent)] != parent or len(parts) <= len(parent):
            continue
        name = parts[len(parent)]
        if name == module.stem or (name.startswith(module.stem + ".") and name != module.name):
            return True
    return False


def editable_path(path):
    relative = PurePosixPath(path)
    parts = relative.parts
    if path in {
        "nezubot_discrod/server_log.py",
        "nezubot_discrod/server_log_dense.py",
        "nezubot_discrod/server_log_discord.py",
        "nezubot_discrod/server_log_tools.py",
    }:
        return True
    if (len(parts) == 2 and parts[0] == "tests" and relative.suffix == ".py"
            and relative.name.startswith("test_server_log")):
        return True
    if feature_module(path) and relative.stem.isidentifier():
        return True
    if relative.suffix == ".py" and (
        (len(parts) >= 3 and parts[:2] == ("tests", "features"))
        or (len(parts) == 2 and parts[0] == "tests" and relative.name.startswith("test_feature_"))
    ):
        return True
    return path == "README.md" or (
        len(parts) >= 2 and parts[0] == "docs" and relative.suffix == ".md"
    )


def host_write_forbidden(path, *, checkout):
    try:
        candidates = (Path(path).resolve(), Path(os.path.abspath(path)))
        roots = (
            Path(__file__).resolve().parent.parent,
            Path(checkout).resolve(),
            Path(os.path.abspath(checkout)),
        )
        return any(candidate.is_relative_to(root) for root in roots for candidate in candidates)
    except (OSError, RuntimeError):
        return True


def guard_host_write(path, *, checkout, allowed_root):
    try:
        candidate = Path(path).resolve()
        if host_write_forbidden(path, checkout=checkout) or not candidate.is_relative_to(
            Path(allowed_root).resolve()
        ):
            raise PermissionError("Host write outside the permitted data root or inside checkout")
        return candidate
    except (OSError, RuntimeError) as error:
        raise PermissionError("Could not safely resolve host write path") from error


def validate_workspace(workspace, raw_diff, untracked):
    blocked = set()
    changed = set(filter(None, untracked.split("\0")))
    records = raw_diff.split("\0") if raw_diff else []
    if records:
        if records.pop() != "" or len(records) % 2:
            raise PolicyViolation(["<invalid git diff>"])
        for header, path in zip(records[::2], records[1::2], strict=True):
            changed.add(path)
            fields = header.split()
            if len(fields) != 5 or not fields[0].startswith(":"):
                raise PolicyViolation(["<invalid git diff>"])
            if protected_path(path) or {fields[0][1:], fields[1]} & {"120000", "160000"}:
                blocked.add(path)
    for path in filter(None, untracked.split("\0")):
        if protected_path(path):
            blocked.add(path)
    count = 0

    def walk_error(error):
        raise PolicyViolation([str(error.filename)])

    for root, directories, files in os.walk(workspace, followlinks=False, onerror=walk_error):
        for name in directories + files:
            count += 1
            if count > 10000:
                raise PolicyViolation(["<workspace file limit exceeded>"])
            path = Path(root) / name
            mode = path.lstat().st_mode
            relative = path.relative_to(workspace).as_posix()
            if (
                unsafe_loader_path(relative)
                or shadowed_module(relative)
                or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                or name
                in {
                    ".git",
                    "hooks",
                    ".githooks",
                }
            ):
                blocked.add(relative)
    if blocked:
        raise PolicyViolation(blocked)
    for relative in sorted(changed):
        path = Path(workspace) / relative
        if feature_module(relative) and path.is_file():
            try:
                if path.stat().st_size > 256000:
                    raise PolicyViolation([f"{relative}:1: feature source too large"])
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise PolicyViolation([f"{relative}:1: unreadable feature source"]) from error
            validate_feature_source(relative, source)
