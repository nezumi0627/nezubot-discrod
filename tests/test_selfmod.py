import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.improvement_ui import ApprovalView, handle_improve
from nezubot_discrod.policy import PolicyViolation
from nezubot_discrod.sandbox import docker_arguments
from nezubot_discrod.selfmod import (
    CHECK_COMMAND,
    ImprovementService,
    github_repository,
    host_git,
    restart_bot,
)


def settings(tmp_path):
    return Settings(
        "token",
        "key",
        "https://example.com",
        "test",
        SandboxSettings(),
        owner_id=456,
        guild_id=123,
        github_token="fake-token",
        self_repo="owner/repo",
        self_checkout=str(tmp_path),
    )


def test_restart_cleans_sandboxes_before_exec():
    calls = Mock()
    with (
        patch("nezubot_discrod.selfmod.shutdown_sandboxes", calls.shutdown),
        patch("nezubot_discrod.selfmod.os.execv", calls.exec),
    ):
        restart_bot()
    assert [call[0] for call in calls.mock_calls] == ["shutdown", "exec"]


def test_restart_does_not_exec_if_sandbox_cleanup_fails():
    with (
        patch("nezubot_discrod.selfmod.shutdown_sandboxes", side_effect=RuntimeError("cleanup")),
        patch("nezubot_discrod.selfmod.os.execv") as execute,
        pytest.raises(RuntimeError, match="cleanup"),
    ):
        restart_bot()
    execute.assert_not_called()


def test_failed_restart_keeps_merged_proposal_for_retry(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path)
    asyncio.run(service.propose(456, "fix README"))
    restart.side_effect = RuntimeError("cleanup")
    with pytest.raises(RuntimeError, match="cleanup"):
        asyncio.run(service.approve(456, 7))
    assert service.pending.merged
    restart.side_effect = None
    asyncio.run(service.approve(456, 7))
    github.merge_pr.assert_called_once()
    assert service.pending is None


class FakeGit:
    def __init__(self):
        self.calls = []
        self.dirty = False

    def __call__(self, args, *, cwd, git_dir=None, token=""):
        self.calls.append((args, Path(cwd), git_dir))
        if args[0] == "clone":
            work = Path(args[-1])
            work.mkdir()
            (work / ".git").mkdir()
            (work / "README.md").write_text("original")
        if args[0] == "remote":
            return "https://github.com/owner/repo.git"
        if args[0] == "diff":
            if "--raw" in args:
                return ":100644 100644 " + "a" * 40 + " " + "0" * 40 + " M\0README.md\0"
            return "README.md\0" if "--name-only" in args else " README.md | 1 +"
        if args[0] == "rev-parse":
            return "abc123"
        if args[0] == "branch":
            return "main"
        if args[0] == "status":
            return " M local.py" if self.dirty else ""
        if args[0] == "ls-files":
            return "".join(str(path.relative_to(cwd)) + "\0" for path in Path(cwd).rglob("*")
                           if path.is_file() and path.name != "README.md")
        return ""


def service_for(tmp_path, output="Exit code: 0\npassed"):
    git = FakeGit()
    github = SimpleNamespace(
        default_branch=Mock(return_value="main"),
        create_pr=Mock(
            return_value={"number": 7, "html_url": "https://github.com/owner/repo/pull/7"}
        ),
        merge_pr=Mock(),
        close_pr=Mock(),
        delete_branch=Mock(),
    )

    async def editor(request, workspace, sandbox):
        assert not (workspace / ".git").exists()
        assert (workspace.parent / "git-metadata").is_dir()
        (workspace / "README.md").write_text("improved")
        return "Updated README"

    runner = Mock(return_value=output)
    restart = Mock()
    service = ImprovementService(
        settings(tmp_path), github=github, git=git, editor=editor, runner=runner, restart=restart
    )
    return service, git, github, runner, restart


@pytest.mark.parametrize(
    "options,user", [('[]', "1000:1000"), ('["name=rootless"]', "0:0")]
)
def test_workspace_mount_keeps_all_hardening(tmp_path, options, user):
    with (
        patch("nezubot_discrod.sandbox.os.getuid", return_value=1000),
        patch("nezubot_discrod.sandbox.os.getgid", return_value=1000),
        patch("nezubot_discrod.sandbox.subprocess.run", return_value=Mock(stdout=options)) as info,
    ):
        args = docker_arguments(SandboxSettings(), "test", "id", tmp_path)
    assert args[args.index("--mount") + 1] == f"type=bind,src={tmp_path},dst=/work"
    for flag, expected in {
        "--network": "none",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--memory": "128m",
        "--cpus": "0.5",
        "--pids-limit": "64",
    }.items():
        assert args[args.index(flag) + 1] == expected
    assert "--read-only" in args and "--rm" in args
    assert args[args.index("--user") + 1] == user
    info.assert_called_once_with(
        ["docker", "info", "--format", "{{json .SecurityOptions}}"],
        capture_output=True, text=True, timeout=10, check=True,
    )


def test_failed_checks_never_push_or_open_pr(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path, "Exit code: 1\nfailed")
    with pytest.raises(RuntimeError, match="PR は作成しません"):
        asyncio.run(service.propose(456, "fix README"))
    github.create_pr.assert_not_called()
    assert not any(call[0][0] in {"commit", "push"} for call in git.calls)
    assert service.pending is None
    restart.assert_not_called()
    assert runner.call_args.args[0] == CHECK_COMMAND


@pytest.mark.parametrize("path", ["nezubot_discrod/sandbox.py", "pyproject.toml", "conftest.py"])
def test_protected_edit_is_rejected_before_checks_or_pr(tmp_path, path):
    service, git, github, runner, restart = service_for(tmp_path)

    async def editor(request, workspace, sandbox):
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("untrusted change")
        return "done"

    service.editor = editor

    def changed_git(args, **kwargs):
        if args[0] == "diff" and "--raw" in args:
            return ":100644 100644 " + "a" * 40 + " " + "0" * 40 + " M\0" + path + "\0"
        return git(args, **kwargs)

    service.git = changed_git
    with pytest.raises(PolicyViolation) as error:
        asyncio.run(service.propose(456, "change security"))
    assert path in error.value.paths
    runner.assert_not_called()
    github.create_pr.assert_not_called()
    assert not any(call[0][0] in {"commit", "push"} for call in git.calls)


def test_features_only_change_passes_policy_checks_and_opens_pr(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path)
    path = "nezubot_discrod/features/hello.py"

    async def editor(request, workspace, sandbox):
        feature = workspace / path
        feature.parent.mkdir(parents=True)
        feature.write_text(
            'async def hello(interaction):\n    await interaction.response.send_message("hello")\n'
            'def setup(registry):\n'
            '    registry.register(name="hello", description="hello", callback=hello)\n'
        )
        (workspace / "tests").mkdir()
        (workspace / "tests/test_feature_hello.py").write_text("def test_hello(): assert True\n")
        return "Added hello"

    def feature_git(args, **kwargs):
        if args[0] == "diff" and "--raw" in args:
            return ""
        if args[0] == "diff" and "--name-only" in args:
            return path + "\0tests/test_feature_hello.py\0"
        return git(args, **kwargs)

    service.editor = editor
    service.git = feature_git
    proposal = asyncio.run(service.propose(456, "add hello"))
    assert proposal.number == 7
    runner.assert_called_once()
    github.create_pr.assert_called_once()


def test_checks_cannot_mutate_protected_files_then_open_pr(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path)

    def malicious_check(command, sandbox, *, workspace):
        (workspace / "conftest.py").write_text("untrusted")
        return "Exit code: 0\nfaked success"

    service.runner = malicious_check
    with pytest.raises(PolicyViolation):
        asyncio.run(service.propose(456, "update README"))
    github.create_pr.assert_not_called()


def test_policy_failure_posts_all_paths_without_truncating(tmp_path):
    async def scenario():
        service = SimpleNamespace(settings=settings(tmp_path), propose=AsyncMock(
            side_effect=PolicyViolation([f"protected-{i:03}-" + "x" * 50 for i in range(60)])
        ))
        call = interaction()
        await handle_improve(call, "request", service)
        messages = call.channel.send.call_args_list
        text = "".join(item.args[0] for item in messages)
        assert "protected-000-" in text and "protected-059-" in text
        assert len(messages) > 1
        assert all(item.kwargs["allowed_mentions"].everyone is False for item in messages)
    asyncio.run(scenario())


def test_propose_then_approve_merges_exact_head_pulls_and_restarts(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path)

    async def scenario():
        proposal = await service.propose(456, "fix README")
        assert proposal.branch.startswith("selfmod/")
        assert proposal.stat and proposal.sha == "abc123"
        assert not runner.call_args.kwargs["workspace"].exists()
        with pytest.raises(RuntimeError, match="承認待ち"):
            await service.propose(456, "another")
        restart.assert_not_called()
        await service.approve(456, 7)

    asyncio.run(scenario())
    github.create_pr.assert_called_once()
    github.merge_pr.assert_called_once_with("owner/repo", 7, "abc123")
    assert any(call[0] == ["pull", "--ff-only", "origin", "main"] for call in git.calls)
    restart.assert_called_once()
    assert service.pending is None


def test_reject_closes_pr_and_deletes_branch(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path)

    async def scenario():
        proposal = await service.propose(456, "fix README")
        await service.reject(456, proposal.number)
        return proposal

    proposal = asyncio.run(scenario())
    github.close_pr.assert_called_once_with("owner/repo", 7)
    github.delete_branch.assert_called_once_with("owner/repo", proposal.branch)
    github.merge_pr.assert_not_called()
    restart.assert_not_called()
    assert service.pending is None


@pytest.mark.parametrize("action", ["propose", "approve", "reject"])
def test_service_is_owner_only(tmp_path, action):
    service, git, github, runner, restart = service_for(tmp_path)
    with pytest.raises(PermissionError):
        asyncio.run(getattr(service, action)(1, "request" if action == "propose" else 7))
    assert not git.calls
    github.create_pr.assert_not_called()


def test_dirty_running_checkout_prevents_merge(tmp_path):
    service, git, github, runner, restart = service_for(tmp_path)

    async def scenario():
        await service.propose(456, "fix README")
        git.dirty = True
        with pytest.raises(RuntimeError, match="checkout"):
            await service.approve(456, 7)

    asyncio.run(scenario())
    github.merge_pr.assert_not_called()
    restart.assert_not_called()


def interaction(user_id=456):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id),
        guild_id=123,
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        channel=SimpleNamespace(send=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
        message=SimpleNamespace(edit=AsyncMock()),
    )


def test_nonowner_slash_and_buttons_are_refused(tmp_path):
    service, *_ = service_for(tmp_path)
    service.propose = AsyncMock()

    async def scenario():
        call = interaction(1)
        await handle_improve(call, "request", service)
        assert call.response.send_message.call_args.kwargs["ephemeral"]
        view = ApprovalView(service, SimpleNamespace(number=7))
        await view.decide(call, True)
        await view.decide(call, False)
        assert not call.response.defer.called

    asyncio.run(scenario())
    service.propose.assert_not_called()


def test_owner_command_posts_pr_and_buttons(tmp_path):
    service, *_ = service_for(tmp_path)

    async def scenario():
        call = interaction()
        await handle_improve(call, "fix README", service)
        content = call.channel.send.call_args.args[0]
        assert "https://github.com/owner/repo/pull/7" in content
        assert "README.md" in content
        assert isinstance(call.channel.send.call_args.kwargs["view"], ApprovalView)

    asyncio.run(scenario())


def test_host_git_disables_hooks_filters_and_does_not_use_shell(tmp_path):
    result = SimpleNamespace(returncode=0, stdout=b"name \0")
    with patch("nezubot_discrod.selfmod.subprocess.run", return_value=result) as run:
        output = host_git(["diff", "-z"], cwd=tmp_path, git_dir=tmp_path / "metadata", token="fake")
    assert output == "name \0"
    args = run.call_args.args[0]
    assert "core.hooksPath=/dev/null" in args and "core.fsmonitor=false" in args
    assert "--git-dir" in args and "--work-tree" in args
    assert "shell" not in run.call_args.kwargs
    env = run.call_args.kwargs["env"]
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "fake" not in " ".join(args)


@pytest.mark.parametrize(
    "source", ["owner/repo", "https://github.com/owner/repo.git", "git@github.com:owner/repo.git"]
)
def test_repository_sources(source):
    assert github_repository(source) == "owner/repo"


@pytest.mark.parametrize("source", ["https://evil.test/owner/repo", "../repo", "-x/a/b"])
def test_reject_untrusted_repository_sources(source):
    with pytest.raises(ValueError):
        github_repository(source)


@pytest.mark.parametrize("improvement_request", ["", " " * 10, "x" * 8001])
def test_invalid_improvement_request(tmp_path, improvement_request):
    service, git, *_ = service_for(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(service.propose(456, improvement_request))
    assert not git.calls


def test_selfmod_editor_routes_tools_to_mounted_sandbox(tmp_path):
    service, _, _, runner, _ = service_for(tmp_path)
    tool = SimpleNamespace(
        id="call",
        function=SimpleNamespace(name="run_command", arguments='{"command":"touch new.py"}'),
    )
    message = SimpleNamespace(
        tool_calls=[tool],
        model_dump=lambda **_: {
            "role": "assistant",
            "tool_calls": [],
        },
    )
    final = SimpleNamespace(tool_calls=None, content="done")
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    side_effect=[
                        SimpleNamespace(choices=[SimpleNamespace(message=message)]),
                        SimpleNamespace(choices=[SimpleNamespace(message=final)]),
                    ],
                )
            )
        ),
        close=AsyncMock(),
    )
    sandbox = replace(service.settings.sandbox, image=service.settings.selfmod_image)
    with patch("nezubot_discrod.selfmod.AsyncOpenAI", return_value=client):
        assert asyncio.run(service.edit("request", tmp_path, sandbox)) == "done"
    runner.assert_called_once_with("touch new.py", sandbox, workspace=tmp_path)
    client.close.assert_awaited_once()
    assert (
        "worktree"
        in client.chat.completions.create.call_args_list[0].kwargs["messages"][0]["content"]
    )
