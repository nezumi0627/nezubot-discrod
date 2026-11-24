import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import (
    CODEX_VERSION,
    PASSIVE_ITEM_TYPES,
    SAFE_CONFIG,
    AppServer,
    CodexClient,
    CodexSecurityError,
    app_server_arguments,
)
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.member_wiki import MemberWiki
from nezubot_discrod.prompts import (
    ARBITER_SYSTEM,
    CODEX_WIKI_TOOL_INSTRUCTIONS,
    SERVER_WIKI_BUILDER_SYSTEM,
    WIKI_SYSTEM,
)


def packet(method, params, request_id=None):
    result = {"method": method, "params": params}
    if request_id is not None:
        result["id"] = request_id
    return result


def response(request_id, result):
    return {"id": request_id, "result": result}


def tool(args=None, name="run_command", **kwargs):
    return packet(
        "item/tool/call",
        {
            "threadId": "thread1",
            "turnId": "turn1",
            "callId": "call1",
            "tool": name,
            "arguments": {"command": "echo hi"} if args is None else args,
            **kwargs,
        },
        99,
    )


def conversation(*events, account=None, policy=None):
    return [
        response(1, {}),
        response(2, {"account": {"type": "chatgpt"} if account is None else account}),
        response(
            3,
            {
                "thread": {"id": "thread1"},
                "approvalPolicy": "never",
                "sandbox": {"type": "readOnly"} if policy is None else policy,
            },
        ),
        response(4, {"turn": {"id": "turn1"}}),
        *events,
    ]


def finished(text="done", status="completed"):
    return [
        packet("item/completed", {"item": {"type": "agentMessage", "text": text}}),
        packet("turn/completed", {"turn": {"status": status}}),
    ]


class FakeProcess:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.returncode = None
        self.stdout = SimpleNamespace(readline=self.readline)
        self.stdin = SimpleNamespace(write=self.write, drain=AsyncMock())

    async def readline(self):
        return (json.dumps(self.events.pop(0)) + "\n").encode() if self.events else b""

    def write(self, value):
        self.sent.append(json.loads(value))

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


def test_tool_free_arbiter_single_turn():
    decision = '{"join":false,"ask":""}'
    process = FakeProcess(conversation(*finished(decision)))
    runner = Mock()
    server = AppServer(process, SandboxSettings(), runner=runner)
    assert asyncio.run(server.complete("gpt-6-luna", [], "/empty", arbiter=True)) == decision
    start = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert start["dynamicTools"] == []
    assert start["baseInstructions"] == ARBITER_SYSTEM
    assert start["sandbox"] == "read-only"
    turns = [p for p in process.sent if p.get("method") == "turn/start"]
    assert len(turns) == 1 and turns[0]["params"]["effort"] == "low"
    runner.assert_not_called()


def test_server_wiki_uses_a_fixed_tool_free_codex_turn():
    result = '{"people":[],"events":[]}'
    process = FakeProcess(conversation(*finished(result)))
    server = AppServer(process, SandboxSettings())
    assert asyncio.run(server.complete("gpt-6-luna", [], "/empty", server_wiki=True)) == result
    start = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert start["dynamicTools"] == []
    assert start["baseInstructions"] == SERVER_WIKI_BUILDER_SYSTEM
    assert start["sandbox"] == "read-only"
    assert next(p["params"]["effort"] for p in process.sent if p.get("method") == "turn/start") == (
        "low"
    )
    with pytest.raises(CodexSecurityError, match="fixed"):
        asyncio.run(server.complete(
            "gpt-6-luna", [{"role": "system", "content": "replacement"}], "/empty", server_wiki=True
        ))
    with pytest.raises(CodexSecurityError, match="cannot call tools"):
        process = FakeProcess(conversation(tool(), *finished()))
        asyncio.run(AppServer(process, SandboxSettings()).complete(
            "gpt-6-luna", [], "/empty", server_wiki=True
        ))


def test_long_formatted_server_wiki_json_is_not_truncated():
    entry = {"id": "123456789012345678", "summary": "要" * 240,
             "sources": ["123456789012345679"] * 3}
    result = json.dumps({"people": [entry] * 16, "events": []}, ensure_ascii=False, indent=16)
    assert len(result) > 12000
    server = AppServer(FakeProcess(conversation(*finished(result))), SandboxSettings())
    assert asyncio.run(server.complete("gpt-6-luna", [], "/empty", server_wiki=True)) == result

    server = AppServer(FakeProcess(conversation(*finished(result))), SandboxSettings())
    chat = asyncio.run(server.complete("gpt-6-luna", [], "/empty"))
    assert len(chat) == 8000 and chat.endswith("[output truncated]")


@pytest.mark.parametrize("event", [
    tool(),
    packet("item/started", {"item": {"type": "dynamicToolCall"}}),
    packet("item/started", {"item": {"type": "commandExecution"}}),
])
def test_arbiter_cannot_execute_tools(event):
    process = FakeProcess(conversation(event, *finished()))
    runner = Mock()
    with pytest.raises(CodexSecurityError):
        asyncio.run(AppServer(process, SandboxSettings(), runner=runner).complete(
            "gpt-6-luna", [], "/empty", arbiter=True))
    runner.assert_not_called()


def test_arbiter_timeout_and_fixed_prompt():
    client = CodexClient(Settings("", "", "", "gpt-6-luna", SandboxSettings(), provider="codex"))
    client.create = AsyncMock(return_value="decision")
    timeout = asyncio.timeout
    with patch("nezubot_discrod.codex.asyncio.timeout", wraps=timeout) as limit:
        assert asyncio.run(client.arbitrate(model="gpt-6-luna", messages=[])) == "decision"
    client.create.assert_awaited_once_with(model="gpt-6-luna", messages=[], arbiter=True)
    limit.assert_called_once_with(12)
    server = AppServer(FakeProcess([]), SandboxSettings())
    with pytest.raises(CodexSecurityError):
        asyncio.run(server.complete("test", [{"role": "system", "content": "injected"}],
                                    "/empty", arbiter=True))
    with pytest.raises(CodexSecurityError):
        asyncio.run(server.complete("test", [], "/empty", arbiter=True, web=Mock()))


def test_tool_round_trip_and_thread_policy():
    process = FakeProcess(conversation(tool(), *finished()))
    runner = Mock(return_value="Exit code: 0\nhi")
    sandbox = SandboxSettings()
    server = AppServer(process, sandbox, runner)
    assert asyncio.run(server.complete("gpt-6-luna", [], "/empty")) == "done"
    runner.assert_called_once_with("echo hi", sandbox)
    reply = next(p for p in process.sent if p.get("id") == 99)
    assert reply["result"] == {
        "success": True,
        "contentItems": [
            {"type": "inputText", "text": "Exit code: 0\nhi"},
        ],
    }
    thread = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert thread["sandbox"] == "read-only" and thread["approvalPolicy"] == "never"
    assert thread["ephemeral"] and thread["environments"] == []
    assert thread["runtimeWorkspaceRoots"] == thread["selectedCapabilityRoots"] == []
    assert [t["name"] for t in thread["dynamicTools"]] == ["run_command"]
    assert thread["dynamicTools"][0]["deferLoading"] is False
    assert process.sent[0]["params"]["capabilities"]["experimentalApi"]


def test_wiki_dynamic_tools_enforce_thread_and_speaker_scope(tmp_path):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    process = FakeProcess(conversation(
        tool({"page": "profile", "text": "Wrong thread"}, "wiki_append", threadId="other"),
        tool({"page": "profile", "text": "Wrong member", "member_id": 2}, "wiki_append"),
        tool({"page": "profile", "text": "Rust"}, "wiki_append"),
        tool({"page": "preferences", "key": "editor", "text": "Uses Vim"}, "wiki_remember"),
        tool({"page": "profile"}, "wiki_read"),
        *finished(),
    ))
    runner = Mock()
    server = AppServer(process, SandboxSettings(), runner)
    assert asyncio.run(server.complete("test", [], "/empty", wiki=pages)) == "done"
    thread = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert {t["name"] for t in thread["dynamicTools"]} == {
        "run_command", "wiki_list", "wiki_read", "wiki_search", "wiki_append", "wiki_rewrite",
        "wiki_remember",
    }
    assert thread["baseInstructions"] == WIKI_SYSTEM
    assert thread["developerInstructions"] == CODEX_WIKI_TOOL_INSTRUCTIONS
    replies = [p["result"] for p in process.sent if p.get("id") == 99]
    assert [r["success"] for r in replies] == [False, False, True, True, True]
    assert json.loads(replies[-2]["contentItems"][0]["text"])["changed"] is True
    assert json.loads(replies[-1]["contentItems"][0]["text"])["body"] == "Rust"
    runner.assert_not_called()


@pytest.mark.parametrize("kind", sorted(PASSIVE_ITEM_TYPES))
def test_passive_items_do_not_abort_turn(kind):
    process = FakeProcess(
        conversation(
            packet("item/started", {"item": {"type": kind}}),
            packet("item/completed", {"item": {"type": kind}}),
            *finished(),
        )
    )
    assert (
        asyncio.run(AppServer(process, SandboxSettings()).complete("test", [], "/empty")) == "done"
    )


@pytest.mark.parametrize(
    "method,decision",
    [
        ("execCommandApproval", "abort"),
        ("applyPatchApproval", "abort"),
        ("item/commandExecution/requestApproval", "cancel"),
        ("item/fileChange/requestApproval", "cancel"),
        ("item/permissions/requestApproval", None),
        ("command/exec", None),
        ("mcpServer/elicitation/request", None),
        ("unknown/future/tool", None),
    ],
)
def test_host_requests_are_denied(method, decision):
    process = FakeProcess(conversation(packet(method, {"command": "touch /host"}, 98)))
    runner = Mock()
    with pytest.raises(CodexSecurityError):
        asyncio.run(AppServer(process, SandboxSettings(), runner).complete("test", [], "/empty"))
    runner.assert_not_called()
    denied = process.sent[-1]
    assert denied["id"] == 98
    if decision:
        assert denied["result"]["decision"] == decision
    else:
        assert "error" in denied


@pytest.mark.parametrize(
    "event",
    [
        tool({}),
        tool({"command": ""}),
        tool(name="exec_command"),
        tool({"page": "profile", "text": "Rust"}, name="wiki_append"),
        tool(namespace="host"),
        tool(threadId="other"),
    ],
)
def test_invalid_dynamic_tools(event):
    process = FakeProcess(conversation(event, *finished()))
    runner = Mock()
    asyncio.run(AppServer(process, SandboxSettings(), runner).complete("test", [], "/empty"))
    runner.assert_not_called()
    assert not next(p for p in process.sent if p.get("id") == 99)["result"]["success"]


def test_tool_limit():
    process = FakeProcess(conversation(*[tool() for _ in range(21)]))
    runner = Mock(return_value="ok")
    with pytest.raises(RuntimeError, match="limit"):
        asyncio.run(AppServer(process, SandboxSettings(), runner).complete("test", [], "/empty"))
    assert runner.call_count == 20


@pytest.mark.parametrize("kind", ["commandExecution", "fileChange", "mcpToolCall", "webSearch"])
def test_unexpected_builtin_notification(kind):
    process = FakeProcess(conversation(packet("item/started", {"item": {"type": kind}})))
    with pytest.raises(CodexSecurityError):
        asyncio.run(AppServer(process, SandboxSettings()).complete("test", [], "/empty"))


def test_disabled_builtins_config():
    args = app_server_arguments("/bin/codex")
    assert args[:5] == ["/bin/codex", "app-server", "--listen", "stdio://", "--strict-config"]
    for feature in [
        "shell_tool",
        "unified_exec",
        "code_mode",
        "apps",
        "plugins",
        "hooks",
        "browser_use",
        "computer_use",
        "view_image",
        "image_generation",
        "multi_agent",
        "skill_search",
    ]:
        assert f"features.{feature}=false" in args
    assert SAFE_CONFIG["web_search"] == "disabled"
    assert SAFE_CONFIG["mcp_servers"] == {}
    assert SAFE_CONFIG["features.skip_host_skill_discovery"]
    assert not SAFE_CONFIG["skills.bundled.enabled"]


@pytest.mark.parametrize("refresh_succeeds", [True, False])
def test_server_wiki_warms_expiring_auth_in_an_isolated_home(
    tmp_path, monkeypatch, refresh_succeeds
):
    def auth(expires_in):
        expires = (datetime.now(UTC) + expires_in).timestamp()
        payload = base64.urlsafe_b64encode(json.dumps({"exp": expires}).encode()).decode()
        return json.dumps({"tokens": {"access_token": f"fake.{payload}.fake"}})

    original = tmp_path / "auth.json"
    expired, fresh = auth(timedelta(minutes=4)), auth(timedelta(hours=12))
    original.write_text(expired)
    settings = Settings("", "", "", "test", SandboxSettings(), codex_home=str(tmp_path))
    process = FakeProcess(conversation())
    version = SimpleNamespace(
        returncode=0, communicate=AsyncMock(return_value=((CODEX_VERSION + "\n").encode(), None))
    )

    async def spawn(*args, env, **kwargs):
        if "--version" in args:
            return version
        target = Path(env["CODEX_HOME"]) / "auth.json"
        assert target != original and target.read_text() == expired
        read = process.readline

        async def readline():
            if refresh_succeeds and process.sent[-1].get("method") == "account/read":
                target.write_text(fresh)
            return await read()

        process.stdout.readline = readline
        return process

    monkeypatch.setattr("nezubot_discrod.codex.asyncio.create_subprocess_exec", AsyncMock(
        side_effect=spawn
    ))

    async def scenario():
        leader, worker = CodexClient(settings), CodexClient(settings)
        try:
            if refresh_succeeds:
                snapshot = await leader.warm_server_wiki_auth()
                assert snapshot.read_text() == fresh
                seeded = worker.prepare_runtime(auth_file=snapshot) / "home" / "auth.json"
                assert seeded != snapshot and seeded.read_text() == fresh
                assert seeded.stat().st_mode & 0o777 == 0o600
            else:
                with pytest.raises(RuntimeError, match="did not produce fresh credentials"):
                    await leader.warm_server_wiki_auth()
        finally:
            await leader.close()
            await worker.close()

    asyncio.run(scenario())
    assert original.read_text() == expired
    assert process.sent[-1]["params"] == {"refreshToken": True}


def test_process_lifecycle_history_and_auth_isolation(tmp_path, monkeypatch):
    (tmp_path / "auth.json").write_text('{"test":"fake credentials"}')
    (tmp_path / "config.toml").write_text('notify = ["touch", "/host"]')
    monkeypatch.setenv("AI_API_KEY", "must-not-leak")
    settings = Settings(
        "token", "", "", "gpt-6-luna", SandboxSettings(), provider="codex", codex_home=str(tmp_path)
    )
    process = FakeProcess(conversation(tool(), *finished()))
    discovery = FakeProcess([
        response(1, {}), response(2, {"account": {"type": "chatgpt"}}),
        response(3, {"data": [], "nextCursor": None}),
    ])
    version = SimpleNamespace(
        returncode=0, communicate=AsyncMock(return_value=((CODEX_VERSION + "\n").encode(), None))
    )
    runner = Mock(return_value="hi")

    async def scenario():
        client = CodexClient(settings, runner=runner)
        assistant = Assistant(client, settings.model, settings.sandbox)
        async def spawn(*args, **kwargs):
            if "--version" in args:
                return version
            if any(arg.startswith("model_catalog_json=") for arg in args):
                catalog_arg = next(arg for arg in args if arg.startswith("model_catalog_json="))
                catalog = Path(json.loads(catalog_arg.split("=", 1)[1]))
                assert json.loads(catalog.read_text())["models"][0]["tool_mode"] == "direct"
                assert json.loads(catalog.read_text())["models"][0]["context_window"] == 1234
                return process
            home = Path(kwargs["env"]["CODEX_HOME"])
            (home / "models_cache.json").write_text(json.dumps({"models": [
                {"slug": settings.model, "tool_mode": "code_mode_only", "context_window": 1234}
            ]}))
            return discovery

        with patch(
            "nezubot_discrod.codex.asyncio.create_subprocess_exec", AsyncMock(side_effect=spawn)
        ) as start:
            assert await assistant.reply(1, "hello") == "done"
            assert assistant.history[1] == [("hello", "done")]
            assistant.reset(1)
            assert not assistant.history
            kwargs = start.call_args.kwargs
            assert not Path(kwargs["cwd"]).exists()
            home = Path(kwargs["env"]["CODEX_HOME"])
            assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
            assert not (home / "config.toml").exists()
            assert "AI_API_KEY" not in kwargs["env"]
            assert home != tmp_path
            runtime = Path(client.runtime.name)
            await client.close()
            assert not runtime.exists()
        assert process.returncode == -15
        assert discovery.returncode == -15
        assert [p.get("method") for p in discovery.sent if "method" in p] == [
            "initialize", "initialized", "account/read", "model/list"
        ]

    asyncio.run(scenario())
    runner.assert_called_once()


@pytest.mark.parametrize("models", [[], None, [{}], ["invalid"]])
def test_invalid_model_metadata_never_starts_a_turn(tmp_path, models):
    home = tmp_path / "home"
    home.mkdir()
    (home / "models_cache.json").write_text(json.dumps({"models": models}))
    process = FakeProcess([
        response(1, {}), response(2, {"account": {"type": "chatgpt"}}), response(3, {}),
    ])
    client = CodexClient(Settings("", "", "", "test", SandboxSettings(), provider="codex"))
    with patch(
        "nezubot_discrod.codex.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
    ), pytest.raises(CodexSecurityError, match="metadata"):
        asyncio.run(client.direct_catalog(tmp_path, tmp_path, {}, "test"))
    assert not (tmp_path / "direct-models.json").exists()
    assert not any(p.get("method") == "thread/start" for p in process.sent)


def test_direct_catalog_preserves_metadata_reuses_snapshot_and_rejects_unknown_model(tmp_path):
    catalog = tmp_path / "direct-models.json"
    catalog.write_text(json.dumps({"models": [{"slug": "known", "tool_mode": "direct"}]}))
    client = CodexClient(Settings("", "", "", "test", SandboxSettings(), provider="codex"))
    assert asyncio.run(client.direct_catalog(tmp_path, tmp_path, {}, "known")) == catalog
    with pytest.raises(RuntimeError, match="missing"):
        asyncio.run(client.direct_catalog(tmp_path, tmp_path, {}, "unknown"))


@pytest.mark.parametrize(
    "events,error",
    [
        (conversation(*finished("")), None),
        (conversation(*finished(status="failed")), RuntimeError),
        (conversation(account={"type": "apiKey"}), RuntimeError),
        (conversation(policy={"type": "dangerFullAccess"}), CodexSecurityError),
        (conversation(policy={"type": "readOnly", "networkAccess": True}), CodexSecurityError),
        ([], RuntimeError),
    ],
)
def test_empty_errors_and_security_policy(events, error):
    process = FakeProcess(events)
    operation = AppServer(process, SandboxSettings()).complete("test", [], "/empty")
    if error:
        with pytest.raises(error):
            asyncio.run(operation)
    else:
        assert asyncio.run(operation) == ""


def test_provider_config(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "test")
    monkeypatch.setenv("AI_PROVIDER", "codex")
    monkeypatch.setenv("DISCORD_GUILD_ID", "123")
    monkeypatch.setenv("DISCORD_OWNER_ID", "456")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)
    assert Settings.from_env().model == "gpt-6-luna"
    monkeypatch.setenv("AI_PROVIDER", "invalid")
    with pytest.raises(ValueError, match="AI_PROVIDER"):
        Settings.from_env()
