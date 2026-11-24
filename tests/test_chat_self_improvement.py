import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from test_assistant import client_for, message, tool
from test_bot import settings
from test_codex import FakeProcess, conversation, finished
from test_codex import tool as codex_tool

from nezubot_discrod.assistant import Assistant, request_improvement
from nezubot_discrod.bot import Bot
from nezubot_discrod.codex import AppServer
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.improvement_ui import ApprovalView, improvement_callback, run_improvement
from nezubot_discrod.policy import PolicyViolation, protected_path


def incoming(*, owner=True, dm=False, ambient=False):
    user = SimpleNamespace(id=99)
    return SimpleNamespace(
        id=1000,
        content="<@99> RAG を高速化して",
        author=SimpleNamespace(id=456 if owner else 12, bot=False),
        guild=None if dm else SimpleNamespace(id=123, me=user),
        mentions=[] if ambient else [user],
        reference=None,
        channel=SimpleNamespace(
            id=1, send=AsyncMock(), typing=Mock(side_effect=asyncio.Lock),
            permissions_for=Mock(return_value=SimpleNamespace(
                view_channel=True, read_message_history=True, send_messages=True,
            )),
        ),
    )


def service_for():
    return SimpleNamespace(
        settings=settings(), lock=asyncio.Lock(), pending=None,
        propose=AsyncMock(return_value=SimpleNamespace(
            url="https://github.com/owner/repo/pull/7", stat="server_log.py | 2 +", number=7,
        )),
    )


def test_owner_mention_starts_same_flow_and_posts_approval():
    async def scenario():
        msg = incoming()
        service = service_for()
        client, sent = client_for(
            message(calls=[tool('{"request":"Optimize RAG"}', "request_self_improvement")]),
            message("開始しました"),
        )
        bot = SimpleNamespace(
            settings=settings(), user=msg.mentions[0], busy=set(), slots=asyncio.Semaphore(4),
            assistant=Assistant(client, "test", SandboxSettings()),
            improvements=service, improvement_tasks=set(),
        )
        with patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[])):
            await Bot.on_message(bot, msg)
        assert bot.improvement_tasks
        await asyncio.gather(*bot.improvement_tasks)
        service.propose.assert_awaited_once()
        user_id, request = service.propose.call_args.args
        assert user_id == settings().owner_id
        assert request == "RAG を高速化して"
        assert "request_self_improvement" in {t["function"]["name"] for t in sent[0]["tools"]}
        posts = msg.channel.send.call_args_list
        proposal = next(call for call in posts if "レビュー待ち" in call.args[0])
        assert "pull/7" in proposal.args[0] and "server_log.py" in proposal.args[0]
        assert isinstance(proposal.kwargs["view"], ApprovalView)
    asyncio.run(scenario())


def test_bot_does_not_expose_callback_to_nonowner():
    async def scenario():
        msg = incoming(owner=False)
        bot = SimpleNamespace(
            settings=settings(), user=SimpleNamespace(id=99), busy=set(),
            slots=asyncio.Semaphore(4),
            assistant=SimpleNamespace(reply=AsyncMock(return_value="ok")),
            server_log=Mock(), archive=AsyncMock(),
        )
        with patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[1])):
            await Bot.on_message(bot, msg)
        assert "self_improve" not in bot.assistant.reply.call_args.kwargs
    asyncio.run(scenario())


@pytest.mark.parametrize("state", [
    "lock", "pending", "task", "shutdown", "dm", "nonowner", "other_guild",
])
def test_chat_refuses_busy_shutdown_dms_and_nonowners(state):
    async def scenario():
        service = service_for()
        tasks = {object()} if state == "task" else set()
        msg = incoming(owner=state != "nonowner", dm=state == "dm")
        if state == "other_guild":
            msg.guild.id = 999
        if state == "lock":
            await service.lock.acquire()
        if state == "pending":
            service.pending = object()
        with patch(
            "nezubot_discrod.improvement_ui.accepting_work", return_value=state != "shutdown"
        ):
            result = await improvement_callback(service, msg, tasks, 99)("change")
        assert "Started" not in result
        if state == "dm":
            assert "DM" in result and "指定サーバー" in result
        service.propose.assert_not_awaited()
    asyncio.run(scenario())


def test_callback_rechecks_author_at_execution_time():
    async def scenario():
        service, msg, tasks = service_for(), incoming(), set()
        callback = improvement_callback(service, msg, tasks, 99)
        msg.author.id = 12
        assert "Only the owner" in await callback("change")
        assert not tasks
        service.propose.assert_not_awaited()
    asyncio.run(scenario())


def test_nonowner_forged_model_tool_call_never_starts_propose():
    async def scenario():
        msg, service = incoming(owner=False), service_for()
        client, _ = client_for(
            message(calls=[tool('{"request":"change"}', "request_self_improvement")]),
            message("Only the owner can request changes"),
        )
        bot = SimpleNamespace(
            settings=settings(), user=msg.mentions[0], busy=set(), slots=asyncio.Semaphore(4),
            assistant=Assistant(client, "test", SandboxSettings()),
            improvements=service, improvement_tasks=set(),
        )
        with patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[])):
            await Bot.on_message(bot, msg)
        service.propose.assert_not_awaited()
        assert not bot.improvement_tasks
    asyncio.run(scenario())


def test_assistant_passes_owner_callback_to_codex_completion():
    callback = AsyncMock()
    complete = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=message("ok"))],
    ))
    assistant = Assistant(Mock(), "test", SandboxSettings(), wiki_completion=complete)
    asyncio.run(assistant.reply(1, "change", self_improve=callback))
    assert complete.call_args.kwargs["self_improve"] is callback


def test_duplicate_chat_requests_are_reserved_before_background_task_runs():
    async def scenario():
        service, msg, tasks = service_for(), incoming(), set()
        callback = improvement_callback(service, msg, tasks, 99)
        assert "Started" in await callback("change")
        assert "Started" not in await callback("another change")
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        assert not tasks
        service.propose.assert_awaited_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "codex"])
@pytest.mark.parametrize("mention", ["<@99>", "<@!99>"])
def test_model_supplied_private_context_never_reaches_propose(provider, mention):
    async def scenario():
        service, msg, tasks = service_for(), incoming(), set()
        msg.content = f"{mention} RAG を高速化して <@123>"
        callback = improvement_callback(service, msg, tasks, 99)
        secret = "PRIVATE_WIKI_LOG_WEB_SECRET"
        if provider == "openai":
            client, _ = client_for(
                message(calls=[tool(
                    '{"request":"' + secret + '"}', "request_self_improvement",
                )]), message("started"),
            )
            await Assistant(client, "test", SandboxSettings()).reply(
                1, msg.content, self_improve=callback,
            )
        else:
            process = FakeProcess(conversation(
                codex_tool({"request": secret}, "request_self_improvement"), *finished(),
            ))
            await AppServer(process, SandboxSettings()).complete(
                "test", [], "/empty", self_improve=callback,
            )
        await asyncio.gather(*tasks)
        service.propose.assert_awaited_once_with(456, "RAG を高速化して <@123>")
        assert secret not in service.propose.call_args.args[1]
    asyncio.run(scenario())


@pytest.mark.parametrize("content", ["<@99>  ", "<@99> " + "x" * 8001])
def test_original_owner_message_must_be_nonempty_and_bounded(content):
    async def scenario():
        service, msg, tasks = service_for(), incoming(), set()
        msg.content = content
        result = await improvement_callback(service, msg, tasks, 99)("model-written replacement")
        assert "Started" not in result
        assert not tasks
        service.propose.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize("error", [
    PolicyViolation(["nezubot_discrod/sandbox.py"]), RuntimeError("failed"),
])
def test_background_flow_reports_policy_and_proposal_errors(error):
    async def scenario():
        service, msg = service_for(), incoming()
        service.propose.side_effect = error
        await run_improvement(service, "change", msg.channel, 456)
        assert msg.channel.send.await_count
        assert all("view" not in call.kwargs for call in msg.channel.send.call_args_list)
    asyncio.run(scenario())


@pytest.mark.parametrize("args", [None, {}, {"request": ""}, {"request": " "},
                                  {"request": 1}, {"request": "x" * 8001},
                                  {"request": "change", "extra": True}])
def test_invalid_improvement_tool_arguments(args):
    callback = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(request_improvement(callback, args))
    callback.assert_not_awaited()


@pytest.mark.parametrize("ambient,exposed", [(False, True), (True, False)])
def test_codex_improvement_dynamic_tool_roundtrip(ambient, exposed):
    callback = AsyncMock(return_value="started")
    process = FakeProcess(conversation(
        codex_tool({"request": "change RAG"}, "request_self_improvement"), *finished(),
    ))
    server = AppServer(process, SandboxSettings())
    asyncio.run(server.complete("test", [], "/empty", self_improve=callback, ambient=ambient))
    start = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert ("request_self_improvement" in {t["name"] for t in start["dynamicTools"]}) == exposed
    if exposed:
        callback.assert_awaited_once_with("change RAG")
    else:
        callback.assert_not_awaited()


@pytest.mark.parametrize("ambient,provided", [(False, False), (True, True)])
def test_openai_does_not_publish_or_execute_unavailable_improvement_tool(ambient, provided):
    client, sent = client_for(
        message(calls=[tool('{"request":"change"}', "request_self_improvement")]), message("ok"),
    )
    callback = AsyncMock()
    assistant = Assistant(client, "test", SandboxSettings())
    asyncio.run(assistant.reply(1, "change", ambient=ambient,
                               self_improve=callback if provided else None))
    assert "request_self_improvement" not in {t["function"]["name"] for t in sent[0]["tools"]}
    callback.assert_not_awaited()


@pytest.mark.parametrize("path", [
    "nezubot_discrod/server_log.py", "nezubot_discrod/server_log_dense.py",
    "nezubot_discrod/server_log_discord.py", "nezubot_discrod/server_log_tools.py",
    "tests/test_server_log.py", "tests/test_server_log_integration.py",
    "tests/test_server_log_new.py",
])
def test_rag_files_are_editable(path):
    assert not protected_path(path)


@pytest.mark.parametrize("path", [
    "nezubot_discrod/sandbox.py", "nezubot_discrod/prompts.py", "nezubot_discrod/policy.py",
    "nezubot_discrod/server_log_evil.py", "nezubot_discrod/server_log/__init__.py",
    "nezubot_discrod/server_log.pyc", "nezubot_discrod/server_log_dense.so",
])
def test_rag_allowlist_keeps_protected_and_new_modules_locked(path):
    assert protected_path(path)
