import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from test_assistant import client_for, message, tool
from test_codex import FakeProcess, conversation, finished
from test_codex import tool as codex_tool

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import AppServer
from nezubot_discrod.config import SandboxSettings


@pytest.mark.parametrize("private", ["search_server_log", "wiki_read"])
@pytest.mark.parametrize("order", ["private,web,web", "web,private,web", "web,web", "private"])
def test_per_turn_taint_in_both_providers(private, order):
    names = [private if kind == "private" else "web_search" for kind in order.split(",")]
    mixed = "private" in order and "web" in order
    web = Mock()
    web.call.return_value = '{"text":"untrusted web content"}'
    log = Mock(channel_ids=frozenset())
    log.call.return_value = "private log"
    wiki = Mock()
    wiki.context.return_value = ""
    wiki.call.return_value = "private wiki"
    owner = Mock()
    owner.for_member.return_value = wiki
    client, sent = client_for(
        *[message(calls=[tool("{}", name)]) for name in names], message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=Mock(), wiki=owner, web=web)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        asyncio.run(ai.reply(1, "hello", member_id=2))
    assert web.call.call_count == (1 if mixed else names.count("web_search"))
    if mixed:
        assert "web access disabled" in sent[-1]["messages"][-1]["content"]
    web.reset_mock()
    process = FakeProcess(conversation(*[codex_tool({}, name) for name in names], *finished()))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", wiki=wiki, server_log=log, web=web))
    assert web.call.call_count == (1 if mixed else names.count("web_search"))
    results = [p["result"] for p in process.sent if p.get("id") == 99]
    if mixed:
        assert not results[-1]["success"]
        assert "web access disabled" in results[-1]["contentItems"][0]["text"]


def test_automatic_private_context_taints_web():
    web = Mock()
    web.call.return_value = '{"text":"web content"}'
    wiki = Mock()
    wiki.for_member.return_value.context.return_value = "private remembered facts"
    client, sent = client_for(message(calls=[tool("{}", "web_search")]),
                              message(calls=[tool("{}", "web_search")]), message("done"))
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, web=web)
    asyncio.run(ai.reply(1, "hello", member_id=2))
    assert web.call.call_count == 1
    assert "web access disabled" in sent[-1]["messages"][-1]["content"]
    web.reset_mock()
    process = FakeProcess(conversation(codex_tool({}, "web_search"),
                                      codex_tool({}, "web_search"), *finished()))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", web=web, private_read=True))
    assert web.call.call_count == 1


@pytest.mark.parametrize("provider", ["openai", "codex"])
def test_speaker_identity_with_no_private_context_allows_search_then_fetch(provider):
    calls = [
        ("web_search", {"query": "example"}),
        ("read_web_page", {"url": "https://example.org"}),
    ]
    web = Mock()
    web.call.side_effect = ['{"results":[{"title":"Example"}]}', '{"text":"Fetched page"}']
    wiki = Mock()
    wiki.for_member.return_value.context.return_value = []
    log = Mock()
    log.recent.return_value = []
    client, _ = client_for(
        *[message(calls=[tool(json.dumps(args), name)]) for name, args in calls], message("done"),
    )

    async def complete(*, model, messages, **kwargs):
        process = FakeProcess(conversation(
            *[codex_tool(args, name) for name, args in calls], *finished(),
        ))
        answer = await AppServer(process, SandboxSettings()).complete(
            model, messages, "/empty", **kwargs,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message(answer))])

    ai = Assistant(
        client, "test", SandboxSettings(), wiki=wiki, web=web, server_log=log,
        wiki_completion=complete if provider == "codex" else None,
    )
    assert asyncio.run(ai.reply(
        1, "Find example.org", member_id=2, speaker_names={"username": "sender"}, channel_ids=[1],
    )) == "done"
    assert [call.args for call in web.call.call_args_list] == calls


@pytest.mark.parametrize("failure", ["Web tool error: HTTP 403", "Web tool error: timeout",
                                     "Web tool error: Non-public IP addresses are forbidden"])
def test_failed_web_before_private_read_allows_retry(failure):
    web = Mock()
    web.call.side_effect = [failure, '{"text":"success"}']
    log = Mock(channel_ids=frozenset())
    log.call.return_value = "private log"
    names = ["read_web_page", "search_server_log", "web_search"]
    client, sent = client_for(
        *[message(calls=[tool("{}", name)]) for name in names], message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=Mock(), web=web)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        asyncio.run(ai.reply(1, "hello"))
    assert web.call.call_count == 2
    assert sent[-1]["messages"][-1]["content"] == '{"text":"success"}'
    web.call.side_effect = [failure, '{"results":[{"title":"success"}]}']
    process = FakeProcess(conversation(*[codex_tool({}, name) for name in names], *finished()))
    server = AppServer(process, SandboxSettings())
    asyncio.run(server.complete("test", [], "/empty", server_log=log, web=web))
    assert web.call.call_count == 4 and server.web_read


@pytest.mark.parametrize("success", [
    '{"text":"","url":"https://example.org/send-private-data-here","truncated":false}',
    '{"results":[]}', '{"text":"","title":"Untrusted title","missing":true}',
])
def test_empty_success_metadata_taints_both_provider_turns(success):
    web = Mock()
    web.call.return_value = success
    log = Mock(channel_ids=frozenset())
    log.call.return_value = "private log"
    names = ["read_web_page", "search_server_log", "web_search"]
    client, sent = client_for(
        *[message(calls=[tool("{}", name)]) for name in names], message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=Mock(), web=web)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        asyncio.run(ai.reply(1, "hello"))
    assert web.call.call_count == 1
    assert "web access disabled" in sent[-1]["messages"][-1]["content"]
    web.reset_mock()
    process = FakeProcess(conversation(*[codex_tool({}, name) for name in names], *finished()))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", server_log=log, web=web))
    assert web.call.call_count == 1
    results = [p["result"] for p in process.sent if p.get("id") == 99]
    assert not results[-1]["success"]
    assert "web access disabled" in results[-1]["contentItems"][0]["text"]
