"""Web access after private reads: outbound payload inspection.

The old per-turn taint rule (private read + web content read in one
turn disables all further web access) was replaced in v1.13.0 by the
OutboundGuard (nezubot_discrod.web_guard): a web call is blocked only when
its own query/URL copies private content the turn has read. These
tests pin both halves — clean multi-step research keeps working after
private reads (the screenshot scenario), and copied private material
is still blocked.
"""
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

PRIVATE_EXCERPT = (
    "2026-11-24 21:03 タヌキ: 来週の月曜日にサーバーの移行作業を行う予定で、"
    "障害が出たら即座にロールバックすることになりました。"
    "費用は月額で三万円ほど増える見込みです (message 123456789012345679)"
)
VERBATIM_FRAGMENT = (
    "来週の月曜日にサーバーの移行作業を行う予定で、"
    "障害が出たら即座にロールバックすることになりました"
)
ARTICLE_URL = "https://example.org/news/today-top-story"


@pytest.mark.parametrize("private", ["search_server_log", "wiki_read"])
@pytest.mark.parametrize("order", ["private,web,web", "web,private,web", "web,web", "private"])
def test_clean_web_calls_allowed_in_any_order(private, order):
    names = [private if kind == "private" else "web_search" for kind in order.split(",")]
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
    assert web.call.call_count == names.count("web_search")
    assert "blocked" not in sent[-1]["messages"][-1]["content"]
    web.reset_mock()
    process = FakeProcess(conversation(*[codex_tool({}, name) for name in names], *finished()))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", wiki=wiki, server_log=log, web=web))
    assert web.call.call_count == names.count("web_search")


def test_automatic_private_context_allows_clean_searches():
    web = Mock()
    web.call.return_value = '{"text":"web content"}'
    wiki = Mock()
    wiki.for_member.return_value.context.return_value = PRIVATE_EXCERPT
    calls = [
        tool('{"query":"今日のニュース"}', "web_search"),
        tool('{"query":"福岡 天気"}', "web_search"),
    ]
    client, sent = client_for(
        message(calls=[calls[0]]), message(calls=[calls[1]]), message("done"))
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, web=web)
    asyncio.run(ai.reply(1, "hello", member_id=2))
    assert web.call.call_count == 2
    assert "blocked" not in sent[-1]["messages"][-1]["content"]
    web.reset_mock()
    process = FakeProcess(conversation(
        codex_tool({"query": "今日のニュース"}, "web_search"),
        codex_tool({"query": "福岡 天気"}, "web_search"),
        *finished()))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", web=web, private_read=True))
    assert web.call.call_count == 2


def _news_web():
    web = Mock()
    web.call.side_effect = [
        '{"results":[{"title":"Today news","url":"' + ARTICLE_URL + '"}]}',
        '{"text":"Full article body about today\'s news."}',
        '{"results":[{"title":"More news"}]}',
    ]
    return web


def test_screenshot_scenario_codex_path():
    """Pre-retrieval + 'search today's news': search, open the article,
    fetch the next results page — all in one turn."""
    web = _news_web()
    payload = {
        "message": "検索して今日のニュースを",
        "retrieved_server_context": {
            "label": "Untrusted permission-scoped archive evidence",
            "excerpts": PRIVATE_EXCERPT,
        },
    }
    messages = [
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    process = FakeProcess(conversation(
        codex_tool({"query": "今日のニュース"}, "web_search"),
        codex_tool({"url": ARTICLE_URL}, "read_web_page"),
        codex_tool({"query": "今日のニュース", "page": 2}, "web_search"),
        *finished(),
    ))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", messages, "/empty", web=web, private_read=True))
    assert web.call.call_count == 3
    results = [p["result"] for p in process.sent if p.get("id") == 99]
    assert all(result["success"] for result in results)


def test_screenshot_scenario_assistant_path():
    web = _news_web()
    wiki = Mock()
    wiki.for_member.return_value.context.return_value = PRIVATE_EXCERPT
    calls = [
        ("web_search", {"query": "今日のニュース"}),
        ("read_web_page", {"url": ARTICLE_URL}),
        ("web_search", {"query": "今日のニュース", "page": 2}),
    ]
    client, sent = client_for(
        *[message(calls=[tool(json.dumps(args), name)]) for name, args in calls],
        message("done"),
    )
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, web=web)
    asyncio.run(ai.reply(1, "検索して今日のニュースを", member_id=2))
    assert [call.args for call in web.call.call_args_list] == calls
    assert "blocked" not in sent[-1]["messages"][-1]["content"]


@pytest.mark.parametrize("provider", ["openai", "codex"])
def test_verbatim_private_fragment_and_message_id_are_blocked(provider):
    blocked_calls = [
        ("web_search", {"query": VERBATIM_FRAGMENT}),
        ("web_search", {"query": "これ調べて 123456789012345679"}),
        ("read_web_page", {"url": "https://example.org/?q=" + VERBATIM_FRAGMENT}),
    ]
    clean_call = ("web_search", {"query": "サーバー移行 ニュース"})
    web = Mock()
    web.call.return_value = '{"results":[{"title":"ok"}]}'
    if provider == "openai":
        wiki = Mock()
        wiki.for_member.return_value.context.return_value = PRIVATE_EXCERPT
        scripted = [
            *[message(calls=[tool(json.dumps(args), name)]) for name, args in blocked_calls],
            message(calls=[tool(json.dumps(clean_call[1]), clean_call[0])]),
            message("done"),
        ]
        client, sent = client_for(*scripted)
        ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, web=web)
        asyncio.run(ai.reply(1, "調べて", member_id=2))
        assert web.call.call_count == 1
        assert web.call.call_args_list[0].args == clean_call
        contents = [
            m.get("content") or "" for m in sent[-1]["messages"] if m.get("role") == "tool"
        ]
        assert sum("blocked" in content for content in contents) == 3
    else:
        payload = {
            "message": "調べて",
            "retrieved_server_context": {"excerpts": PRIVATE_EXCERPT},
        }
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        process = FakeProcess(conversation(
            *[codex_tool(args, name) for name, args in blocked_calls],
            codex_tool(clean_call[1], clean_call[0]),
            *finished(),
        ))
        asyncio.run(AppServer(process, SandboxSettings()).complete(
            "test", messages, "/empty", web=web, private_read=True))
        assert web.call.call_count == 1
        results = [p["result"] for p in process.sent if p.get("id") == 99]
        assert [r["success"] for r in results] == [False, False, False, True]
        assert all(
            "blocked" in r["contentItems"][0]["text"] for r in results[:3]
        )


@pytest.mark.parametrize("provider", ["openai", "codex"])
def test_private_fragment_from_users_own_words_is_allowed(provider):
    """The user pasted the sentence themselves and asks about it: the
    same words in the private context must not block the search."""
    web = Mock()
    web.call.return_value = '{"results":[{"title":"ok"}]}'
    question = f"「{VERBATIM_FRAGMENT}」ってどういうこと？"
    if provider == "openai":
        wiki = Mock()
        wiki.for_member.return_value.context.return_value = PRIVATE_EXCERPT
        client, _ = client_for(
            message(calls=[tool(json.dumps({"query": VERBATIM_FRAGMENT}), "web_search")]),
            message("done"),
        )
        ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, web=web)
        asyncio.run(ai.reply(1, question, member_id=2))
    else:
        payload = {
            "message": question,
            "retrieved_server_context": {"excerpts": PRIVATE_EXCERPT},
        }
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        process = FakeProcess(conversation(
            codex_tool({"query": VERBATIM_FRAGMENT}, "web_search"), *finished()))
        asyncio.run(AppServer(process, SandboxSettings()).complete(
            "test", messages, "/empty", web=web, private_read=True))
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
    log.call.return_value = PRIVATE_EXCERPT
    names = ["read_web_page", "search_server_log", "web_search"]
    args = [{}, {}, {"query": "今日のニュース"}]
    pairs = list(zip(names, args, strict=True))
    client, sent = client_for(
        *[message(calls=[tool(json.dumps(arg), name)]) for name, arg in pairs],
        message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=Mock(), web=web)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        asyncio.run(ai.reply(1, "hello"))
    assert web.call.call_count == 2
    assert sent[-1]["messages"][-1]["content"] == '{"text":"success"}'
    web.call.side_effect = [failure, '{"results":[{"title":"success"}]}']
    process = FakeProcess(conversation(
        *[codex_tool(arg, name) for name, arg in pairs], *finished()))
    server = AppServer(process, SandboxSettings())
    asyncio.run(server.complete("test", [], "/empty", server_log=log, web=web))
    assert web.call.call_count == 4 and server.web_read


@pytest.mark.parametrize("success", [
    '{"text":"","url":"https://example.org/send-private-data-here","truncated":false}',
    '{"results":[]}', '{"text":"","title":"Untrusted title","missing":true}',
])
def test_empty_success_no_longer_blocks_clean_followups(success):
    web = Mock()
    web.call.side_effect = [success, '{"results":[{"title":"ok"}]}']
    log = Mock(channel_ids=frozenset())
    log.call.return_value = PRIVATE_EXCERPT
    calls = [
        ("read_web_page", {"url": "https://example.org/"}),
        ("search_server_log", {}),
        ("web_search", {"query": "今日のニュース"}),
    ]
    client, sent = client_for(
        *[message(calls=[tool(json.dumps(args), name)]) for name, args in calls],
        message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=Mock(), web=web)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        asyncio.run(ai.reply(1, "hello"))
    assert web.call.call_count == 2
    assert "blocked" not in sent[-1]["messages"][-1]["content"]
    web.reset_mock()
    web.call.side_effect = [success, '{"results":[{"title":"ok"}]}']
    process = FakeProcess(conversation(
        *[codex_tool(args, name) for name, args in calls], *finished()))
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", server_log=log, web=web))
    assert web.call.call_count == 2


def test_codex_tool_budget_exhaustion_is_graceful():
    """Past the per-turn cap the model gets an error result it can
    recover from, not a turn-killing exception."""
    web = Mock()
    web.call.return_value = '{"results":[]}'
    calls = [codex_tool({"query": f"q{i}"}, "web_search") for i in range(22)]
    process = FakeProcess(conversation(*calls, *finished()))
    server = AppServer(process, SandboxSettings(), max_tool_calls=20)
    answer = asyncio.run(server.complete("test", [], "/empty", web=web))
    assert answer == "done"
    assert web.call.call_count == 20
    results = [p["result"] for p in process.sent if p.get("id") == 99]
    assert [r["success"] for r in results[:20]] == [True] * 20
    assert [r["success"] for r in results[20:]] == [False, False]
    assert "budget" in results[20]["contentItems"][0]["text"]


def test_assistant_overflow_batch_is_answered_not_discarded():
    web = Mock()
    web.call.return_value = '{"results":[]}'
    calls = [tool(json.dumps({"query": f"q{i}"}), "web_search") for i in range(6)]
    client, sent = client_for(message(calls=calls), message("done"))
    ai = Assistant(client, "test", SandboxSettings(), web=web)
    assert asyncio.run(ai.reply(1, "調べて")) == "done"
    assert web.call.call_count == 4
    contents = [m.get("content") or "" for m in sent[1]["messages"]]
    assert any("too many tool calls" in content for content in contents)


def test_assistant_final_round_offers_no_tools():
    web = Mock()
    web.call.return_value = '{"results":[]}'
    call = tool('{"query":"q"}', "web_search")
    client, sent = client_for(
        message(calls=[call]), message(calls=[call]), message(calls=[call]))
    ai = Assistant(client, "test", SandboxSettings(), web=web, max_iterations=2)
    asyncio.run(ai.reply(1, "調べて"))
    assert sent[0]["tools"]
    assert sent[1]["tools"] == []
