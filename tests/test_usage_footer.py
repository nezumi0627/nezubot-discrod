import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from test_ambient import incoming
from test_assistant import message, tool
from test_bot import settings
from test_codex import FakeProcess, conversation, finished, packet
from test_codex import tool as codex_tool

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.bot import Bot
from nezubot_discrod.codex import CODEX_VERSION, AppServer, CodexClient
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.messages import (
    Reply,
    Source,
    clean_sources,
    collect_tool_sources,
    make_teaser,
    split_reply,
    token_usage,
    tool_sources,
)
from nezubot_discrod.server_log import LogHit, format_hits
from nezubot_discrod.server_log_tools import LogResult, LogTools

JUMP = "https://discord.com/channels/123/10/42"
WEB = "https://ja.wikipedia.org/wiki/Python"


def response(text=None, *, calls=None, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=message(text, calls))], usage=usage)


def client_for(*responses):
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(side_effect=responses)),
    ))


def test_openai_sums_all_tool_rounds_and_does_not_persist_footer():
    client = client_for(
        response(calls=[tool()], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10)),
        response(calls=[tool()], usage={"prompt_tokens": 200, "completion_tokens": 20}),
        response("answer\n-# copied model 999 tokens", usage={
            "prompt_tokens": 12045, "completion_tokens": 648,
        }),
    )
    ai = Assistant(client, "gpt-6-luna", SandboxSettings(), runner=Mock(return_value="ok"))
    reply = asyncio.run(ai.reply(1, "question"))
    assert reply == "answer"
    assert reply.usage == (12345, 678)
    assert ai.history[1] == [("question", "answer")]
    assert split_reply(reply, "gpt-6-luna") == [
        "answer\n-# gpt-6-luna · 入力 12,345 / 出力 678 tokens",
    ]


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": -1, "completion_tokens": 2}])
def test_missing_or_partial_usage_never_displays_misleading_totals(usage):
    client = client_for(
        response(calls=[tool()], usage={"prompt_tokens": 100, "completion_tokens": 10}),
        response("ok", usage=usage),
    )
    reply = asyncio.run(Assistant(
        client, "model", SandboxSettings(), runner=Mock(return_value="ok"),
    ).reply(1, "question"))
    assert reply.usage is None
    assert split_reply(reply, "model") == ["ok\n-# model"]


def usage_event(inputs, outputs, thread="thread1"):
    return packet("thread/tokenUsage/updated", {
        "threadId": thread, "turnId": "turn1",
        "tokenUsage": {"total": {"inputTokens": inputs, "outputTokens": outputs},
                       "last": {"inputTokens": 999, "outputTokens": 999}},
    })


@pytest.mark.parametrize("with_usage", [True, False])
@pytest.mark.parametrize("with_tool", [True, False])
def test_codex_cumulative_usage_and_sources_reach_assistant(tmp_path, with_usage, with_tool):
    async def scenario():
        events = [usage_event(100, 10)] if with_usage else []
        if with_tool:
            events += [codex_tool({"query": "past"}, "search_server_log")]
        if with_usage:
            events += [usage_event(300, 30), usage_event(300, 30),
                       usage_event(999999, 999999, "another-thread")]
        events += finished("answer\n-# fake footer")
        process = FakeProcess(conversation(*events))
        server = AppServer(process, SandboxSettings())
        log = Mock(channel_ids=("10",))
        log.call.return_value = LogResult.from_hits([
            LogHit("42", "10", "general", "Alice", "2026-01-01", "retrieved text", JUMP),
        ])

        @asynccontextmanager
        async def transport(*args):
            yield server

        client = CodexClient(settings())
        client.prepare_runtime = Mock(return_value=tmp_path)
        client.direct_catalog = AsyncMock(return_value=tmp_path / "catalog.json")
        client.server = transport
        version = SimpleNamespace(
            returncode=0,
            communicate=AsyncMock(return_value=((CODEX_VERSION + "\n").encode(), b"")),
        )
        ai = Assistant(client, "model", SandboxSettings(), wiki_completion=client.create,
                       server_log=Mock(recent=Mock(return_value=[LogHit(
                           "99", "10", "general", "member", "2026-01-01", "hello",
                           "https://discord.com/channels/123/10/99",
                       )])))
        with (patch("nezubot_discrod.codex.asyncio.create_subprocess_exec",
                    AsyncMock(return_value=version)),
              patch("nezubot_discrod.assistant.LogTools", return_value=log)):
            reply = await ai.reply(10, "question", channel_ids=[10])
        assert reply == "answer"
        assert reply.usage == ((300, 30) if with_usage else None)
        assert reply.sources == ((JUMP,) if with_tool else ())
        lines = split_reply(reply, "model")[-1].splitlines()
        assert lines[-2] == "answer"
        assert lines[-1] == ("-# model · 入力 300 / 出力 30 tokens" if with_usage else "-# model")
        assert all(JUMP not in part for part in split_reply(reply, "model"))
    asyncio.run(scenario())


@pytest.mark.parametrize("tool_result", [None, "No messages found.", "Tool error: unavailable"])
def test_recent_conversation_is_not_a_source_without_retrieved_tool_urls(tool_result):
    replies = [] if tool_result is None else [
        response(calls=[tool('{"query":"past"}', "search_server_log")]),
    ]
    client = client_for(*replies, response("hello"))
    store = Mock(recent=Mock(return_value=[LogHit(
        "42", "10", "general", "member", "2026-01-01", "hello", JUMP,
    )]))
    log = Mock(channel_ids=("10",))
    log.call.return_value = tool_result
    ai = Assistant(client, "model", SandboxSettings(), server_log=store)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        reply = asyncio.run(ai.reply(10, "hi", channel_ids=[10]))
    sent = client.chat.completions.create.call_args_list[0].kwargs["messages"]
    user_message = next(item for item in sent if item["role"] == "user")
    assert JUMP in json.loads(user_message["content"])["recent_messages"]
    assert reply.sources == ()
    assert split_reply(reply, "model") == ["hello\n-# model"]


def test_openai_rag_and_web_sources_are_code_owned_and_footer_is_last():
    client = client_for(
        response(calls=[tool('{"query":"past"}', "search_server_log")]),
        response(calls=[tool('{"title":"Python"}', "wikipedia_read")]),
        response("answer\n-# fabricated\n   -# <https://evil.example/>"),
    )
    log = Mock(channel_ids=("10",))
    log.call.return_value = f"1. text with an unrelated https://evil.example/ {JUMP}"
    web = Mock()
    web.call.return_value = json.dumps({"url": WEB, "title": "Python", "text": "page"})
    ai = Assistant(client, "model", SandboxSettings(), server_log=Mock(), web=web)
    with patch("nezubot_discrod.assistant.LogTools", return_value=log):
        reply = asyncio.run(ai.reply(1, "question"))
    assert reply == "answer"
    assert [str(source) for source in reply.sources] == [WEB, JUMP]
    assert split_reply(reply, "model") == ["answer\n-# model"]


@pytest.mark.parametrize("text", ["x" * 1999, "x" * 4001, "😀" * 2100])
def test_split_messages_put_footer_only_in_last_message(text):
    reply = Reply(text + "\n-# model copied", usage=(1, 2), sources=[JUMP])
    parts = split_reply(reply, "model")
    assert len(parts) > 1
    assert all(len(part.encode("utf-16-le")) // 2 <= 2000 for part in parts)
    assert all("-#" not in part for part in parts[:-1])
    assert parts[-1].endswith("-# model · 入力 1 / 出力 2 tokens")
    assert all(JUMP not in part for part in parts)
    assert "".join(parts).startswith(text)


def test_sources_are_deduplicated_bounded_and_only_structured_urls_are_used():
    assert clean_sources([JUMP, JUMP, WEB, "https://example.com/", "https://fourth.example/"]) == [
        JUMP, WEB, "https://example.com/", "https://fourth.example/",
    ]
    assert clean_sources(["javascript:alert(1)", "https://user:pass@example.com/",
                          "https://example.com/>@everyone", "https://example.com/\n"]) == []
    assert tool_sources("web_search", json.dumps({"results": [
        {"url": WEB, "snippet": "https://evil.example/"}, {"url": WEB},
    ]})) == [WEB]
    assert tool_sources("run_command", WEB) == []
    assert tool_sources("wiki_read", WEB) == []
    assert tool_sources("read_web_page", "Web tool error: https://example.com/") == []


@pytest.mark.parametrize("provider", ["openai", "codex"])
@pytest.mark.parametrize("read_name", ["read_web_page", "wikipedia_read", "read_server_log"])
def test_read_sources_take_priority_without_discarding_search_results(provider, read_name):
    is_log = read_name == "read_server_log"
    urls = [f"https://discord.com/channels/123/10/{i}" if is_log
            else f"https://example.com/{i}" for i in range(5)]
    search_name = "search_server_log" if is_log else "web_search"
    search_result = (LogResult.from_hits([
        LogHit(str(i), "10", "general", "Alice", "2026-01-01", "text", url)
        for i, url in enumerate(urls)
    ]) if is_log else json.dumps({"results": [
        {"url": url, "title": f"Page {i}"} for i, url in enumerate(urls)
    ]}))
    read_result = f"text {urls[3]}" if is_log else json.dumps({"url": urls[3], "text": "page"})
    backend = Mock(channel_ids=())
    backend.call.side_effect = [search_result, read_result, search_result]
    if provider == "openai":
        client = client_for(
            response(calls=[tool('{}', search_name)]),
            response(calls=[tool('{}', read_name)]),
            response(calls=[tool('{}', search_name)]),
            response("answer"),
        )
        ai = Assistant(client, "model", SandboxSettings(),
                       server_log=Mock() if is_log else None,
                       web=None if is_log else backend)
        with patch("nezubot_discrod.assistant.LogTools", return_value=backend):
            reply = asyncio.run(ai.reply(1, "question"))
    else:
        process = FakeProcess(conversation(
            codex_tool({}, search_name), codex_tool({}, read_name),
            codex_tool({}, search_name), *finished("answer"),
        ))
        server = AppServer(process, SandboxSettings())
        answer = asyncio.run(server.complete(
            "model", [], "/empty", server_log=backend if is_log else None,
            web=None if is_log else backend,
        ))
        reply = Reply(answer, sources=server.sources)
    assert reply.sources == (urls[3], urls[0], urls[1], urls[2], urls[4])
    assert reply.sources[0].title_text == ("#general · Alice · 2026-01-01" if is_log else "Page 3")
    assert split_reply(reply, "model") == ["answer\n-# model"]


def test_reply_rendering_never_shows_sources():
    reply = Reply("answer", sources=[Source(JUMP, title="#secret · Alice · 2026-01-01")])
    assert split_reply(reply, "model") == ["answer\n-# model"]


@pytest.mark.parametrize("name,args", [("search_server_log", {"query": "past"}),
                                      ("read_server_log", {"message_id": "42"})])
def test_log_metadata_comes_from_structured_hits_not_display_parsing(name, args):
    hit = LogHit("42", "10", "Project Planning", "Alice: Bob Smith", "2026-01-01T10:00:00",
                 "misleading: body", JUMP)
    store = Mock()
    store.search.return_value = store.read.return_value = [hit]
    result = LogTools(store, [10]).call(name, args)
    assert isinstance(result, str) and result == format_hits([hit])
    sources = tool_sources(name, result)
    assert sources[0].title_text == "#Project Planning · Alice: Bob Smith · 2026-01-01"
    unstructured = tool_sources(name, str(result))
    assert [str(source) for source in unstructured] == [JUMP]
    assert getattr(unstructured[0], "title_text", None) is None


def test_collection_tool_sources_preserve_structured_metadata_and_plain_urls():
    hit = LogHit("42", "10", "Project Planning", "Alice", "2026-01-01T10:00:00",
                 "body", JUMP)
    store = Mock()
    store.collect.return_value = {"message_count": 1, "hits": [hit]}
    result = LogTools(store, [10]).call("collect_server_log", {"author": "20"})
    sources = tool_sources("collect_server_log", result)
    assert sources == [JUMP]
    assert sources[0].title_text == "#Project Planning · Alice · 2026-01-01"
    assert tool_sources("collect_server_log", str(result)) == [JUMP]
    assert tool_sources("collect_server_log", "No results") == []


@pytest.mark.parametrize("title", [None, "", {}, 123])
def test_missing_or_invalid_source_title_is_dropped(title):
    sources = tool_sources("wikipedia_read", json.dumps({"url": WEB, "title": title}))
    assert [str(source) for source in sources] == [WEB]
    assert getattr(sources[0], "title_text", None) is None


def test_empty_or_failed_tool_results_do_not_add_sources():
    for result in ("Web tool error: failed", "not json", "[]", '{"results": []}'):
        assert collect_tool_sources([], "web_search", result) == []


@pytest.mark.parametrize("value", [None, {}, {"prompt_tokens": True, "completion_tokens": 3}])
def test_unknown_usage(value):
    assert token_usage(value) is None


@pytest.mark.parametrize("body", ["answer", "x" * 4001, "😀" * 2100])
@pytest.mark.parametrize("has_sources", [False, True])
def test_bot_sends_footer_for_replies(body, has_sources):
    async def scenario():
        msg = incoming()
        msg.mentions = [msg.guild.me]
        bot = SimpleNamespace(
            settings=settings(), user=msg.guild.me, server_log=Mock(), archive=AsyncMock(),
            busy=set(), slots=asyncio.Semaphore(4),
            assistant=SimpleNamespace(reply=AsyncMock(return_value=Reply(
                body + "\n-# copied", usage=(30, 5), sources=[JUMP] if has_sources else [],
            ))),
        )
        with patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[10])):
            await Bot.on_message(bot, msg)
        calls = msg.channel.send.call_args_list
        assert all(call.kwargs == {} for call in calls)
        assert all("copied" not in call.args[0] for call in calls)
        assert all(JUMP not in call.args[0] for call in calls)
        text = calls[-1].args[0]
        assert text.endswith("入力 30 / 出力 5 tokens")
    asyncio.run(scenario())


def test_stripped_ambient_skip_is_not_posted():
    client = client_for(response("(skip)\n-# copied footer"))
    reply = asyncio.run(Assistant(client, "model", SandboxSettings()).reply(
        1, "question", ambient=True,
    ))
    assert reply == "(skip)"


def test_teaser_cuts_at_sentence_boundary():
    text = "あ" * 60 + "。" + "い" * 200
    teaser = make_teaser(text)
    assert teaser == "あ" * 60 + "。\n続きはコチラから⬇️"


def test_teaser_hard_cut_adds_ellipsis():
    teaser = make_teaser("あ" * 300)
    assert teaser == "あ" * 100 + "…\n続きはコチラから⬇️"
