import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nezubot_discrod.assistant import Assistant, web_lookup_requested
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.server_log_tools import LogResult


@pytest.mark.parametrize("content", [
    "今日のAI関係のニュースを Web で調べて",
    "Wikipedia で SQLite の歴史を調べて",
    "Webで調べて", "Ｗｅｂ　で調べて", "ググって", "ネットで検索",
    "ウェブで調べて", "ウィキペディアで調べて",
    "search the web for AI news", "Look up on Wikipedia", "web search SQLite",
    "look up SQLite on Wikipedia", "search online for SQLite history",
])
def test_web_lookup_intent(content):
    assert web_lookup_requested(content)


@pytest.mark.parametrize("content", [
    "", "俺について簡単にまとめて", "vexってどう思う？", "SQLiteの使い方",
    "私はPythonについて知りたい", "Webアプリを作って", "Wikipediaって信用できる？",
])
def test_non_web_lookup_intent(content):
    assert not web_lookup_requested(content)


def assistant_with_context(monkeypatch, injected):
    retrieve = AsyncMock(return_value=LogResult(injected))
    monkeypatch.setattr("nezubot_discrod.assistant.pre_retrieve", retrieve)
    store = Mock()
    store.recent.return_value = []
    wiki = Mock()
    wiki.for_member.return_value.context.return_value = []
    completion = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
    ))
    assistant = Assistant(Mock(), "test", SandboxSettings(), server_log=store, wiki=wiki,
                          wiki_completion=completion, web=Mock())
    return assistant, retrieve, store, wiki, completion


def reply(assistant, content):
    return asyncio.run(assistant.reply(
        10, content, member_id=20, guild_id=100, channel_ids=[10],
    ))


@pytest.mark.parametrize("content", [
    "今日のAI関係のニュースを Web で調べて", "Wikipedia で SQLite の歴史を調べて",
    "ググって", "ネットで検索", "search the web", "look up on Wikipedia",
])
def test_web_lookup_skips_all_automatic_private_context(monkeypatch, content):
    assistant, retrieve, store, wiki, completion = assistant_with_context(monkeypatch, "private")
    store.recent.side_effect = AssertionError("must not fetch recent archive messages")
    wiki.for_member.return_value.context.side_effect = AssertionError("must not read private wiki")
    assert reply(assistant, content) == "done"
    retrieve.assert_not_awaited()
    store.recent.assert_not_called()
    wiki.for_member.return_value.context.assert_not_called()
    sent = completion.call_args.kwargs
    assert sent.get("private_read", False) is False
    payload = json.loads(sent["messages"][-1]["content"])
    assert set(payload) == {"message", "now"}
    assert payload["message"] == content
    assert payload["now"]["timezone"] == "Asia/Tokyo"
    assert sent["server_log"] is not None
    assert sent["wiki"] is wiki.for_member.return_value
    assert sent["web"] is assistant.web


@pytest.mark.parametrize(("injected", "private_read"), [
    ("", False), (" \n ", False), ("archive evidence", True),
])
def test_non_web_retrieval_taints_only_nonempty_injection(monkeypatch, injected, private_read):
    assistant, retrieve, store, wiki, completion = assistant_with_context(monkeypatch, injected)
    reply(assistant, "俺について簡単にまとめて")
    retrieve.assert_awaited_once()
    store.recent.assert_called_once_with(10, 16)
    wiki.for_member.return_value.context.assert_called_once()
    sent = completion.call_args.kwargs
    assert sent.get("private_read", False) is private_read
    payload = json.loads(sent["messages"][-1]["content"])
    assert ("retrieved_server_context" in payload) is private_read


def test_empty_retrieval_does_not_clear_other_private_context(monkeypatch):
    assistant, _, _, wiki, completion = assistant_with_context(monkeypatch, "")
    wiki.for_member.return_value.context.return_value = [{"page": "profile", "excerpt": "private"}]
    reply(assistant, "俺について簡単にまとめて")
    assert completion.call_args.kwargs["private_read"] is True


def test_web_lookup_does_not_clear_existing_history_taint(monkeypatch):
    assistant, retrieve, _, _, completion = assistant_with_context(monkeypatch, "")
    reply(assistant, "俺について簡単にまとめて")
    retrieve.reset_mock()
    reply(assistant, "search the web for SQLite")
    retrieve.assert_not_awaited()
    assert completion.call_args.kwargs["private_read"] is True
