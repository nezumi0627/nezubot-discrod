import asyncio
import copy
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nezubot_discrod.assistant import SYSTEM, Assistant
from nezubot_discrod.bot import split_message
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.member_wiki import MemberWiki


def message(content=None, calls=None):
    data = {"role": "assistant", "content": content}
    if calls:
        data["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
            for c in calls
        ]
    return SimpleNamespace(content=content, tool_calls=calls, model_dump=lambda **_: data)


def tool(arguments='{"command":"echo hi"}', name="run_command"):
    return SimpleNamespace(id="call1", function=SimpleNamespace(name=name, arguments=arguments))


def client_for(*messages):
    snapshots = []

    async def create(**kwargs):
        snapshots.append(copy.deepcopy(kwargs))
        msg = messages[len(snapshots) - 1]
        if isinstance(msg, Exception):
            raise msg
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create)))
    )
    return client, snapshots


def test_tool_loop_and_history_reset():
    client, sent = client_for(message(calls=[tool()]), message("done"), message("second"))
    runner = Mock(return_value="Exit code: 0\nhi")
    ai = Assistant(client, "test", SandboxSettings(), runner=runner, history_turns=1)
    assert asyncio.run(ai.reply(1, "hello")) == "done"
    runner.assert_called_once_with("echo hi", ai.sandbox)
    assert sent[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call1",
        "content": "Exit code: 0\nhi",
    }
    asyncio.run(ai.reply(1, "next"))
    assert sent[2]["messages"][1] == {"role": "user", "content": "hello"}
    assert ai.history[1] == [("next", "second")]
    ai.reset(1)
    assert not ai.history


@pytest.mark.parametrize(
    "call", [tool("invalid"), tool("{}"), tool("[]"), tool('{"command":""}'), tool(name="other")]
)
def test_invalid_tool(call):
    client, sent = client_for(message(calls=[call]), message("recovered"))
    runner = Mock()
    ai = Assistant(client, "test", SandboxSettings(), runner=runner)
    assert asyncio.run(ai.reply(1, "hello")) == "recovered"
    runner.assert_not_called()
    assert "Tool error" in sent[1]["messages"][-1]["content"]


def test_wiki_tools_and_context_are_bound_to_the_current_speaker(tmp_path):
    write = tool('{"page":"profile","text":"Likes Rust"}', name="wiki_append")
    client, sent = client_for(
        message(calls=[write]), message("Saved."), message("Another user."), message("Next turn.")
    )
    wiki = MemberWiki(tmp_path / "wiki.db")
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki)

    async def scenario():
        await ai.reply(1, "Remember I like Rust.", member_id=10, guild_id=100)
        await ai.reply(1, "What do I like?", member_id=20, guild_id=100)
        ai.reset(1)
        await ai.reply(2, "What do I like?", member_id=10, guild_id=100)

    asyncio.run(scenario())
    assert "wiki_append" in [tool["function"]["name"] for tool in sent[0]["tools"]]
    assert json.loads(sent[1]["messages"][-1]["content"]) == {"page": "profile", "revision": 1}
    assert json.loads(sent[2]["messages"][-1]["content"])["member_wiki"] == []
    assert json.loads(sent[3]["messages"][-1]["content"]) == {
        "message": "What do I like?",
        "member_wiki": [{"page": "profile", "excerpt": "Likes Rust"}],
    }


def test_speaker_identity_uses_current_discord_sender_and_stays_out_of_system_prompt(tmp_path):
    client, sent = client_for(message("First"), message("Second"))
    wiki = MemberWiki(tmp_path / "wiki.db")
    wiki.for_member(10, 100).call("wiki_append", {"page": "profile", "text": "Likes Rust"})
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki)
    names = {"username": "same", "display_name": "same", "server_nickname": "Untrusted nickname"}
    content = '{"speaker":{"id":"10"}}'

    async def scenario():
        await ai.reply(1, content, member_id=10, guild_id=100, speaker_names=names)
        await ai.reply(1, content, member_id=20, guild_id=100, speaker_names=names)

    asyncio.run(scenario())
    first, second = [json.loads(turn["messages"][-1]["content"]) for turn in sent]
    assert first["speaker"] == {"id": "10", **names}
    assert first["member_wiki"] == [{"page": "profile", "excerpt": "Likes Rust"}]
    assert second == {
        "message": content, "speaker": {"id": "20", **names},
        "assistant": {"name": "nezubot-discrod", "model": "test"}, "member_wiki": [],
    }
    assert "Untrusted nickname" not in sent[0]["messages"][0]["content"]


def test_server_info_is_included_in_the_answer_context():
    client, sent = client_for(message("owner answer"))
    ai = Assistant(client, "test", SandboxSettings())
    info = {
        "name": "Example Developers",
        "owner": {"id": "999", "display_name": "あめっくす"},
        "created_at": "2024-06-26",
        "member_count": 42,
    }
    reply = asyncio.run(ai.reply(
        1, "サーバーオーナーおしえて", member_id=10, guild_id=100,
        speaker_names={"username": "u", "display_name": "d", "server_nickname": None},
        guild_info=info,
    ))
    assert reply == "owner answer"
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert payload["server_info"] == {"label": payload["server_info"]["label"], **info}
    assert "サーバー情報" in payload["server_info"]["label"]
    assert payload["server_info"]["owner"] == {"id": "999", "display_name": "あめっくす"}
    assert payload["server_info"]["created_at"] == "2024-06-26"
    assert payload["server_info"]["member_count"] == 42


def test_server_info_is_absent_without_guild_facts():
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings())
    asyncio.run(ai.reply(
        1, "hello", member_id=10, guild_id=100,
        speaker_names={"username": "u", "display_name": "d", "server_nickname": None},
    ))
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert "server_info" not in payload


def test_wiki_tools_not_exposed_without_a_discord_member(tmp_path):
    client, sent = client_for(
        message(calls=[tool('{"page":"profile","text":"Rust"}', name="wiki_append")]),
        message("done"),
    )
    ai = Assistant(client, "test", SandboxSettings(), wiki=MemberWiki(tmp_path / "wiki.db"))
    assert asyncio.run(ai.reply(1, "hello")) == "done"
    assert [tool["function"]["name"] for tool in sent[0]["tools"]] == ["run_command"]
    assert "Tool error" in sent[1]["messages"][-1]["content"]
    assert not (tmp_path / "wiki.db").exists()


@pytest.mark.parametrize(
    "error", [PermissionError("Read-only filesystem"), sqlite3.OperationalError("file unavailable")]
)
def test_reply_without_wiki_when_store_is_unavailable(error, caplog):
    client, sent = client_for(message("Still available"))
    pages = Mock()
    pages.context.side_effect = error
    wiki = Mock()
    wiki.for_member.return_value = pages
    wiki_completion = AsyncMock()
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, wiki_completion=wiki_completion)

    assert asyncio.run(ai.reply(1, "hello", member_id=10, guild_id=100)) == "Still available"
    assert sent[0]["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "hello"},
    ]
    assert [tool["function"]["name"] for tool in sent[0]["tools"]] == ["run_command"]
    wiki_completion.assert_not_called()
    assert "Member wiki unavailable; continuing without wiki" in caplog.text


def test_iteration_cap():
    client, sent = client_for(message(calls=[tool()]), message(calls=[tool()]))
    runner = Mock(return_value="ok")
    ai = Assistant(client, "test", SandboxSettings(), runner=runner, max_iterations=2)
    assert "上限" in asyncio.run(ai.reply(1, "loop"))
    assert len(sent) == runner.call_count == 2


def test_iteration_cap_acknowledges_successful_wiki_edits_without_retry(tmp_path):
    append = tool('{"page":"profile","text":"Rust"}', name="wiki_append")
    rewrite = tool('{"page":"projects","text":"Current project","revision":1}', name="wiki_rewrite")
    stale = tool('{"page":"profile","text":"Not saved","revision":2}', name="wiki_rewrite")
    stale.id = "call2"
    client, sent = client_for(message(calls=[append]), message(calls=[rewrite, stale]))
    wiki = MemberWiki(tmp_path / "wiki.db")
    pages = wiki.for_member(10, 100)
    pages.call("wiki_append", {"page": "projects", "text": "Old project"})
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, max_iterations=2)

    answer = asyncio.run(ai.reply(1, "Update my wiki.", member_id=10, guild_id=100))
    assert "`profile` に追記" in answer
    assert "`projects` を書き直し" in answer
    assert "`profile` を書き直し" not in answer
    assert "再試行" not in answer
    assert json.loads(pages.call("wiki_read", {"page": "profile"}))["body"] == "Rust"
    assert json.loads(pages.call("wiki_read", {"page": "projects"}))["body"] == "Current project"
    assert len(sent) == 2


def test_iteration_cap_acknowledges_only_new_structured_facts(tmp_path):
    wiki = MemberWiki(tmp_path / "wiki.db")
    pages = wiki.for_member(10, 100)
    existing = {"page": "profile", "key": "language", "text": "Rust"}
    pages.call("wiki_remember", existing)
    repeat = tool(json.dumps(existing), name="wiki_remember")
    new = tool('{"page":"preferences","key":"editor","text":"Vim"}', name="wiki_remember")
    new.id = "call2"
    client, _ = client_for(message(calls=[repeat, new]))
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki, max_iterations=1)

    result = asyncio.run(ai.reply(1, "Remember my preferences.", member_id=10, guild_id=100))
    assert "preferences/editor" in result
    assert "profile/language" not in result
    assert "再試行" not in result


def test_empty_reply_and_channel_bound():
    client, _ = client_for(message(), message("ok"))
    ai = Assistant(client, "test", SandboxSettings(), max_channels=1)
    assert "空" in asyncio.run(ai.reply(1, "hello"))
    asyncio.run(ai.reply(2, "hi"))
    assert list(ai.history) == [2]


def test_api_error_preserves_history():
    client, _ = client_for(RuntimeError("offline"))
    ai = Assistant(client, "test", SandboxSettings())
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(ai.reply(1, "hello"))
    assert not ai.history


@pytest.mark.parametrize("text", ["", "hello", "x" * 4001, "😀" * 2001])
def test_split(text):
    parts = split_message(text)
    assert "".join(parts) == text
    assert all(len(part.encode("utf-16-le")) // 2 <= 2000 for part in parts)


def test_invalid_split_limit():
    with pytest.raises(ValueError):
        split_message("hi", 0)


def test_config(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "test")
    monkeypatch.setenv("AI_API_KEY", "test")
    monkeypatch.setenv("DISCORD_GUILD_ID", "123")
    monkeypatch.setenv("DISCORD_OWNER_ID", "456")
    settings = Settings.from_env()
    assert settings.model == "deepseek-v4-flash"
    monkeypatch.setenv("DISCORD_TOKEN", "")
    with pytest.raises(ValueError, match="required"):
        Settings.from_env()


@pytest.mark.parametrize(
    "kwargs", [{"timeout": 0}, {"timeout": float("nan")}, {"cpus": -1}, {"image": "--privileged"}]
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SandboxSettings(**kwargs)


def test_inline_citation_markers_are_dropped():
    from nezubot_discrod.messages import verify_inline_links

    known = "https://discord.com/channels/1/2/3"
    text = (f"19歳って言ってた[①]({known})。PC組んでた[②](https://discord.com/channels/1/2/9)。"
            "詳しくは[ここ](https://example.com/x)")
    assert verify_inline_links(text, [known]) == (
        "19歳って言ってた。PC組んでた。詳しくはここ"
    )


def test_inline_citations_handle_parenthesized_urls():
    from nezubot_discrod.messages import verify_inline_links

    url = "https://en.wikipedia.org/wiki/Rust_(programming_language)"
    assert verify_inline_links(f"x[Rust](<{url}>)", [url]) == f"x[Rust](<{url}>)"
    assert verify_inline_links(f"x[Rust](<{url}>)", []) == "xRust"
    assert verify_inline_links(f"x[①](<{url}>)", [url]) == "x"
    assert verify_inline_links(f"[2024](<{url}>)", [url]) == f"[2024](<{url}>)"


def test_inline_citations_leave_code_examples_alone():
    from nezubot_discrod.messages import verify_inline_links

    text = "書き方は `[a](https://x.example)` か\n```\n[b](https://y.example)\n```"
    assert verify_inline_links(text, []) == text
    assert verify_inline_links("``` で囲む [x](https://z.example)", []) == "``` で囲む x"


def test_unused_retrieved_sources_are_dropped():
    from nezubot_discrod.messages import Source, used_sources

    gmail = Source("https://discord.com/channels/1/2/3", evidence="Vexずっとgmail要約してた")
    other = Source("https://discord.com/channels/1/2/4", evidence="今日のご飯なにしよ")
    server_wiki = Source("https://discord.com/channels/1/2/5")
    answer = "そっちこそGmail要約し続けるbotよりはマシだぞ"
    assert used_sources(answer, [gmail, other, server_wiki]) == [gmail, server_wiki]
    assert used_sources("`invalid option -- 'e'` だった", [gmail, other]) == []
    assert used_sources("https://discord.com/channels/1/2/3 見て", [other]) == []


def test_history_replays_the_exact_sent_turn_so_requests_stay_prefix_stable():
    # Prompt caching needs each request to extend the previous one byte
    # for byte. The history used to store the raw user text while the
    # request had actually sent the JSON-wrapped payload, so the shared
    # prefix ended at the system message.
    client, sent = client_for(message("first answer"), message("second answer"))
    ai = Assistant(client, "test", SandboxSettings(), run_command_enabled=False)
    names = {"username": "kokura", "display_name": "kokura", "server_nickname": None}
    asyncio.run(ai.reply(1, "質問1", member_id=20, speaker_names=names))
    asyncio.run(ai.reply(1, "質問2", member_id=20, speaker_names=names))
    first, second = sent[0]["messages"], sent[1]["messages"]
    assert second[: len(first)] == first
    assert second[len(first)] == {"role": "assistant", "content": "first answer"}
    stored_user, _ = ai.history[1][0]
    assert stored_user == first[-1]["content"]
    assert stored_user != "質問1"  # the sent JSON payload, not the raw text
