import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import discord
import pytest
from test_assistant import client_for, message, tool
from test_bot import settings
from test_codex import FakeProcess, conversation, finished
from test_codex import tool as codex_tool
from test_server_log import record

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.bot import Bot, search_logs
from nezubot_discrod.codex import AppServer, CodexSecurityError
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.member_wiki import MemberWiki
from nezubot_discrod.prompts import LOG_GUIDANCE, LOG_SYSTEM, WIKI_LOG_SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_discord import (
    allowed_channel_ids,
    audience_channel_ids,
    backfill_channel,
    backfill_guild,
    is_call,
)
from nezubot_discrod.server_log_tools import LOG_TOOLS, LogTools


@pytest.fixture
def archive(tmp_path):
    store = ServerLog(tmp_path / "data" / "server_log.db")
    store.upsert_many([record(1), record(2), record(3, channel="11", text="secret Rust")])
    return store


def audience_channel(number, public=True, kind=discord.ChannelType.text):
    target = SimpleNamespace(id=number, type=kind, overwrites={}, is_nsfw=Mock(return_value=False))
    target.permissions_for = Mock(return_value=SimpleNamespace(
        view_channel=public, read_message_history=public,
    ))
    return target


def audience_scope(reply, *sources):
    channels = {value.id: value for value in (reply, *sources)}
    reply.guild = SimpleNamespace(default_role=object(), get_channel_or_thread=channels.get)
    return audience_channel_ids(SimpleNamespace(log_threads={}), reply, list(channels))


@pytest.mark.parametrize("public_reply,expected", [(True, [10, 12]), (False, [10])])
def test_automatic_audience_scope_public_and_private_reply(public_reply, expected):
    role = object()
    channels = {10: audience_channel(10, public_reply), 11: audience_channel(11, False),
                12: audience_channel(12), 13: audience_channel(13),
                14: audience_channel(14, kind=discord.ChannelType.private_thread)}
    guild = SimpleNamespace(default_role=role, get_channel_or_thread=channels.get)
    channels[10].guild = guild
    client = SimpleNamespace(log_threads={})
    assert audience_channel_ids(client, channels[10], [10, 11, 12, 14, 999]) == expected
    channels[10].permissions_for.assert_called_with(role)


@pytest.mark.parametrize("target_type", [discord.Role, discord.Member])
@pytest.mark.parametrize("permission", ["view_channel", "read_message_history"])
def test_audience_scope_excludes_any_explicit_read_denial(target_type, permission):
    reply, source, safe = (audience_channel(value) for value in (10, 11, 12))
    source.overwrites[discord.Object(id=20, type=target_type)] = discord.PermissionOverwrite(
        **{permission: False},
    )
    assert audience_scope(reply, source, safe) == [10, 12]


@pytest.mark.parametrize("kind", [
    discord.ChannelType.public_thread, discord.ChannelType.news_thread,
])
@pytest.mark.parametrize("permission", ["view_channel", "read_message_history"])
def test_audience_scope_checks_thread_parent_overwrites(kind, permission):
    reply, parent = audience_channel(10), audience_channel(11)
    thread = audience_channel(12, kind=kind)
    thread.parent = parent
    del thread.overwrites
    assert audience_scope(reply, thread) == [10, 12]
    parent.overwrites[object()] = discord.PermissionOverwrite(**{permission: False})
    assert audience_scope(reply, thread) == [10]


def test_audience_scope_checks_thread_own_overwrites_if_present():
    reply = audience_channel(10)
    thread = audience_channel(12, kind=discord.ChannelType.public_thread)
    thread.parent = audience_channel(11)
    thread.overwrites[object()] = discord.PermissionOverwrite(view_channel=False)
    assert audience_scope(reply, thread) == [10]


@pytest.mark.parametrize("missing", ["overwrites", "parent", "parent_overwrites"])
def test_audience_scope_fails_closed_for_missing_permissions(missing):
    reply, source = audience_channel(10), audience_channel(11)
    if missing == "overwrites":
        del source.overwrites
    else:
        source.type = discord.ChannelType.public_thread
        source.parent = None if missing == "parent" else audience_channel(12)
        if source.parent is not None:
            del source.parent.overwrites
    assert audience_scope(reply, source) == [10]


def test_audience_scope_allows_unrelated_denials_and_explicit_read_allows():
    reply, source = audience_channel(10), audience_channel(11)
    source.overwrites[object()] = discord.PermissionOverwrite(
        view_channel=True, read_message_history=True, send_messages=False,
    )
    assert audience_scope(reply, source) == [10, 11]


@pytest.mark.parametrize("reply_nsfw", [False, True])
@pytest.mark.parametrize("source_nsfw", [False, True])
@pytest.mark.parametrize("thread", [False, True])
def test_audience_scope_respects_source_and_reply_age_restrictions(reply_nsfw, source_nsfw, thread):
    reply, source = audience_channel(10), audience_channel(11)
    reply.is_nsfw.return_value = reply_nsfw
    if thread:
        source.type = discord.ChannelType.public_thread
        source.parent = audience_channel(12)
        source.parent.is_nsfw.return_value = source_nsfw
        source.is_nsfw = lambda: discord.Thread.is_nsfw(source)
    else:
        source.is_nsfw.return_value = source_nsfw
    expected = [10, 11] if not source_nsfw or reply_nsfw else [10]
    assert audience_scope(reply, source) == expected


@pytest.mark.parametrize("metadata", [None, "missing", "error"])
def test_audience_scope_excludes_unknown_source_age_restrictions(metadata):
    reply, source = audience_channel(10), audience_channel(11)
    if metadata == "missing":
        del source.is_nsfw
    elif metadata == "error":
        source.is_nsfw.side_effect = discord.ClientException("Unknown age restriction")
    else:
        source.is_nsfw.return_value = metadata
    assert audience_scope(reply, source) == [10]


def test_dm_automatic_scope_excludes_nsfw_and_unknown_channels():
    safe, restricted = audience_channel(10), audience_channel(11)
    restricted.is_nsfw.return_value = True
    channels = {10: safe, 11: restricted}
    client = SimpleNamespace(get_channel=channels.get, log_threads={})
    assert audience_channel_ids(client, SimpleNamespace(id=100), [10, 11, 99]) == [10]
    assert audience_channel_ids(SimpleNamespace(), SimpleNamespace(id=100), [10, 11]) == []


@pytest.mark.parametrize("text", ["nezubot-discrod、これ何？", "Nezu help", "@nezubot-discrod"])
def test_display_name_is_not_a_call(text):
    user = SimpleNamespace(id=99)
    msg = SimpleNamespace(
        mentions=[], reference=None, content=text,
        guild=SimpleNamespace(me=SimpleNamespace(display_name="nezubot-discrod")),
    )
    assert not asyncio.run(is_call(msg, SimpleNamespace(id=99)))
    msg.mentions = [user]
    assert asyncio.run(is_call(msg, user))


def test_reply_trigger_with_uncached_message():
    user = SimpleNamespace(id=99)
    msg = SimpleNamespace(
        mentions=[],
        reference=SimpleNamespace(message_id=1, resolved=None),
        content="continue",
        guild=SimpleNamespace(me=SimpleNamespace(display_name="Nezu")),
        channel=SimpleNamespace(fetch_message=AsyncMock(return_value=SimpleNamespace(author=user))),
    )
    assert asyncio.run(is_call(msg, user))
    msg.channel.fetch_message.return_value.author = SimpleNamespace(id=100)
    assert not asyncio.run(is_call(msg, user))
    msg.channel.fetch_message.side_effect = discord.NotFound(Mock(status=404), "gone")
    assert not asyncio.run(is_call(msg, user))
    msg.mentions = [user]
    assert asyncio.run(is_call(msg, user))


def channel(number, *, visible=True, history=True):
    return SimpleNamespace(
        id=number,
        history=Mock(),
        permissions_for=Mock(
            return_value=SimpleNamespace(
                view_channel=visible, read_message_history=history, manage_threads=False
            )
        ),
    )


def test_scope_includes_archived_threads_and_checks_private_membership():
    member = SimpleNamespace(id=456)
    private = Mock(spec=discord.Thread)
    private.id, private.owner_id = 13, 200
    private.is_private.return_value = True
    private.permissions_for.return_value = SimpleNamespace(
        view_channel=True, read_message_history=True, manage_threads=False
    )
    private.fetch_member = AsyncMock(side_effect=discord.NotFound(Mock(status=404), "not joined"))
    guild = SimpleNamespace(
        channels=[channel(10), channel(11, visible=False), channel(12, history=False)],
        threads=[private],
        get_member=Mock(return_value=None),
        fetch_member=AsyncMock(return_value=member),
    )
    client = SimpleNamespace(
        settings=settings(), get_guild=Mock(return_value=guild), log_threads={14: channel(14)}
    )
    assert asyncio.run(allowed_channel_ids(client, member)) == [10, 14]
    guild.fetch_member.assert_awaited_once_with(456)
    private.owner_id = member.id
    guild.get_member.return_value = member
    assert asyncio.run(allowed_channel_ids(client, member)) == [10, 14]
    private.fetch_member.return_value = member
    private.fetch_member.side_effect = None
    assert asyncio.run(allowed_channel_ids(client, member)) == [10, 13, 14]
    guild.fetch_member.side_effect = discord.Forbidden(Mock(status=403), "denied")
    assert asyncio.run(allowed_channel_ids(client, member)) == []
    client.get_guild.return_value = None
    assert asyncio.run(allowed_channel_ids(client, member)) == []


def test_openai_log_tools_context_and_history_isolation(archive, tmp_path):
    client, sent = client_for(
        message(calls=[tool('{"query":"Rust"}', "search_server_log")]),
        message("done"),
        message("second"),
        message("third"),
    )
    ai = Assistant(
        client,
        "test",
        SandboxSettings(),
        server_log=archive,
        wiki=MemberWiki(tmp_path / "wiki" / "wiki.db"),
    )
    asyncio.run(ai.reply(10, "who?", member_id=20, guild_id=100, channel_ids=[10], message_id=2))
    assert sent[0]["messages"][0]["content"] == WIKI_LOG_SYSTEM
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert "member_wiki" in payload and "recent_messages" in payload
    assert "/10/1" in payload["recent_messages"]
    assert "/10/2" not in payload["recent_messages"]
    assert "secret" not in payload["recent_messages"]
    result = sent[1]["messages"][-1]["content"]
    assert "/10/1" in result and "secret" not in result
    names = {entry["function"]["name"] for entry in sent[0]["tools"]}
    assert {"run_command", "wiki_list", "search_server_log", "read_server_log"} <= names
    asyncio.run(ai.reply(10, "new member", member_id=21, channel_ids=[10]))
    assert len(sent[2]["messages"]) == 2
    asyncio.run(ai.reply(10, "lost permission", member_id=20, channel_ids=[]))
    assert len(sent[3]["messages"]) == 2
    assert json.loads(sent[3]["messages"][-1]["content"])["recent_messages"] == "No messages found."
    ai.reset(10)
    assert not ai.history


INVALID_CALLS = [
    ("search_server_log", {}),
    ("search_server_log", []),
    ("search_server_log", {"query": "Rust", "channel_ids": [11]}),
    ("search_server_log", {"query": 1}),
    ("search_server_log", {"query": " "}),
    ("search_server_log", {"query": "Rust", "limit": True}),
    ("search_server_log", {"query": "Rust", "limit": 21}),
    ("search_server_log", {"query": "Rust", "author": None}),
    ("search_server_log", {"query": "Rust", "after": "not a date"}),
    ("read_server_log", {"message_id": 1}),
    ("read_server_log", {"message_id": "1", "direction": "any"}),
    ("read_server_log", {"message_id": "1", "limit": 31}),
    ("read_server_log", {"message_id": "1", "limit": 0}),
]


@pytest.mark.parametrize("name,args", INVALID_CALLS)
def test_log_validation_both_providers(archive, name, args):
    client, sent = client_for(message(calls=[tool(json.dumps(args), name)]), message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive)
    asyncio.run(ai.reply(10, "query", channel_ids=[10]))
    assert "Tool error" in sent[1]["messages"][-1]["content"]
    process = FakeProcess(conversation(codex_tool(args, name), *finished()))
    server = AppServer(process, SandboxSettings())
    asyncio.run(server.complete("test", [], "/empty", server_log=LogTools(archive, [10])))
    result = next(p["result"] for p in process.sent if p.get("id") == 99)
    assert result["success"] is False


def test_codex_log_tools_fixed_prompt_and_scope(archive):
    process = FakeProcess(
        conversation(
            codex_tool({"query": "Rust"}, "search_server_log", threadId="other"),
            codex_tool({"query": "Rust"}, "search_server_log"),
            codex_tool({"message_id": "3"}, "read_server_log"),
            codex_tool({"message_id": "1", "direction": "after", "limit": 1}, "read_server_log"),
            codex_tool({"kind": "person", "query": "42"}, "read_server_wiki"),
            *finished(),
        )
    )
    log = LogTools(archive, [10], server_wiki_channel=10)
    server = AppServer(process, SandboxSettings())
    assert asyncio.run(server.complete("test", [], "/empty", server_log=log)) == "done"
    thread = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert thread["baseInstructions"] == LOG_SYSTEM
    assert {tool["name"] for tool in thread["dynamicTools"]} == {
        "run_command",
        "search_server_log",
        "read_server_log",
        "collect_server_log",
        "read_server_wiki",
    }
    assert all(not tool["inputSchema"]["additionalProperties"] for tool in thread["dynamicTools"])
    replies = [p["result"] for p in process.sent if p.get("id") == 99]
    assert [r["success"] for r in replies] == [False, True, True, True, True]
    assert "secret" not in replies[1]["contentItems"][0]["text"]
    assert replies[2]["contentItems"][0]["text"] == "No messages found."
    assert "/10/2" in replies[3]["contentItems"][0]["text"]
    assert replies[4]["contentItems"][0]["text"] == '{"pages": []}'
    with pytest.raises(CodexSecurityError, match="fixed"):
        asyncio.run(
            server.complete(
                "test", [{"role": "system", "content": LOG_SYSTEM + "x"}], "/empty", server_log=log
            )
        )
    assert "untrusted" in LOG_GUIDANCE and "-#" in LOG_GUIDANCE
    assert "誰が/いつ/前に言ってた" in LOG_GUIDANCE
    assert all(not item["function"]["parameters"]["additionalProperties"] for item in LOG_TOOLS)


def test_codex_collection_dynamic_tool_builds_scoped_citations(archive):
    process = FakeProcess(conversation(
        codex_tool({"author": "20", "limit": 2}, "collect_server_log"), *finished(),
    ))
    server = AppServer(process, SandboxSettings())
    assert asyncio.run(server.complete("test", [], "/empty",
                                       server_log=LogTools(archive, [10]))) == "done"
    result = next(packet["result"] for packet in process.sent if packet.get("id") == 99)
    assert result["success"] is True and "message_count" in result["contentItems"][0]["text"]
    assert "secret Rust" not in result["contentItems"][0]["text"]
    assert len(server.sources) == 2
    assert server.sources[0].endswith("/10/2")
    assert all("/10/" in source for source in server.sources)
    assert server.sources[0].title_text == "#general · alice · 2026-01-01"
    assert server.private_read is True


def test_slash_search_is_ephemeral_scoped_and_bounded(archive):
    interaction = SimpleNamespace(
        guild_id=123,
        user=SimpleNamespace(id=456),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    bot = SimpleNamespace(settings=settings(), server_log=archive)
    with (
        patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[10])),
        patch.object(archive, "search", wraps=archive.search) as search,
    ):
        asyncio.run(search_logs(interaction, bot, "Rust"))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    search.assert_called_once_with("Rust", channel_ids=[10], limit=8)
    text = "".join(call.args[0] for call in interaction.followup.send.call_args_list)
    assert "/10/1" in text and "secret" not in text
    assert all(c.kwargs["ephemeral"] for c in interaction.followup.send.call_args_list)
    interaction.guild_id = None
    asyncio.run(search_logs(interaction, bot, "Rust"))
    interaction.response.send_message.assert_awaited_once()


def discord_message(number, target):
    return SimpleNamespace(
        id=number,
        guild=SimpleNamespace(id=123),
        channel=target,
        author=SimpleNamespace(id=20, name="alice", display_name="Alice", bot=True),
        created_at=discord.utils.utcnow(),
        edited_at=None,
        reference=None,
        content="archived Rust",
        embeds=[],
        attachments=[],
        stickers=[],
    )


def test_backfill_is_batched_resumable_and_tolerates_missing_channel(archive):
    target = SimpleNamespace(id=10, name="general", parent=None)
    messages = [discord_message(i, target) for i in range(100, 205)]

    async def history(**kwargs):
        start = kwargs["after"].id if kwargs["after"] else 0
        for msg in messages:
            if msg.id > start:
                yield msg

    target.history = Mock(side_effect=history)
    with patch.object(archive, "upsert_many", wraps=archive.upsert_many) as upsert:
        asyncio.run(backfill_channel(archive, target))
        assert upsert.call_count == 1
        assert upsert.call_args.kwargs["cursor_channel_id"] == 10
        assert archive.get_cursor(10) == "204"
        asyncio.run(backfill_channel(archive, target))
        assert upsert.call_count == 1
    assert target.history.call_args.kwargs["after"].id == 204
    assert target.history.call_args.kwargs["oldest_first"] is True

    async def missing(**kwargs):
        raise discord.NotFound(Mock(status=404), "gone")
        yield

    target.history = missing
    asyncio.run(backfill_channel(archive, target))


@pytest.mark.parametrize("error", [
    discord.HTTPException(Mock(status=429), "rate limited"),
    discord.HTTPException(Mock(status=500), "server error"),
    discord.HTTPException(Mock(status=502), "bad gateway"),
    discord.HTTPException(Mock(status=503), "unavailable"),
    discord.HTTPException(Mock(status=504), "timeout"),
    aiohttp.ClientConnectionError("connection lost"),
    aiohttp.ClientPayloadError("incomplete response"),
    TimeoutError("timed out"),
    ConnectionResetError("reset"),
])
def test_backfill_retries_transient_failure_from_last_persisted_message(archive, error):
    target = SimpleNamespace(id=10, name="general", parent=None)
    messages = [discord_message(i, target) for i in range(100, 205)]
    attempts = []
    written = []
    persist = archive.upsert_many

    def persist_batch(batch, **kwargs):
        written.extend(row["id"] for row in batch)
        persist(batch, **kwargs)

    async def history(**kwargs):
        after = kwargs["after"].id if kwargs["after"] else None
        attempts.append(after)
        assert kwargs["limit"] is None and kwargs["oldest_first"] is True
        for msg in messages:
            if after is None or msg.id > after:
                yield msg
                if len(attempts) == 1 and msg.id == 201:
                    raise error

    async def pause(delay):
        assert delay == 1
        assert archive.get_cursor(10) == "201"
        assert archive.read("201", channel_ids=[10])
        assert not archive.read("202", channel_ids=[10])

    target.history = history
    with (
        patch(
            "nezubot_discrod.server_log_discord.asyncio.sleep", AsyncMock(side_effect=pause),
        ) as sleep,
        patch.object(archive, "upsert_many", side_effect=persist_batch) as upsert,
    ):
        asyncio.run(backfill_channel(archive, target))
    sleep.assert_awaited_once_with(1)
    assert attempts == [None, 201]
    assert written == [str(i) for i in range(100, 205)]
    assert upsert.call_count == 2
    assert archive.get_cursor(10) == "204"
    with archive.connect() as db:
        assert {row[0] for row in db.execute("SELECT id FROM messages")} >= {
            str(i) for i in range(100, 205)
        }


def test_backfill_caps_backoff_and_empty_retry_does_not_advance_cursor(archive):
    target = SimpleNamespace(id=10, name="general", parent=None)
    attempts = []

    async def history(**kwargs):
        attempts.append(kwargs["after"])
        if len(attempts) <= 8:
            raise discord.HTTPException(Mock(status=503), "unavailable")
        return
        yield

    target.history = history
    with patch("nezubot_discrod.server_log_discord.asyncio.sleep", AsyncMock()) as sleep:
        asyncio.run(backfill_channel(archive, target))
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2, 4, 8, 16, 32, 60, 60]
    assert attempts == [None] * 9
    assert archive.get_cursor(10) is None


@pytest.mark.parametrize("status", [403, 404])
def test_backfill_skips_inaccessible_channel_and_flushes_partial_batch(archive, status):
    target = SimpleNamespace(id=10, name="general", parent=None)

    async def history(**kwargs):
        yield discord_message(100, target)
        raise discord.HTTPException(Mock(status=status), "inaccessible")

    target.history = Mock(side_effect=history)
    with patch("nezubot_discrod.server_log_discord.asyncio.sleep", AsyncMock()) as sleep:
        asyncio.run(backfill_channel(archive, target))
    sleep.assert_not_awaited()
    assert target.history.call_count == 1
    assert archive.get_cursor(10) == "100"
    assert archive.read("100", channel_ids=[10])


def test_backfill_cancellation_during_backoff_keeps_durable_cursor(archive):
    target = SimpleNamespace(id=10, name="general", parent=None)

    async def history(**kwargs):
        yield discord_message(100, target)
        raise discord.HTTPException(Mock(status=503), "unavailable")

    target.history = history
    with (
        patch("nezubot_discrod.server_log_discord.asyncio.sleep", AsyncMock(
            side_effect=asyncio.CancelledError,
        )),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(backfill_channel(archive, target))
    assert archive.get_cursor(10) == "100"


def test_backfill_does_not_retry_nontransient_http_error(archive):
    target = SimpleNamespace(id=10, name="general", parent=None)

    async def history(**kwargs):
        raise discord.HTTPException(Mock(status=400), "bad request")
        yield

    target.history = history
    with (
        patch("nezubot_discrod.server_log_discord.asyncio.sleep", AsyncMock()) as sleep,
        pytest.raises(discord.HTTPException),
    ):
        asyncio.run(backfill_channel(archive, target))
    sleep.assert_not_awaited()
    assert archive.get_cursor(10) is None


def test_archive_runs_before_bot_and_trigger_filters(archive):
    target = SimpleNamespace(id=10, name="general", parent=None)
    msg = discord_message(100, target)
    bot = SimpleNamespace(settings=settings(), server_log=archive, user=SimpleNamespace(id=99))
    bot.archive = lambda msg: Bot.archive(bot, msg)
    asyncio.run(Bot.on_message(bot, msg))
    assert archive.read("100", channel_ids=[10])
    asyncio.run(Bot.on_raw_message_delete(bot, SimpleNamespace(guild_id=123, message_id=100)))
    assert archive.read("100", channel_ids=[10]) == []


def test_raw_edit_fetches_complete_message_and_ignores_other_guild(archive):
    target = SimpleNamespace(id=10, name="general", parent=None)
    target.fetch_message = AsyncMock(return_value=discord_message(100, target))
    bot = SimpleNamespace(
        settings=settings(), server_log=archive, get_channel=Mock(return_value=target)
    )
    bot.archive = lambda msg: Bot.archive(bot, msg)
    event = SimpleNamespace(guild_id=123, channel_id=10, message_id=100)
    asyncio.run(Bot.on_raw_message_edit(bot, event))
    assert archive.read("100", channel_ids=[10], limit=1)[0].text == "archived Rust"
    event.guild_id = 456
    asyncio.run(Bot.on_raw_message_edit(bot, event))
    target.fetch_message.assert_awaited_once()


def test_backfill_enumerates_forum_and_archived_threads(archive):
    visible = channel(10)
    visible.threads = []
    forum = channel(20)
    del forum.history
    forum.threads = [channel(21)]

    async def archived(**kwargs):
        yield channel(22)

    visible.archived_threads = archived
    forum.archived_threads = archived
    guild = SimpleNamespace(
        me=SimpleNamespace(id=99), text_channels=[visible], threads=[channel(11)], forums=[forum],
        voice_channels=[], stage_channels=[],
    )
    bot = SimpleNamespace(
        settings=settings(), server_log=archive, log_threads={}, get_guild=Mock(return_value=guild)
    )
    with patch("nezubot_discrod.server_log_discord.backfill_channel", AsyncMock()) as fill:
        asyncio.run(backfill_guild(bot))
    assert {c.args[1].id for c in fill.call_args_list} == {10, 11, 21, 22}


def test_backfill_includes_voice_and_stage_text_but_not_hidden_histories(archive):
    voice, stage = Mock(spec=discord.VoiceChannel), Mock(spec=discord.StageChannel)
    for target, number in [(voice, 30), (stage, 40)]:
        target.id, target.name, target.parent = number, "chat", None
        target.category = None
        target.permissions_for.return_value = SimpleNamespace(
            view_channel=True, read_message_history=True,
        )

        async def history(*, after, limit, oldest_first, target=target):
            assert after is None and limit is None and oldest_first is True
            yield discord_message(target.id * 10, target)

        target.history.side_effect = history
    hidden_voice, hidden_stage = channel(31, visible=False), channel(41, history=False)
    guild = SimpleNamespace(
        me=SimpleNamespace(id=99), text_channels=[], threads=[], forums=[],
        voice_channels=[voice, hidden_voice], stage_channels=[stage, hidden_stage],
    )
    bot = SimpleNamespace(
        settings=settings(), server_log=archive, log_threads={}, get_guild=Mock(return_value=guild),
    )
    asyncio.run(backfill_guild(bot))
    assert archive.read("300", channel_ids=[30])
    assert archive.read("400", channel_ids=[40])
    assert not archive.read("300", channel_ids=[40])
    assert archive.get_cursor(30) == "300"
    assert archive.get_cursor(40) == "400"
    hidden_voice.history.assert_not_called()
    hidden_stage.history.assert_not_called()


def test_backfill_does_not_overwrite_newer_edit(archive):
    value = record(1, text="new Rust")
    value["edited_timestamp"] = "2026-01-02T00:00:00Z"
    archive.upsert(value)
    archive.upsert(record(1, text="stale Rust"))
    assert archive.read("1", channel_ids=[10])[0].text == "new Rust"


@pytest.mark.parametrize("installed", [True, False])
def test_embedder_starts_only_with_importable_fastembed(installed):
    bot = SimpleNamespace(
        settings=settings(), server_log=Mock(), tree=Mock(sync=AsyncMock(return_value=[]))
    )
    with (
        patch("nezubot_discrod.bot.load_features"),
        patch(
            "nezubot_discrod.bot.importlib.import_module",
            side_effect=None if installed else ImportError("not installed"),
        ),
    ):
        asyncio.run(Bot.setup_hook(bot))
    assert bot.server_log.start_embedder.call_count == int(installed)


def test_ready_schedules_only_one_nonblocking_backfill():
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def fill():
            started.set()
            await finish.wait()

        bot = SimpleNamespace(settings=settings(), user=None, backfill_task=None, backfill=fill)
        await Bot.on_ready(bot)
        original = bot.backfill_task
        await asyncio.wait_for(started.wait(), 5)
        assert not original.done()
        await Bot.on_ready(bot)
        assert bot.backfill_task is original
        finish.set()
        await original

    asyncio.run(run())


def test_bot_registers_search_and_shares_one_log():
    async def run():
        with (
            patch("nezubot_discrod.bot.ServerLog") as factory,
            patch("nezubot_discrod.bot.AsyncOpenAI", return_value=AsyncMock()),
            patch("nezubot_discrod.bot.ImprovementService"),
        ):
            bot = Bot(settings())
            try:
                assert bot.tree.get_command("search") is not None
                assert bot.assistant.server_log is bot.server_log is factory.return_value
                factory.assert_called_once_with(checkout=None, guild_id=123)
            finally:
                await bot.close()

    asyncio.run(run())


def test_deleted_threads_leave_the_permission_registry():
    bot = SimpleNamespace(settings=settings(), log_threads={10: object()})
    asyncio.run(Bot.on_raw_thread_delete(bot, SimpleNamespace(guild_id=123, thread_id=10)))
    assert bot.log_threads == {}


def test_log_context_routes_through_codex_completion_without_wiki(archive):
    completion = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=message("done"))])
    )
    ai = Assistant(
        Mock(), "test", SandboxSettings(), server_log=archive, wiki_completion=completion
    )
    asyncio.run(ai.reply(10, "who?", channel_ids=[10]))
    args = completion.call_args.kwargs
    assert args["wiki"] is None and isinstance(args["server_log"], LogTools)
    assert args["server_log"].channel_ids == ("10",)
    assert args["messages"][0]["content"] == LOG_SYSTEM


def test_log_tools_and_slash_recover_from_unavailable_database(archive):
    with patch.object(archive, "search", side_effect=OSError("unavailable")):
        assert "unavailable" in LogTools(archive, [10]).call("search_server_log", {"query": "Rust"})
        interaction = SimpleNamespace(
            guild_id=123,
            user=SimpleNamespace(id=456),
            response=SimpleNamespace(defer=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        bot = SimpleNamespace(settings=settings(), server_log=archive)
        with patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[10])):
            asyncio.run(search_logs(interaction, bot, "Rust"))
        assert "失敗" in interaction.followup.send.call_args.args[0]
