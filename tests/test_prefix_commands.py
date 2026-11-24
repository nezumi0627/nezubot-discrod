import asyncio
import logging
import sqlite3
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from nezubot_discrod.bot import Bot, deliver_answer
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.prefix_commands import (
    HELP_TEXT,
    handle_prefix_command,
    parse_prefix_command,
    record_answer_authors,
)
from nezubot_discrod.status import (
    build_status_embed,
    collect_status,
    format_size,
    format_uptime,
)

TEACHER = "teacher-model"
STATIC = "static-model"
BOT_ID = 99
OWNER_ID = 777
GUILD_OWNER_ID = 999


@pytest.mark.parametrize(
    "content,expected",
    [
        ("n!help", "help"),
        ("N!HELP", "help"),
        ("n!Status", "status"),
        ("n!del", "del"),
        ("  n!del  ", "del"),
        ("n!status please", "status"),
        ("n!unknown", None),
        ("n!", None),
        ("hello n!help", None),
        ("n!helpme", None),
        ("", None),
        (None, None),
        ("通常のメッセージ", None),
    ],
)
def test_parse_prefix_command(content, expected):
    assert parse_prefix_command(content) == expected


def make_bot(**overrides):
    bot = SimpleNamespace(
        user=SimpleNamespace(id=BOT_ID),
        settings=SimpleNamespace(owner_id=OWNER_ID, model="test-model"),
        answer_authors={},
        server_log=None,
        started_at=time.monotonic(),
        _status_cache=None,
    )
    for key, value in overrides.items():
        setattr(bot, key, value)
    return bot


def make_message(*, content="n!del", author_id=100, reference=None, guild_owner=GUILD_OWNER_ID):
    channel = SimpleNamespace(send=AsyncMock(), fetch_message=AsyncMock())
    return SimpleNamespace(
        id=555,
        content=content,
        author=SimpleNamespace(id=author_id, bot=False, send=AsyncMock()),
        channel=channel,
        reference=reference,
        guild=SimpleNamespace(owner_id=guild_owner) if guild_owner is not None else None,
        add_reaction=AsyncMock(),
    )


def make_target(*, author_id=BOT_ID, delete_error=None):
    target = SimpleNamespace(id=42, author=SimpleNamespace(id=author_id))
    target.delete = AsyncMock(side_effect=delete_error)
    return target


def reference_to(target):
    return SimpleNamespace(message_id=target.id, resolved=target)


def http_error(status=403):
    return discord.HTTPException(
        SimpleNamespace(status=status, reason="error"), "request failed"
    )


def test_help_sends_usage_by_dm_only():
    message = make_message(content="n!help")
    assert asyncio.run(handle_prefix_command(make_bot(), message)) is True
    message.author.send.assert_awaited_once_with(content=HELP_TEXT, embed=None)
    message.channel.send.assert_not_awaited()
    message.add_reaction.assert_awaited_once_with("📩")
    assert "n!del" in HELP_TEXT and "n!status" in HELP_TEXT


def test_help_dm_failure_never_posts_publicly(caplog):
    message = make_message(content="n!help")
    message.author.send = AsyncMock(side_effect=http_error(403))
    with caplog.at_level(logging.WARNING, logger="nezubot_discrod.prefix_commands"):
        assert asyncio.run(handle_prefix_command(make_bot(), message)) is True
    message.channel.send.assert_not_awaited()
    message.add_reaction.assert_awaited_once_with("⚠️")
    assert any("not posting publicly" in record.message for record in caplog.records)


def test_non_command_is_not_handled():
    message = make_message(content="こんにちは")
    assert asyncio.run(handle_prefix_command(make_bot(), message)) is False
    message.channel.send.assert_not_awaited()


def test_del_without_reply_only_guides():
    message = make_message(reference=None)
    asyncio.run(handle_prefix_command(make_bot(), message))
    message.channel.send.assert_awaited_once()
    assert "リプライ" in message.channel.send.call_args.args[0]


def test_del_never_deletes_someone_elses_message():
    target = make_target(author_id=12345)
    message = make_message(reference=reference_to(target))
    asyncio.run(handle_prefix_command(make_bot(), message))
    target.delete.assert_not_awaited()
    message.add_reaction.assert_not_awaited()


def test_del_by_triggering_author():
    target = make_target()
    bot = make_bot(answer_authors={target.id: 100})
    message = make_message(author_id=100, reference=reference_to(target))
    asyncio.run(handle_prefix_command(bot, message))
    target.delete.assert_awaited_once()
    message.add_reaction.assert_awaited_once_with("✅")


def test_del_by_guild_owner_and_bot_owner():
    for author_id in (GUILD_OWNER_ID, OWNER_ID):
        target = make_target()
        message = make_message(author_id=author_id, reference=reference_to(target))
        asyncio.run(handle_prefix_command(make_bot(), message))
        target.delete.assert_awaited_once()


def test_del_rejected_for_unrelated_member():
    target = make_target()
    bot = make_bot(answer_authors={target.id: 100})
    message = make_message(author_id=555, reference=reference_to(target))
    asyncio.run(handle_prefix_command(bot, message))
    target.delete.assert_not_awaited()
    message.add_reaction.assert_not_awaited()
    assert "質問した本人か" in message.channel.send.call_args.args[0]


def test_del_fetches_unresolved_reference():
    target = make_target()
    bot = make_bot(answer_authors={target.id: 100})
    message = make_message(
        author_id=100, reference=SimpleNamespace(message_id=target.id, resolved=None)
    )
    message.channel.fetch_message = AsyncMock(return_value=target)
    asyncio.run(handle_prefix_command(bot, message))
    message.channel.fetch_message.assert_awaited_once_with(target.id)
    target.delete.assert_awaited_once()


def test_del_reports_missing_reference(caplog):
    message = make_message(reference=SimpleNamespace(message_id=42, resolved=None))
    message.channel.fetch_message = AsyncMock(side_effect=http_error(404))
    with caplog.at_level(logging.WARNING, logger="nezubot_discrod.prefix_commands"):
        asyncio.run(handle_prefix_command(make_bot(), message))
    assert "見つからなかった" in message.channel.send.call_args.args[0]
    assert any("could not be fetched" in record.message for record in caplog.records)


def test_del_failure_is_logged_and_reported(caplog):
    target = make_target(delete_error=http_error())
    bot = make_bot(answer_authors={target.id: 100})
    message = make_message(author_id=100, reference=reference_to(target))
    with caplog.at_level(logging.WARNING, logger="nezubot_discrod.prefix_commands"):
        asyncio.run(handle_prefix_command(bot, message))
    assert "削除に失敗" in message.channel.send.call_args.args[0]
    message.add_reaction.assert_not_awaited()
    assert any("failed to delete" in record.message for record in caplog.records)


def test_record_answer_authors_is_bounded():
    bot = make_bot(answer_authors=OrderedDict())
    sent = [SimpleNamespace(id=i) for i in range(1005)]
    record_answer_authors(bot, sent, 100)
    assert len(bot.answer_authors) == 1000
    assert bot.answer_authors[1004] == 100
    assert 0 not in bot.answer_authors


def test_deliver_answer_returns_sent_messages():
    channel = SimpleNamespace(id=7, send=AsyncMock(return_value=SimpleNamespace(id=4242)))
    sent = asyncio.run(deliver_answer(channel, "質問", "短い答え", ["短い答え"]))
    assert [message.id for message in sent] == [4242]


def _write_db(path, rows, vectors):
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, guild_id TEXT, channel_id TEXT, author_id TEXT,
            is_bot INTEGER, deleted INTEGER DEFAULT 0, text TEXT
        );
        CREATE TABLE deleted_messages (id TEXT PRIMARY KEY);
        CREATE TABLE message_vectors (
            message_id TEXT, model TEXT, text_hash TEXT, dimensions INTEGER,
            vector BLOB, PRIMARY KEY (message_id, model)
        );
        """
    )
    db.executemany("INSERT INTO messages VALUES (?, 'g', ?, ?, ?, ?, ?)", rows)
    db.executemany("INSERT INTO message_vectors VALUES (?, ?, 'h', 2, x'00')", vectors)
    db.commit()
    db.close()


def _write_server_wiki(path):
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE server_wiki_entry (guild TEXT, channel TEXT, start INTEGER,"
        " kind TEXT, key TEXT, title TEXT, summary TEXT, sources TEXT)"
    )
    db.executemany(
        "INSERT INTO server_wiki_entry VALUES ('g', 'c', 0, ?, 'k', 't', 's', '[]')",
        [("person",), ("person",), ("event",)],
    )
    db.commit()
    db.close()


@pytest.fixture()
def status_paths(tmp_path):
    db_path = tmp_path / "server_log.db"
    rows = [
        ("m1", "c1", "a1", 0, 0, "hello world"),
        ("m2", "c2", "a1", 0, 0, "foo bar baz"),
        ("m3", "c1", "a2", 0, 0, "x"),
        ("m4", "c1", "a2", 1, 0, "bot message here"),
        ("m5", "c1", "a3", 0, 1, "deleted message"),
        ("m6", "c3", "a3", 0, 0, "tombstoned message"),
    ]
    _write_db(db_path, rows, [("m1", TEACHER), ("m2", STATIC)])
    db = sqlite3.connect(db_path)
    db.execute("INSERT INTO deleted_messages VALUES ('m6')")
    db.commit()
    db.close()
    server_wiki_path = tmp_path / "server_wiki.db"
    _write_server_wiki(server_wiki_path)
    return db_path, server_wiki_path


def test_collect_status_counts(status_paths):
    db_path, server_wiki_path = status_paths
    stats = collect_status(
        db_path, teacher_model=TEACHER, static_model=STATIC, server_wiki_path=server_wiki_path
    )
    assert stats["total_messages"] == 4
    assert stats["people"] == 2
    assert stats["channels"] == 2
    assert stats["eligible"] == 2
    assert stats["teacher_covered"] == 1
    assert stats["static_only"] == 1
    assert stats["server_wiki_people"] == 2
    assert stats["server_wiki_events"] == 1
    assert stats["db_bytes"] >= db_path.stat().st_size


def test_status_embed_shows_rates_and_counts(status_paths):
    db_path, server_wiki_path = status_paths
    stats = collect_status(
        db_path, teacher_model=TEACHER, static_model=STATIC, server_wiki_path=server_wiki_path
    )
    embed = build_status_embed(stats, model="gpt-test", version="1.5.0", uptime_seconds=3661)
    fields = {field.name: field.value for field in embed.fields}
    assert "4 件" in fields["📨 学習メッセージ数"]
    assert "2 人" in fields["👥 学習人数"]
    rate = fields["🎓 学習率（埋め込みカバー率）"]
    assert "50.00%" in rate and "100.00%" in rate
    assert "3 件" in fields["📚 Server Wiki ウィキ記事数"]
    assert "1時間1分" in fields["⏱ 稼働時間"]
    assert "v1.5.0" in embed.description and "gpt-test" in embed.description


def test_status_command_uses_cache(status_paths):
    db_path, server_wiki_path = status_paths
    server_log = SimpleNamespace(
        path=db_path, model=TEACHER, static_model=STATIC
    )
    bot = make_bot(server_log=server_log)
    message = make_message(content="n!status")
    asyncio.run(handle_prefix_command(bot, message))
    first = message.author.send.call_args.kwargs["embed"]
    assert "50.00%" in {
        field.name: field.value for field in first.fields
    }["🎓 学習率（埋め込みカバー率）"]
    message.channel.send.assert_not_awaited()
    db_path.unlink()
    server_wiki_path.unlink()
    message.author.send.reset_mock()
    asyncio.run(handle_prefix_command(bot, message))
    second = message.author.send.call_args.kwargs["embed"]
    assert second.fields[0].value == first.fields[0].value


def test_status_failure_notice_goes_to_dm_only():
    message = make_message(content="n!status")
    asyncio.run(handle_prefix_command(make_bot(server_log=None), message))
    assert "集計に失敗" in message.author.send.call_args.kwargs["content"]
    message.channel.send.assert_not_awaited()


def test_format_helpers():
    assert format_uptime(30) == "30秒"
    assert format_uptime(90) == "1分30秒"
    assert format_uptime(3661) == "1時間1分"
    assert format_uptime(90061) == "1日1時間1分"
    assert format_uptime(None) == "—"
    assert format_size(512) == "512 B"
    assert format_size(3029008384).endswith("GiB")


def test_on_message_routes_prefix_command_without_mention():
    bot = SimpleNamespace(
        settings=Settings(
            "token", "key", "https://example.com", "test",
            SandboxSettings(), guild_id=123, owner_id=456,
        ),
        user=SimpleNamespace(id=BOT_ID),
        assistant=None,
    )
    message = make_message(content="n!help")
    message.guild = SimpleNamespace(id=123, owner_id=GUILD_OWNER_ID)
    message.mentions = []
    asyncio.run(Bot.on_message(bot, message))
    message.author.send.assert_awaited_once_with(content=HELP_TEXT, embed=None)
    message.channel.send.assert_not_awaited()
