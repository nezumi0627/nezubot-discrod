import asyncio
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nezubot_discrod.bot import show_rank
from nezubot_discrod.ranking import (
    build_rank_embed,
    collect_ranking,
    write_ranking_snapshot,
)

GUILD_ID = 123

_SCHEMA = """
CREATE TABLE messages (
    id TEXT PRIMARY KEY, guild_id TEXT NOT NULL, channel_id TEXT NOT NULL,
    channel_name TEXT NOT NULL, parent_name TEXT, author_id TEXT NOT NULL,
    author_name TEXT NOT NULL, is_bot INTEGER NOT NULL,
    created_at TEXT NOT NULL, edited_at TEXT, reply_to TEXT,
    text TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
    author_username TEXT NOT NULL DEFAULT '',
    author_global_name TEXT NOT NULL DEFAULT '',
    author_nickname TEXT NOT NULL DEFAULT ''
);
CREATE TABLE users (
    user_id TEXT PRIMARY KEY, guild_id TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '', display_name TEXT NOT NULL DEFAULT '',
    nickname TEXT NOT NULL DEFAULT '', avatar_url TEXT NOT NULL DEFAULT '',
    joined_at TEXT NOT NULL DEFAULT '', synced_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE deleted_messages (id TEXT PRIMARY KEY);
"""


def _insert(db, msg_id, author_id, created_at, text, *, is_bot=0, deleted=0,
            author_name="", author_global_name="", author_nickname=""):
    db.execute(
        "INSERT INTO messages (id, guild_id, channel_id, channel_name, "
        "author_id, author_name, is_bot, created_at, text, deleted, "
        "author_global_name, author_nickname) "
        "VALUES (?, '1', '10', 'general', ?, ?, ?, ?, ?, ?, ?, ?)",
        (msg_id, author_id, author_name, is_bot, created_at, text, deleted,
         author_global_name, author_nickname),
    )


def make_db(tmp_path):
    """Fixture archive.

    alice (111) is #1 with 5 surviving messages; a deleted message
    (500 chars) and a tombstoned message (400 chars) must not count
    toward her total or her longest message. The bot author (999)
    out-posts everyone but is excluded. bob..frank fill ranks 2-6,
    with erin/frank tied at 1 so the author-id tiebreak decides 5th.
    """
    path = tmp_path / "server_log.db"
    with closing(sqlite3.connect(path)) as db:
        db.executescript(_SCHEMA)
        db.execute(
            "INSERT INTO users VALUES ('111', '1', 'alice_u', 'Alice D', "
            "'ありす', '', '', '')"
        )
        db.execute(
            "INSERT INTO users VALUES ('333', '1', 'carol_u', 'Carol D', "
            "'', '', '', '')"
        )
        # alice: JST hours 9, 21, 22, 22, 23 across Jan 1 and Jan 3.
        _insert(db, "a1", "111", "2025-01-01T00:00:00+00:00", "a")
        _insert(db, "a2", "111", "2025-01-03T12:00:00+00:00", "hello world")
        _insert(db, "a3", "111", "2025-01-03T13:00:00+00:00", "x")
        _insert(db, "a4", "111", "2025-01-03T13:30:00+00:00", "z")
        _insert(db, "a5", "111", "2025-01-03T14:00:00+00:00", "y")
        _insert(db, "a6", "111", "2025-01-02T00:00:00+00:00", "d" * 500,
                deleted=1)
        _insert(db, "a7", "111", "2025-01-02T01:00:00+00:00", "t" * 400)
        db.execute("INSERT INTO deleted_messages VALUES ('a7')")
        # bob: no users row, so the name comes from the message snapshot.
        for i in range(4):
            _insert(db, f"b{i}", "222", f"2025-01-0{i + 1}T00:00:00+00:00",
                    "b", author_name="bob_u", author_global_name="Bob G")
        for i in range(3):
            _insert(db, f"c{i}", "333", f"2025-01-0{i + 1}T00:00:00+00:00", "c")
        for i in range(2):
            _insert(db, f"d{i}", "444", f"2025-01-0{i + 1}T00:00:00+00:00", "d")
        _insert(db, "e0", "555", "2025-01-01T00:00:00+00:00", "e")
        _insert(db, "f0", "666", "2025-01-01T00:00:00+00:00", "f")
        for i in range(10):
            _insert(db, f"bot{i}", "999", f"2025-01-0{i % 9 + 1}T00:00:00+00:00",
                    "beep", is_bot=1)
        db.commit()
    return path


def test_ranking_order_counts_and_bot_exclusion(tmp_path):
    ranking = collect_ranking(make_db(tmp_path))
    entries = ranking["entries"]
    assert [e["author_id"] for e in entries] == ["111", "222", "333", "444", "555"]
    assert [e["count"] for e in entries] == [5, 4, 3, 2, 1]
    assert entries[0]["name"] == "ありす"  # users.nickname wins
    assert entries[1]["name"] == "Bob G"  # message global-name fallback
    assert entries[2]["name"] == "Carol D"  # users.display_name


def test_top_detail_average_longest_and_hours(tmp_path):
    detail = collect_ranking(make_db(tmp_path))["detail"]
    assert detail["author_id"] == "111"
    assert detail["count"] == 5
    # Jan 1 -> Jan 3 JST is 2 calendar days; 5 / 2 = 2.5.
    assert detail["per_day"] == 2.5
    # "hello world"; the deleted 500-char and tombstoned 400-char
    # messages must not win.
    assert detail["max_chars"] == 11
    # JST buckets: 22h twice, then 9h and 21h once each (hour asc on ties).
    assert detail["top_hours"] == [22, 9, 21]
    assert detail["last_at"] == "2025-01-03T14:00:00+00:00"


def test_snapshot_is_preferred_over_live_data(tmp_path):
    path = make_db(tmp_path)
    written = write_ranking_snapshot(path)
    assert written["entries"][0]["count"] == 5
    with closing(sqlite3.connect(path)) as db:
        for i in range(50):
            _insert(db, f"b-extra{i}", "222", "2025-02-01T00:00:00+00:00", "b")
        db.commit()
    ranking = collect_ranking(path)
    assert ranking["entries"][0]["author_id"] == "111"
    assert ranking["entries"][0]["count"] == 5


def test_empty_archive_returns_none(tmp_path):
    path = tmp_path / "server_log.db"
    with closing(sqlite3.connect(path)) as db:
        db.executescript(_SCHEMA)
        db.commit()
    assert collect_ranking(path) is None
    embed = build_rank_embed(None)
    assert "まだ集計できるメッセージがありません" in embed.description


def test_build_rank_embed_contents(tmp_path):
    embed = build_rank_embed(collect_ranking(make_db(tmp_path)))
    assert embed.title == "📊 メッセージ数ランキング（全期間）"
    fields = {field.name: field.value for field in embed.fields}
    assert "ありす" in fields["トップ5"]
    assert "5 件" in fields["トップ5"]
    detail = fields["🥇 1位の詳細: ありす"]
    assert "2025年1月3日 23:00" in detail  # last message, JST
    assert "1日平均: 2.5 件" in detail
    assert "最長メッセージ: 11 文字" in detail
    assert "22時台" in detail


def test_show_rank_sends_ephemeral_embed(tmp_path):
    path = make_db(tmp_path)
    interaction = SimpleNamespace(
        guild_id=GUILD_ID,
        user=SimpleNamespace(id=100),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    bot = SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID),
        server_log=SimpleNamespace(path=path),
    )
    asyncio.run(show_rank(interaction, bot))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    _args, kwargs = interaction.followup.send.call_args
    assert kwargs["ephemeral"] is True
    assert kwargs["embed"].title == "📊 メッセージ数ランキング（全期間）"
