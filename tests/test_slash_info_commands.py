import asyncio
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nezubot_discrod.bot import HELP_TEXT, show_help, show_limit, show_status
from nezubot_discrod.feature_loader import BUILTIN_COMMANDS, command_registry

GUILD_ID = 123
TEACHER = "teacher-model"
STATIC = "static-model"

RESPONSE = {
    "rateLimits": {
        "limitId": "codex",
        "planType": "plus",
        "primary": {"usedPercent": 23, "windowDurationMins": 300, "resetsAt": 1764054780},
        "secondary": {"usedPercent": 41, "windowDurationMins": 10080, "resetsAt": 1764480000},
        "credits": None,
    }
}


def make_interaction(guild_id=GUILD_ID):
    return SimpleNamespace(
        guild_id=guild_id,
        user=SimpleNamespace(id=100),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def make_status_bot(db_path):
    return SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID, model="test-model"),
        server_log=SimpleNamespace(path=db_path, model=TEACHER, static_model=STATIC),
        started_at=time.monotonic(),
        _status_cache=None,
    )


def make_limit_bot(rate_limits):
    return SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID, model="test-model"),
        ai_client=SimpleNamespace(rate_limits=rate_limits),
        _limit_cache=None,
    )


def test_show_help_is_ephemeral_and_lists_slash_commands():
    interaction = make_interaction()
    asyncio.run(show_help(interaction, make_limit_bot(None)))
    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    assert kwargs["ephemeral"] is True
    assert args[0] == HELP_TEXT
    for needle in ("/help", "/status", "/limit", "n!del", "本人だけ"):
        assert needle in HELP_TEXT
    interaction.followup.send.assert_not_awaited()


def test_show_help_wrong_guild_refuses_ephemerally():
    interaction = make_interaction(guild_id=999)
    asyncio.run(show_help(interaction, make_limit_bot(None)))
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True
    assert "指定サーバー" in interaction.response.send_message.call_args.args[0]
    interaction.followup.send.assert_not_awaited()


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
        [("person",), ("event",)],
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
    ]
    _write_db(db_path, rows, [("m1", TEACHER), ("m2", STATIC)])
    server_wiki_path = tmp_path / "server_wiki.db"
    _write_server_wiki(server_wiki_path)
    return db_path, server_wiki_path


def test_show_status_defers_then_sends_embed_ephemerally(status_paths):
    db_path, server_wiki_path = status_paths
    bot = make_status_bot(db_path)
    interaction = make_interaction()
    asyncio.run(show_status(interaction, bot))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()
    kwargs = interaction.followup.send.call_args.kwargs
    assert kwargs["ephemeral"] is True
    embed = kwargs["embed"]
    fields = {field.name: field.value for field in embed.fields}
    assert "50.00%" in fields["🎓 学習率（埋め込みカバー率）"]
    # Second call is served from the cache even with the DB gone.
    db_path.unlink()
    server_wiki_path.unlink()
    interaction2 = make_interaction()
    asyncio.run(show_status(interaction2, bot))
    embed2 = interaction2.followup.send.call_args.kwargs["embed"]
    assert embed2.fields[0].value == embed.fields[0].value


def test_show_status_failure_notice_is_ephemeral():
    bot = make_status_bot(None)
    bot.server_log = None
    interaction = make_interaction()
    asyncio.run(show_status(interaction, bot))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()
    args, kwargs = interaction.followup.send.call_args
    assert kwargs["ephemeral"] is True
    assert "集計に失敗" in args[0]


def test_show_status_wrong_guild_never_defers():
    interaction = make_interaction(guild_id=999)
    asyncio.run(show_status(interaction, make_status_bot(None)))
    interaction.response.send_message.assert_awaited_once()
    interaction.response.defer.assert_not_awaited()
    interaction.followup.send.assert_not_awaited()


def test_show_limit_defers_then_sends_embed_ephemerally_and_caches():
    fetch = AsyncMock(return_value=RESPONSE)
    bot = make_limit_bot(fetch)
    interaction = make_interaction()
    asyncio.run(show_limit(interaction, bot))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()
    kwargs = interaction.followup.send.call_args.kwargs
    assert kwargs["ephemeral"] is True
    assert "残り 77%" in kwargs["embed"].fields[0].value
    interaction2 = make_interaction()
    asyncio.run(show_limit(interaction2, bot))
    fetch.assert_awaited_once()


def test_show_limit_failure_is_ephemeral():
    bot = make_limit_bot(AsyncMock(side_effect=RuntimeError("app-server down")))
    interaction = make_interaction()
    asyncio.run(show_limit(interaction, bot))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    args, kwargs = interaction.followup.send.call_args
    assert kwargs["ephemeral"] is True
    assert "取得できませんでした" in args[0]


def test_show_limit_without_codex_client_is_ephemeral_failure():
    bot = make_limit_bot(None)
    bot.ai_client = SimpleNamespace()
    interaction = make_interaction()
    asyncio.run(show_limit(interaction, bot))
    args, kwargs = interaction.followup.send.call_args
    assert kwargs["ephemeral"] is True
    assert "取得できませんでした" in args[0]


def test_show_limit_wrong_guild_never_defers():
    interaction = make_interaction(guild_id=999)
    asyncio.run(show_limit(interaction, make_limit_bot(AsyncMock(return_value=RESPONSE))))
    interaction.response.send_message.assert_awaited_once()
    interaction.response.defer.assert_not_awaited()
    interaction.followup.send.assert_not_awaited()


def test_builtin_commands_reserve_info_names():
    assert {"help", "status", "limit"} <= BUILTIN_COMMANDS


async def _dummy_callback(interaction):
    pass


@pytest.mark.parametrize("name", ["help", "status", "limit"])
def test_feature_registry_rejects_info_names(name):
    tree = SimpleNamespace(get_commands=lambda guild=None: [])
    registry, _pending = command_registry(tree, GUILD_ID, 0)
    with pytest.raises(ValueError):
        registry.register(name=name, description="x", callback=_dummy_callback)


def test_cached_status_serves_stale_and_refreshes_in_background(tmp_path, monkeypatch):
    import nezubot_discrod.bot as bot_module
    from nezubot_discrod.bot import cached_status

    calls = {"count": 0}

    def fake_collect(path, *, teacher_model, static_model=None, server_wiki_path=None):
        calls["count"] += 1
        return {"total_messages": calls["count"]}

    monkeypatch.setattr(bot_module, "collect_status", fake_collect)
    bot = make_status_bot(tmp_path / "unused.db")
    bot._status_refreshing = False

    async def run():
        first = await cached_status(bot)
        assert first["total_messages"] == 1 and calls["count"] == 1
        # Fresh cache: no refetch.
        assert (await cached_status(bot))["total_messages"] == 1
        assert calls["count"] == 1
        # Stale cache: the old value comes back immediately and exactly
        # one background refresh is scheduled.
        bot._status_cache = (time.monotonic() - 10000, {"total_messages": 1})
        stale = await cached_status(bot)
        assert stale["total_messages"] == 1
        await asyncio.sleep(0.1)
        assert calls["count"] == 2
        assert bot._status_cache[1]["total_messages"] == 2
        assert bot._status_refreshing is False

    asyncio.run(run())
