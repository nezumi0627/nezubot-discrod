import pytest

from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_people import (
    build_people,
    mentioned_user_ids,
    sync_users,
)


def record(number, *, author_id="20", author="alice", text="hello", reply_to=None):
    value = {
        "id": str(number),
        "guild_id": "100",
        "channel_id": "10",
        "_channel": "general",
        "author": {"id": author_id, "username": author, "bot": False},
        "timestamp": f"2026-01-01T00:{number % 60:02d}:00+00:00",
        "content": text,
    }
    if reply_to is not None:
        value["message_reference"] = {"message_id": str(reply_to)}
    return value


@pytest.fixture
def store(tmp_path):
    return ServerLog(tmp_path / "data" / "server_log.db")


def people(store):
    with store.connect() as db:
        return {
            (row["message_id"], row["user_id"], row["relation"])
            for row in db.execute("SELECT * FROM message_people")
        }


def test_mentioned_user_ids_markup_only():
    assert mentioned_user_ids("<@123> <@!456> <@&789> <#111> <@123>") == {"123", "456"}
    assert mentioned_user_ids("no mentions") == set()
    assert mentioned_user_ids(None) == set()


def test_upsert_hook_tracks_author_mention_reply(store):
    store.upsert(record(1, text="hi <@21> and <@!22>, not <@&99> or <#10>"))
    store.upsert(record(2, author_id="21", author="bob", text="re: hi", reply_to=1))
    assert people(store) == {
        ("1", "20", "author"),
        ("1", "21", "mention"),
        ("1", "22", "mention"),
        ("2", "21", "author"),
        ("2", "20", "reply"),
    }


def test_upsert_hook_follows_edits_and_delete(store):
    store.upsert(record(1, text="ping <@21>"))
    edited = record(1, text="ping <@22>")
    edited["edited_timestamp"] = "2026-01-02T00:00:00Z"
    store.upsert(edited)
    assert ("1", "21", "mention") not in people(store)
    assert ("1", "22", "mention") in people(store)
    store.mark_deleted("1")
    assert people(store) == set()


def test_build_people_idempotent_and_skips_missing_reply_target(store):
    store.upsert(record(1, text="hey <@21>"))
    store.upsert(record(2, author_id="21", author="bob", text="yo", reply_to=1))
    store.upsert(record(3, author_id="21", author="bob", text="orphan", reply_to=999))
    with store.connect() as db:
        db.execute("DELETE FROM message_people")
    first = build_people(store, chunk_rows=2, pause=0)
    second = build_people(store, chunk_rows=2, pause=0)
    assert first == second
    assert first["author"] == 3 and first["mention"] == 1 and first["reply"] == 1
    assert ("3", "20", "reply") not in people(store)


def test_sync_users_fetched_and_fallback(store):
    for number in range(1, 4):
        store.upsert(record(number, text=f"msg {number}"))
        store.upsert(record(100 + number, author_id="21", author="bob", text=f"bob {number}"))

    def fetch(guild_id, user_id):
        assert guild_id == "100"
        if user_id == "20":
            return {
                "user": {"username": "alice_u", "global_name": "Alice G", "avatar": "abc"},
                "nick": "ally",
                "joined_at": "2025-05-01T00:00:00Z",
            }
        return None

    result = sync_users(store, fetch, min_messages=3, pause=0, now="2026-01-03T00:00:00Z")
    assert result == {"targets": 2, "synced": 2, "fetched": 1}
    with store.connect() as db:
        rows = {row["user_id"]: row for row in db.execute("SELECT * FROM users")}
    alice = rows["20"]
    assert alice["username"] == "alice_u" and alice["display_name"] == "Alice G"
    assert alice["nickname"] == "ally"
    assert alice["avatar_url"] == "https://cdn.discord.com/avatars/20/abc.png?size=256"
    assert alice["joined_at"] == "2025-05-01T00:00:00Z"
    assert alice["synced_at"] == "2026-01-03T00:00:00Z"
    bob = rows["21"]
    assert bob["username"] == "bob" and bob["avatar_url"] == "" and bob["joined_at"] == ""
