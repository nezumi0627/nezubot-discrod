import pytest

from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_people import (
    backfill_members,
    build_people,
    ensure_people_schema,
    mentioned_user_ids,
    snowflake_created_at,
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


# --- v1.38.0: member-record facts (created_at / premium_since / roles) ----

def test_snowflake_created_at_derives_the_account_creation_instant():
    # A full-length snowflake (the placeholder owner id): the
    # derived creation instant is 2016-09-05T05:11:46Z.
    assert snowflake_created_at("222222222222222222") == "2016-09-05T05:11:46Z"
    assert snowflake_created_at(222222222222222222) == "2016-09-05T05:11:46Z"


def test_snowflake_created_at_rejects_non_snowflakes():
    assert snowflake_created_at("30") == ""
    assert snowflake_created_at("not-an-id") == ""
    assert snowflake_created_at(None) == ""
    assert snowflake_created_at("") == ""


def test_ensure_people_schema_migrates_the_old_users_table(store):
    with store.connect() as db:
        db.execute("DROP TABLE users")
        db.execute(
            "CREATE TABLE users ("
            "user_id TEXT PRIMARY KEY, guild_id TEXT NOT NULL, "
            "username TEXT NOT NULL DEFAULT '', "
            "display_name TEXT NOT NULL DEFAULT '', "
            "nickname TEXT NOT NULL DEFAULT '', "
            "avatar_url TEXT NOT NULL DEFAULT '', "
            "joined_at TEXT NOT NULL DEFAULT '', "
            "synced_at TEXT NOT NULL DEFAULT '')"
        )
        db.execute("INSERT INTO users (user_id, guild_id) VALUES ('20', '100')")
        ensure_people_schema(db)
        ensure_people_schema(db)  # idempotent
        columns = {row[1] for row in db.execute("PRAGMA table_info(users)")}
        row = db.execute("SELECT * FROM users WHERE user_id = '20'").fetchone()
    assert {"created_at", "premium_since", "roles"} <= columns
    assert row["created_at"] == "" and row["premium_since"] == ""
    assert row["roles"] == ""


def test_backfill_into_a_migrated_table_lands_values_in_the_right_columns(store):
    # Regression: on a migrated table the added columns sit at the
    # end, so the upsert must name its columns — a positional INSERT
    # put roles into premium_since and the timestamp into roles when
    # the v1.38.0 backfill first ran against the production table.
    with store.connect() as db:
        db.execute("DROP TABLE users")
        db.execute(
            "CREATE TABLE users ("
            "user_id TEXT PRIMARY KEY, guild_id TEXT NOT NULL, "
            "username TEXT NOT NULL DEFAULT '', "
            "display_name TEXT NOT NULL DEFAULT '', "
            "nickname TEXT NOT NULL DEFAULT '', "
            "avatar_url TEXT NOT NULL DEFAULT '', "
            "joined_at TEXT NOT NULL DEFAULT '', "
            "synced_at TEXT NOT NULL DEFAULT '')"
        )
        ensure_people_schema(db)
    members = [{
        "user": {"id": "222222222222222222", "username": "testuser0627",
                 "global_name": "テストユーザー", "avatar": None},
        "nick": "",
        "joined_at": "2024-04-10T08:00:00.000000+00:00",
        "premium_since": "2024-05-01T00:00:00.000000+00:00",
        "roles": ["111"],
    }]
    backfill_members(store, members, "100", now="2026-11-27T00:00:00Z")
    with store.connect() as db:
        row = db.execute("SELECT * FROM users").fetchone()
    assert row["joined_at"] == "2024-04-10T08:00:00.000000+00:00"
    assert row["created_at"] == "2016-09-05T05:11:46Z"
    assert row["premium_since"] == "2024-05-01T00:00:00.000000+00:00"
    assert row["roles"] == '["111"]'
    assert row["synced_at"] == "2026-11-27T00:00:00Z"


def test_sync_users_records_member_facts(store):
    person = "222222222222222222"
    for number in range(1, 4):
        store.upsert(record(number, author_id=person, author="testuser0627",
                            text=f"msg {number}"))

    def fetch(guild_id, user_id):
        return {
            "user": {"username": "testuser0627", "global_name": "テストユーザー", "avatar": None},
            "nick": "",
            "joined_at": "2024-04-10T08:00:00.000000+00:00",
            "premium_since": "2024-05-01T00:00:00.000000+00:00",
            "roles": ["111", "222"],
        }

    result = sync_users(store, fetch, min_messages=3, pause=0,
                        now="2026-11-27T00:00:00Z")
    assert result == {"targets": 1, "synced": 1, "fetched": 1}
    with store.connect() as db:
        row = db.execute(
            "SELECT * FROM users WHERE user_id = ?", (person,)).fetchone()
    assert row["joined_at"] == "2024-04-10T08:00:00.000000+00:00"
    assert row["created_at"] == "2016-09-05T05:11:46Z"
    assert row["premium_since"] == "2024-05-01T00:00:00.000000+00:00"
    assert row["roles"] == '["111", "222"]'


def test_backfill_members_upserts_a_full_listing(store):
    members = [
        {
            "user": {"id": "222222222222222222", "username": "testuser0627",
                     "global_name": "テストユーザー", "avatar": "abc"},
            "nick": "テストユーザー",
            "joined_at": "2024-04-10T08:00:00.000000+00:00",
            "premium_since": None,
            "roles": ["111"],
        },
        {
            "user": {"id": "123456789012345678", "username": "tanuki_dev",
                     "global_name": "タヌキ", "avatar": None},
            "nick": "",
            "joined_at": "2024-03-01T12:00:00.000000+00:00",
            "premium_since": "2024-05-01T00:00:00.000000+00:00",
            "roles": [],
        },
        {"user": {}, "joined_at": "2026-01-01T00:00:00Z"},  # no id: skipped
    ]
    result = backfill_members(store, members, "100", now="2026-11-27T00:00:00Z")
    assert result == {"backfilled": 2}
    with store.connect() as db:
        rows = {row["user_id"]: row for row in db.execute("SELECT * FROM users")}
    assert set(rows) == {"222222222222222222", "123456789012345678"}
    owner_row = rows["222222222222222222"]
    assert owner_row["joined_at"] == "2024-04-10T08:00:00.000000+00:00"
    assert owner_row["created_at"] == "2016-09-05T05:11:46Z"
    assert owner_row["premium_since"] == ""  # null in the payload, stored empty
    assert owner_row["roles"] == '["111"]'
    other_row = rows["123456789012345678"]
    assert other_row["created_at"] == "2015-12-07T16:13:12Z"
    assert other_row["premium_since"] == "2024-05-01T00:00:00.000000+00:00"
