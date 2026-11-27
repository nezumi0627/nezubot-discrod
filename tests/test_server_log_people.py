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

# Synthetic snowflakes (never a real member's id): PERSON_A was
# composed so its creation instant is exactly 2024-03-01T12:00:00Z,
# PERSON_B's is 2019-06-15T08:30:00Z. The join / boost instants used
# below are likewise invented, in the same relative shape the member
# record takes (join minutes after creation).
PERSON_A = "1213093380096424242"
PERSON_A_CREATED = "2024-03-01T12:00:00Z"
PERSON_A_JOINED = "2024-03-01T12:08:00.000000+00:00"
PERSON_B = "589371015168000777"
PERSON_B_CREATED = "2019-06-15T08:30:00Z"
PERSON_B_JOINED = "2019-06-16T10:00:00.000000+00:00"
BOOST_SINCE = "2024-05-06T07:08:09.000000+00:00"


def test_snowflake_created_at_derives_the_account_creation_instant():
    assert snowflake_created_at(PERSON_A) == PERSON_A_CREATED
    assert snowflake_created_at(int(PERSON_A)) == PERSON_A_CREATED
    assert snowflake_created_at(PERSON_B) == PERSON_B_CREATED


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
        "user": {"id": PERSON_A, "username": "fakea",
                 "global_name": "Fake A", "avatar": None},
        "nick": "",
        "joined_at": PERSON_A_JOINED,
        "premium_since": BOOST_SINCE,
        "roles": ["111"],
    }]
    backfill_members(store, members, "100", now="2026-11-27T00:00:00Z")
    with store.connect() as db:
        row = db.execute("SELECT * FROM users").fetchone()
    assert row["joined_at"] == PERSON_A_JOINED
    assert row["created_at"] == PERSON_A_CREATED
    assert row["premium_since"] == BOOST_SINCE
    assert row["roles"] == '["111"]'
    assert row["synced_at"] == "2026-11-27T00:00:00Z"


def test_sync_users_records_member_facts(store):
    person = PERSON_A
    for number in range(1, 4):
        store.upsert(record(number, author_id=person, author="fakea",
                            text=f"msg {number}"))

    def fetch(guild_id, user_id):
        return {
            "user": {"username": "fakea", "global_name": "Fake A",
                     "avatar": None},
            "nick": "",
            "joined_at": PERSON_A_JOINED,
            "premium_since": BOOST_SINCE,
            "roles": ["111", "222"],
        }

    result = sync_users(store, fetch, min_messages=3, pause=0,
                        now="2026-11-27T00:00:00Z")
    assert result == {"targets": 1, "synced": 1, "fetched": 1}
    with store.connect() as db:
        row = db.execute(
            "SELECT * FROM users WHERE user_id = ?", (person,)).fetchone()
    assert row["joined_at"] == PERSON_A_JOINED
    assert row["created_at"] == PERSON_A_CREATED
    assert row["premium_since"] == BOOST_SINCE
    assert row["roles"] == '["111", "222"]'


def test_backfill_members_upserts_a_full_listing(store):
    members = [
        {
            "user": {"id": PERSON_A, "username": "fakea",
                     "global_name": "Fake A", "avatar": "abc"},
            "nick": "ニック",
            "joined_at": PERSON_A_JOINED,
            "premium_since": None,
            "roles": ["111"],
        },
        {
            "user": {"id": PERSON_B, "username": "fakeb",
                     "global_name": "Fake B", "avatar": None},
            "nick": "",
            "joined_at": PERSON_B_JOINED,
            "premium_since": BOOST_SINCE,
            "roles": [],
        },
        {"user": {}, "joined_at": "2026-01-01T00:00:00Z"},  # no id: skipped
    ]
    result = backfill_members(store, members, "100", now="2026-11-27T00:00:00Z")
    assert result == {"backfilled": 2, "departed": 0}
    with store.connect() as db:
        rows = {row["user_id"]: row for row in db.execute("SELECT * FROM users")}
    assert set(rows) == {PERSON_A, PERSON_B}
    fakea = rows[PERSON_A]
    assert fakea["joined_at"] == PERSON_A_JOINED
    assert fakea["created_at"] == PERSON_A_CREATED
    assert fakea["premium_since"] == ""  # null in the payload, stored empty
    assert fakea["roles"] == '["111"]'
    fakeb = rows[PERSON_B]
    assert fakeb["created_at"] == PERSON_B_CREATED
    assert fakeb["premium_since"] == BOOST_SINCE
