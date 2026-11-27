"""v1.39.0 regression tests: member-info enhancement.

Four owner-directed additions on top of the v1.38.0 member block:

1. Role names: the recorded role ids are resolved against a synced
   guild_roles cache and presented by name (@everyone excluded,
   unresolvable ids omitted, never guessed).
2. The boost start (recorded since v1.38.0) stays presented.
3. Server-wide join-date questions (「最近入ったのは誰？」「参加が
   古い順」「2026年に入った人は？」) get a deterministic roster block
   ordered by the recorded join date, capped, present members only.
4. Departed members are distinguished: a full member listing marks
   who it saw (in_guild 1) and who it did not (0); the block states
   a departed person's record as a pre-departure snapshot, and an
   unverified presence as unverified.

All fixtures are synthetic: invented ids, invented dates, invented
names. No real member's id, date or handle fragment appears here
(the v1.38.0 people tests carried real fixtures that the public
export then had to neutralize; that is not repeated).
"""
import asyncio
from datetime import UTC, datetime

import pytest
from test_server_log import record

from nezubot_discrod.prompts import LOG_GUIDANCE
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import (
    ROUTE_BUDGETS_MS,
    _assess_sufficiency,
    _member_roster,
    _member_roster_signals,
    _person_member_info,
    pre_retrieve,
)
from nezubot_discrod.server_log_people import (
    backfill_members,
    role_names,
    sync_roles,
    sync_users,
)
from nezubot_discrod.server_log_tools import LogResult, LogTools

CALLER = "999"
GUILD = "100"

# A synthetic snowflake whose creation instant is 2025-05-20T09:12:00Z
# (the same construction the v1.38.0 tests use).
_EPOCH_MS = 1420070400000
_CREATED_MS = int(datetime(2025, 5, 20, 9, 12, tzinfo=UTC).timestamp() * 1000)
PERSON = str(((_CREATED_MS - _EPOCH_MS) << 22) + 777)
JOINED_AT = "2025-05-21T00:00:00.000000+00:00"   # 2025年5月21日 09:00 JST
PREMIUM_SINCE = "2026-01-02T03:04:05+00:00"      # 2026年1月2日 12:04 JST

ROLE_EVERYONE = GUILD  # the @everyone role id IS the guild id
ROLE_ADMIN = "700001"
ROLE_MEMBER = "700002"
ROLE_UNKNOWN = "700099"

FAKE_ROLES = [
    {"id": ROLE_EVERYONE, "name": "@everyone"},
    {"id": ROLE_ADMIN, "name": "管理人"},
    {"id": ROLE_MEMBER, "name": "常連"},
]


def _insert_user(store, user_id, *, joined_at=JOINED_AT, premium_since="",
                 roles="[]", in_guild=1, guild_id=GUILD,
                 username="memberfake", display_name="メンバー"):
    with store.connect() as db:
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, display_name, "
            "nickname, joined_at, premium_since, roles, in_guild) VALUES "
            "(?, ?, ?, ?, '', ?, ?, ?, ?)",
            (user_id, guild_id, username, display_name, joined_at,
             premium_since, roles, in_guild),
        )


@pytest.fixture
def member_store(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id=GUILD)
    return store


# --- role cache (people layer) ----------------------------------------------

def test_sync_roles_replaces_the_cache(member_store):
    result = sync_roles(member_store, lambda guild_id: FAKE_ROLES, GUILD)
    assert result == {"roles": 3, "fetched": True}
    assert role_names(member_store, GUILD) == {
        ROLE_EVERYONE: "@everyone", ROLE_ADMIN: "管理人", ROLE_MEMBER: "常連",
    }
    # A second sync replaces wholesale: a renamed role lands, a
    # deleted one disappears.
    renamed = [{"id": ROLE_ADMIN, "name": "運営"}]
    sync_roles(member_store, lambda guild_id: renamed, GUILD)
    assert role_names(member_store, GUILD) == {ROLE_ADMIN: "運営"}


def test_sync_roles_keeps_the_cache_when_the_fetch_fails(member_store):
    sync_roles(member_store, lambda guild_id: FAKE_ROLES, GUILD)
    result = sync_roles(member_store, lambda guild_id: None, GUILD)
    assert result == {"roles": 0, "fetched": False}
    assert role_names(member_store, GUILD)[ROLE_ADMIN] == "管理人"


def test_sync_users_rides_the_role_sync(member_store):
    for number in range(1, 4):
        value = record(number, channel="10", text=f"msg {number}")
        value["author"] = {"id": PERSON, "username": "memberfake", "bot": False}
        member_store.upsert(value)

    def fetch(guild_id, user_id):
        return {"user": {"username": "memberfake"}, "joined_at": JOINED_AT}

    result = sync_users(
        member_store, fetch, min_messages=3, pause=0,
        now="2026-11-27T00:00:00Z", fetch_roles=lambda guild_id: FAKE_ROLES)
    assert result["roles"] == {GUILD: {"roles": 3, "fetched": True}}
    assert role_names(member_store, GUILD)[ROLE_MEMBER] == "常連"


# --- presence marking (people layer) ------------------------------------------

def _member_payload(user_id, username):
    return {
        "user": {"id": user_id, "username": username,
                 "global_name": username, "avatar": None},
        "nick": "", "joined_at": JOINED_AT, "premium_since": None,
        "roles": [],
    }


def test_backfill_marks_listed_present_and_unlisted_departed(member_store):
    _insert_user(member_store, "800001", username="stale", in_guild=None)
    _insert_user(member_store, "800002", username="otherguild",
                 in_guild=None, guild_id="200")
    result = backfill_members(
        member_store, [_member_payload(PERSON, "memberfake")], GUILD,
        now="2026-11-27T00:00:00Z")
    assert result == {"backfilled": 1, "departed": 1}
    with member_store.connect() as db:
        rows = {row["user_id"]: row["in_guild"]
                for row in db.execute("SELECT * FROM users")}
    assert rows[PERSON] == 1
    assert rows["800001"] == 0       # same guild, not listed: departed
    assert rows["800002"] is None    # another guild: untouched


def test_sync_users_marks_presence_only_on_a_successful_fetch(member_store):
    for number in range(1, 4):
        value = record(number, channel="10", text=f"a {number}")
        value["author"] = {"id": PERSON, "username": "memberfake", "bot": False}
        member_store.upsert(value)
        value = record(100 + number, channel="10", text=f"b {number}")
        value["author"] = {"id": "800003", "username": "ghostfake", "bot": False}
        value["timestamp"] = "2026-01-01T00:00:00+00:00"
        member_store.upsert(value)

    def fetch(guild_id, user_id):
        if user_id == PERSON:
            return {"user": {"username": "memberfake"}, "joined_at": JOINED_AT}
        return None  # an API failure looks the same: proves nothing

    sync_users(member_store, fetch, min_messages=3, pause=0,
               now="2026-11-27T00:00:00Z")
    with member_store.connect() as db:
        rows = {row["user_id"]: row["in_guild"]
                for row in db.execute("SELECT * FROM users")}
    assert rows[PERSON] == 1
    assert rows["800003"] is None


# --- the person block: roles, boost, presence --------------------------------

def test_member_info_block_names_current_roles_and_boost(member_store):
    sync_roles(member_store, lambda guild_id: FAKE_ROLES, GUILD)
    _insert_user(
        member_store, PERSON, premium_since=PREMIUM_SINCE,
        roles=f'["{ROLE_EVERYONE}", "{ROLE_ADMIN}", "{ROLE_UNKNOWN}", '
              f'"{ROLE_MEMBER}"]')
    text = str(_person_member_info(member_store, PERSON))
    # @everyone is dropped, the unresolvable id is dropped (never
    # guessed), the resolvable names are presented in record order.
    assert "- 現在のロール: 管理人、常連" in text
    assert ROLE_UNKNOWN not in text
    assert "- ブースト開始日: 2026年1月2日 12:04（JST）" in text
    assert "在籍状況" not in text  # verified present: no presence line


def test_member_info_block_omits_roles_without_a_cache(member_store):
    _insert_user(member_store, PERSON, roles=f'["{ROLE_ADMIN}"]')
    text = str(_person_member_info(member_store, PERSON))
    assert "現在のロール" not in text


def test_member_info_block_marks_a_departed_member(member_store):
    sync_roles(member_store, lambda guild_id: FAKE_ROLES, GUILD)
    _insert_user(member_store, PERSON, roles=f'["{ROLE_ADMIN}"]', in_guild=0)
    text = str(_person_member_info(member_store, PERSON))
    assert "- 退会前のロール（記録）: 管理人" in text
    assert "在籍状況: 現在はサーバーに在籍していない" in text
    assert "退会前の記録" in text
    # The recorded join date itself still stands as a fact.
    assert "サーバー参加日: 2025年5月21日 9:00（JST）" in text


def test_member_info_block_marks_unverified_presence(member_store):
    _insert_user(member_store, PERSON, in_guild=None)
    text = str(_person_member_info(member_store, PERSON))
    assert "在籍状況: 未確認" in text
    assert "断定しない" in text


# --- roster signals -----------------------------------------------------------

def test_roster_signals_recognize_roster_shapes():
    assert _member_roster_signals("最近入ったのは誰？") == (True, "desc", None)
    topic, order, _ = _member_roster_signals("参加が古い順に教えて")
    assert (topic, order) == (True, "asc")
    topic, _, year = _member_roster_signals("2026年に入った人は？")
    assert (topic, year) == (True, 2026)
    topic, order, _ = _member_roster_signals("古参を教えて")
    assert (topic, order) == (True, "asc")


def test_roster_signals_ignore_person_and_unrelated_questions():
    assert _member_roster_signals("<@123> のサーバー参加日はいつ？")[0] is False
    assert _member_roster_signals("最近入ったサブスクを解約したい")[0] is False
    assert _member_roster_signals("今日の天気は？")[0] is False


# --- the roster block ---------------------------------------------------------

def _roster_store(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id=GUILD)
    for index in range(12):
        _insert_user(
            store, f"9000{index:02d}", username=f"fake{index:02d}",
            display_name=f"フェイク{index:02d}",
            joined_at=f"2026-10-{index + 1:02d}T00:00:00.000000+00:00")
    # The newest joiner of all has left: never listed as a member.
    _insert_user(store, "900099", username="gonefake", display_name="退会済",
                 joined_at="2026-11-26T00:00:00.000000+00:00", in_guild=0)
    # Presence never verified: not listed either.
    _insert_user(store, "900098", username="unknownfake", display_name="未確認",
                 joined_at="2026-11-25T00:00:00.000000+00:00", in_guild=None)
    return store


def test_member_roster_lists_newest_present_members_capped(tmp_path):
    chunk = _member_roster(_roster_store(tmp_path), GUILD)
    text = str(chunk)
    assert "メンバー参加日一覧（確定情報）" in text
    assert "全12人中10人まで" in text
    assert "フェイク11（@fake11）: 2026年10月12日 9:00（JST）" in text
    assert "フェイク02" in text and "フェイク01" not in text  # capped at 10
    assert "退会済" not in text and "未確認" not in text


def test_member_roster_oldest_first_and_year_filter(tmp_path):
    store = _roster_store(tmp_path)
    text = str(_member_roster(store, GUILD, order="asc"))
    assert "参加日が古い順" in text
    assert text.index("フェイク00") < text.index("フェイク09")
    assert "フェイク11" not in text  # the cap cuts the newest end here
    text = str(_member_roster(store, GUILD, year=2026))
    assert "2026年にサーバー参加したメンバー" in text
    assert _member_roster(store, GUILD, year=2024) is None


# --- pre_retrieve integration -------------------------------------------------

def _msg(number, text, author_id=PERSON, username="memberfake"):
    value = record(number, channel="10", text=text)
    value["id"] = str(2000 + number)
    value["author"] = {"id": author_id, "username": username, "bot": False}
    value["timestamp"] = "2026-11-20T00:00:00+00:00"
    return value


def test_pre_retrieve_delivers_the_roster_for_a_roster_question(tmp_path):
    store = _roster_store(tmp_path)
    store.upsert_many([_msg(1, "こんにちは"), _msg(2, "よろしく")])
    log = LogTools(store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "最近入ったのは誰？", timeout=5, budget=6000))
    assert result.diagnostics["member_roster_question"] is True
    assert result.diagnostics["member_roster_delivered"] is True
    assert "メンバー参加日一覧（確定情報）" in str(result)


def test_pre_retrieve_keeps_the_roster_off_person_questions(tmp_path):
    store = _roster_store(tmp_path)
    store.upsert_many([_msg(1, "こんにちは"), _msg(2, "よろしく")])
    _insert_user(store, PERSON)
    log = LogTools(store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, f"<@{PERSON}> について教えて", timeout=5, budget=6000))
    assert result.diagnostics["member_roster_question"] is False
    assert result.diagnostics["member_roster_delivered"] is False
    assert result.diagnostics["member_info_delivered"] is True


# --- roster retry (sufficiency gate, mirroring the GitHub block) --------------

def test_sufficiency_forces_one_roster_retry():
    verdict = _assess_sufficiency({}, [], roster_topic=True)
    assert verdict["sufficient"] is False
    assert "member_roster" in verdict["weak"]
    verdict = _assess_sufficiency(
        {}, [], roster_topic=True, roster_retried=True)
    assert "member_roster" not in verdict["weak"]
    verdict = _assess_sufficiency(
        {"member_roster": [LogResult("roster")]}, [], roster_topic=True)
    assert "member_roster" not in verdict["weak"]
    verdict = _assess_sufficiency({}, [])
    assert "member_roster" not in verdict["weak"]


def test_roster_block_retried_after_first_pass_failure(tmp_path, monkeypatch):
    # The phase-1 build dies transiently; the sufficiency gate must
    # spend one extension retry on the roster instead of leaving a
    # roster question with no evidence at all.
    import nezubot_discrod.server_log_context as context_module

    store = _roster_store(tmp_path)
    real = context_module._member_roster
    calls = {"count": 0}

    def flaky(store_, guild_id, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient failure")
        return real(store_, guild_id, **kwargs)

    monkeypatch.setattr(context_module, "_member_roster", flaky)
    log = LogTools(store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "最近入ったのは誰？", timeout=5, budget=6000,
        budgets=ROUTE_BUDGETS_MS))
    assert calls["count"] >= 2
    assert result.diagnostics["member_roster_delivered"] is True
    assert "メンバー参加日一覧（確定情報）" in str(result)


# --- guidance -----------------------------------------------------------------

def test_guidance_covers_the_roster_and_presence_lines():
    assert "メンバー参加日一覧（確定情報）" in LOG_GUIDANCE
    assert "現在のロール line" in LOG_GUIDANCE
    assert "no longer in the server" in LOG_GUIDANCE
    assert "ブースト開始日 line" in LOG_GUIDANCE
