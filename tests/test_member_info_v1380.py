"""v1.38.0 regression tests: member-record facts + correction style.

Two owner complaints from the 2026-11-27 10:25 person summary:

1. The summary opened by narrating the bot's own past error
   (「サーバー内の『最初の発言』は10月23日ではなく…さっきの10月23日
   という説明は誤り」). The "correct the earlier statement" wording
   in the first-activity guidance — and the draft gate's nudge, which
   literally instructed 訂正 — made the model perform the correction
   instead of just stating the fact. The guidance now says the block
   wins and the correct fact replaces the wrong one WITHOUT narration,
   plus a general correct-once-and-drop-it rule; the nudge matches.

2. The same summary closed with 「サーバー参加日そのものは分からない」
   although joined_at had been sitting in the users table since the
   member sync — nothing carried it into the materials. The fix
   records the full member record (premium_since, roles; created_at
   derived from the snowflake) and delivers a deterministic
   メンバー情報（確定情報） block with the person bundle, the join
   date labelled as Discord's record and kept distinct from the
   archive's earliest message.
"""
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_server_log import record

from nezubot_discrod.devin_answer import build_devin_prompt
from nezubot_discrod.prompts import LOG_GUIDANCE, LOG_SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import _person_member_info, pre_retrieve
from nezubot_discrod.server_log_tools import LogTools

CALLER = "999"

# A synthetic snowflake whose creation instant is 2025-05-20T09:12:00Z.
_EPOCH_MS = 1420070400000
_CREATED_MS = int(datetime(2025, 5, 20, 9, 12, tzinfo=UTC).timestamp() * 1000)
PERSON = str(((_CREATED_MS - _EPOCH_MS) << 22) + 777)
JOINED_AT = "2025-05-21T00:00:00.000000+00:00"   # 2025年5月21日 09:00 JST
PREMIUM_SINCE = "2026-01-02T03:04:05+00:00"      # 2026年1月2日 12:04 JST


def _msg(number, text, author_id=PERSON, username="memberfake"):
    value = record(number, channel="10", text=text)
    value["id"] = str(2000 + number)
    value["author"] = {"id": author_id, "username": username, "bot": False}
    value["timestamp"] = "2026-11-20T00:00:00+00:00"
    return value


@pytest.fixture
def member_store(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        _msg(1, "最近の発言その一"),
        _msg(2, "最近の発言その二"),
        _msg(3, "最近の発言その三"),
    ])
    with store.connect() as db:
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, display_name, "
            "nickname, joined_at, premium_since, roles) VALUES "
            "(?, '100', 'memberfake', 'メンバー', '', ?, ?, '[\"111\"]')",
            (PERSON, JOINED_AT, PREMIUM_SINCE),
        )
    return store


# --- the deterministic block -------------------------------------------------

def test_member_info_block_states_all_record_facts_in_jst(member_store):
    chunk = _person_member_info(member_store, PERSON)
    assert chunk is not None
    text = str(chunk)
    assert "メンバー情報（確定情報）" in text
    assert "サーバー参加日: 2025年5月21日 9:00（JST）" in text
    assert "アカウント作成日: 2025年5月20日 18:12（JST）" in text
    assert "ブースト開始日: 2026年1月2日 12:04（JST）" in text
    # The join date and the archive's earliest message are different
    # records; the block says so, in terms.
    assert "最初の発言）とは別の記録" in text
    assert "代用しない" in text


def test_member_info_block_without_a_synced_row_still_gives_creation(member_store):
    chunk = _person_member_info(member_store, "123456789012345679")
    assert chunk is not None
    text = str(chunk)
    assert "サーバー参加日: 未記録" in text
    assert "アカウント作成日:" in text
    assert "ブースト開始日" not in text


def test_member_info_block_omits_boost_line_when_not_boosting(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    with store.connect() as db:
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, joined_at) "
            "VALUES (?, '100', 'memberfake', ?)", (PERSON, JOINED_AT),
        )
    text = str(_person_member_info(store, PERSON))
    assert "サーバー参加日: 2025年5月21日 9:00（JST）" in text
    assert "ブースト開始日" not in text


def test_member_info_block_returns_none_for_a_non_snowflake_unknown(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    assert _person_member_info(store, "30") is None


# --- pre_retrieve integration --------------------------------------------------

def test_pre_retrieve_delivers_member_info_with_the_person_bundle(member_store):
    # A plain person question — the 10:25 incident shape was a person
    # summary, not a join-date question, so the block must ride with
    # the person bundle rather than wait for 参加日 wording.
    log = LogTools(member_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, f"<@{PERSON}> について長くまとめて", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["member_info_delivered"] is True
    text = str(result)
    assert "メンバー情報（確定情報）" in text
    assert "サーバー参加日: 2025年5月21日 9:00（JST）" in text


def test_pre_retrieve_delivers_member_info_for_a_join_date_question(member_store):
    log = LogTools(member_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, f"<@{PERSON}> のサーバー参加日はいつ？", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["member_info_delivered"] is True
    assert "サーバー参加日: 2025年5月21日 9:00（JST）" in str(result)


# --- guidance: correct once, drop it, never narrate ----------------------------

def test_guidance_keeps_the_block_wins_rule_without_narration():
    assert "the block wins" in LOG_GUIDANCE
    assert "Do not narrate" in LOG_GUIDANCE
    assert "the correct fact simply replaces the wrong one" in LOG_GUIDANCE
    assert "correct the earlier statement instead of repeating it" not in LOG_GUIDANCE


def test_guidance_states_the_general_correct_once_rule():
    assert "Correcting your own earlier errors, in general" in LOG_GUIDANCE
    assert "state the correct fact once, briefly, and drop" in LOG_GUIDANCE
    assert "never re-litigate them in later turns" in LOG_GUIDANCE
    prompt = build_devin_prompt([
        {"role": "system", "content": LOG_SYSTEM},
        {"role": "user", "content": "x"},
    ])
    assert "Correcting your own earlier errors, in general" in prompt


def test_guidance_routes_join_date_questions_to_the_member_block():
    assert "メンバー情報（確定情報） block" in LOG_GUIDANCE
    assert "never substitute one for the other" in LOG_GUIDANCE


def test_draft_gate_nudge_no_longer_orders_a_narrated_correction():
    source = Path("nezubot_discrod/assistant.py").read_text("utf-8")
    assert "過去の誤った日付は訂正" not in source
    assert "の誤りへの言及や謝罪" in source
