"""v1.28.0 regression tests: deterministic earliest-activity facts.

Reproduces the 2026-11-26 18:34 incident shape with fake speakers:
the owner asks for his own first-participation date (「俺の過去ログ
さかのぼって初参加の日にち探して」). Production answered 2026-10-23
11:58 UTC — a timestamp of OTHER people's messages — because the
v1.20.1 fix supplied a deterministic block only for the LAST
activity; no route supplied the person's earliest message (the
overview's first_seen sits raw inside a JSON header and its samples
are time-stratified), so the model composed a date from search hits.
The fix mirrors _person_activity: a mechanically aggregated
person_first_activity chunk (earliest message by author id, JST,
with the "earliest confirmable for this account" reservation) for
first-activity-shaped questions.
"""
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from test_server_log import record

from nezubot_discrod.devin_answer import build_devin_prompt
from nezubot_discrod.prompts import LOG_GUIDANCE, LOG_SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import (
    _FIRST_ACTIVITY_QUESTION,
    _person_first_activity,
    pre_retrieve,
)
from nezubot_discrod.server_log_tools import LogTools

JST = ZoneInfo("Asia/Tokyo")
PERSON = "30"
CALLER = "999"

INCIDENT_QUESTION = "俺の過去ログさかのぼって初参加の日にち探して"


def _msg(number, when_utc, text, author_id=PERSON, username="itsuki_fake"):
    value = record(number, channel="10", text=text)
    value["id"] = str(1000 + number)
    value["author"] = {"id": author_id, "username": username, "bot": False}
    value["timestamp"] = when_utc
    return value


@pytest.fixture
def first_store(tmp_path):
    """Person 30's earliest message is 2025-05-26 (the ground truth);
    later messages yesterday and today are the decoys a relevance
    search would surface instead."""
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    now = datetime.now(JST)
    today = (now - timedelta(minutes=5)).astimezone(
        ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    yesterday = (now - timedelta(days=1)).replace(
        hour=13, minute=2, second=31, microsecond=0).astimezone(
        ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    store.upsert_many([
        _msg(1, "2025-05-26T04:57:27+00:00", "むずそう"),
        _msg(2, yesterday, "呼び捨てヤメロ"),
        _msg(3, today, "お久しぶりです"),
    ])
    with store.connect() as db:
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, display_name, "
            "nickname) VALUES ('30', '100', 'itsuki_fake', 'たろう', 'ゆうき')"
        )
    return store


# --- detector -------------------------------------------------------------

def test_first_activity_detector_matches_the_incident_question():
    assert _FIRST_ACTIVITY_QUESTION.search(INCIDENT_QUESTION)
    assert _FIRST_ACTIVITY_QUESTION.search("<@30> の初参加はいつ？")
    assert _FIRST_ACTIVITY_QUESTION.search("Xの最初の発言っていつ？")
    assert _FIRST_ACTIVITY_QUESTION.search("タヌキが初めて話したのいつだっけ")
    assert _FIRST_ACTIVITY_QUESTION.search("あの人いつ参加したの？")


def test_first_activity_detector_matches_more_first_shapes():
    assert _FIRST_ACTIVITY_QUESTION.search("あの人いつからこのサーバーにいるの？")
    assert _FIRST_ACTIVITY_QUESTION.search("Xはいつからいる？")
    assert _FIRST_ACTIVITY_QUESTION.search("Xの初めての発言はいつ？")
    assert _FIRST_ACTIVITY_QUESTION.search("Xが最初に発言したのはいつ？")
    assert _FIRST_ACTIVITY_QUESTION.search("Xはいつから参加してるの？")


def test_first_activity_detector_ignores_other_shapes():
    assert not _FIRST_ACTIVITY_QUESTION.search("タヌキについて教えて")
    assert not _FIRST_ACTIVITY_QUESTION.search("Xってどんな人？")
    assert not _FIRST_ACTIVITY_QUESTION.search("おれの呼ばれ方色々おしえて")
    # The last-activity shape is _ACTIVITY_QUESTION's, not this one's.
    assert not _FIRST_ACTIVITY_QUESTION.search("タヌキ最後いつ話してた？")
    assert not _FIRST_ACTIVITY_QUESTION.search("Xの最終発言っていつ？")


# --- the deterministic lookup ----------------------------------------------

def test_person_first_activity_reports_the_earliest_message_in_jst(first_store):
    chunk = _person_first_activity(first_store, ["10"], PERSON)
    assert chunk is not None
    text = str(chunk)
    # 2025-05-26T04:57:27Z is 13:57 JST — the ground truth, not the
    # yesterday/today decoys.
    assert "最初の発言（確定情報）" in text
    assert "2025年5月26日 13:57" in text
    assert "むずそう" in text
    # The reservation the model must carry into its answer.
    assert "アーカイブで確認できる" in text
    assert "アカウント変更" in text


def test_person_first_activity_unknown_person_returns_none(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([_msg(1, "2026-11-23T04:02:31+00:00", "呼び捨てヤメロ")])
    assert _person_first_activity(store, ["10"], "424242") is None


# --- pre_retrieve integration ----------------------------------------------

def test_pre_retrieve_delivers_first_activity_for_a_mention_question(
        first_store):
    log = LogTools(first_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "<@30> の初参加の日にち探して", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["first_activity_question"] is True
    assert result.diagnostics["first_activity_delivered"] is True
    text = str(result)
    assert "最初の発言（確定情報）" in text
    assert "2025年5月26日 13:57" in text
    # The deterministic block rides ahead of the overview sample, so
    # the model meets the exact date before any sampled message.
    assert text.index("最初の発言（確定情報）") < text.index("Speaker overview")


def test_pre_retrieve_delivers_first_activity_for_the_exact_incident(
        first_store):
    # The incident itself: the owner asking about himself, no mention.
    log = LogTools(first_store, [10], caller_id=PERSON)
    result = asyncio.run(pre_retrieve(
        log, INCIDENT_QUESTION, timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["first_activity_delivered"] is True
    assert "2025年5月26日 13:57" in str(result)


def test_pre_retrieve_omits_first_activity_for_plain_person_questions(
        first_store):
    log = LogTools(first_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "<@30> について教えて", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["first_activity_question"] is False
    assert result.diagnostics["first_activity_delivered"] is False
    assert "最初の発言（確定情報）" not in str(result)


def test_pre_retrieve_omits_first_activity_when_two_people_resolve(
        first_store):
    # A second resolved person makes the question ambiguous: whose
    # first message would the block describe? Nothing is delivered.
    first_store.upsert_many([
        _msg(9, "2025-06-01T00:00:00+00:00", "別人の発言",
             author_id="31", username="otherfake"),
    ])
    log = LogTools(first_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "<@30> と <@31> の初参加はいつ？", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON, "31"]
    assert result.diagnostics["first_activity_question"] is True
    assert result.diagnostics["first_activity_delivered"] is False
    assert "最初の発言（確定情報）" not in str(result)


def test_pre_retrieve_omits_first_activity_when_nobody_resolves(first_store):
    log = LogTools(first_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "知らない誰かの初発言はいつ？", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == []
    assert result.diagnostics["first_activity_question"] is True
    assert result.diagnostics["first_activity_delivered"] is False
    assert "最初の発言（確定情報）" not in str(result)


def test_pre_retrieve_delivers_both_blocks_when_first_and_last_asked(
        first_store):
    # First and last in one question: the v1.20.1 latest block and
    # the new earliest block coexist.
    log = LogTools(first_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "<@30> の最初の発言と最後の発言はいつ？", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["activity_delivered"] is True
    assert result.diagnostics["first_activity_delivered"] is True
    text = str(result)
    assert "発言日時（確定情報）" in text
    assert "最初の発言（確定情報）" in text
    assert "2025年5月26日 13:57" in text


# --- guidance reaches both answer paths -------------------------------------

def test_guidance_covers_the_first_activity_block():
    assert "最初の発言（確定情報）" in LOG_GUIDANCE
    assert "アーカイブで確認できる最古は" in LOG_GUIDANCE
    # The Devin path flattens the same system prompt (LOG_SYSTEM is
    # SYSTEM + LOG_GUIDANCE), so the instruction reaches swe-2-1 too.
    prompt = build_devin_prompt([
        {"role": "system", "content": LOG_SYSTEM},
        {"role": "user", "content": "x"},
    ])
    assert "最初の発言（確定情報）" in prompt
