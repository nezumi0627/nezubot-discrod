"""v1.20.1 regression tests: deterministic latest-activity facts.

Reproduces the 2026-11-25 20:50 incident shape with fake speakers:
the owner asks, mentioning a regular who has reappeared today, when
that person last spoke BEFORE today. Production answered with a
timestamp that exists nowhere in the archive (2025-05-29 13:41 UTC)
because no route supplied the deterministic answer — the overview's
last_seen is the all-time maximum (today, since the person had
already spoken) and free search is relevance-ordered. The fix adds a
mechanically aggregated person_activity chunk (latest overall +
latest before today, JST) for last-activity-shaped questions.
"""
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from test_server_log import record

from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import (
    _ACTIVITY_QUESTION,
    _person_activity,
    pre_retrieve,
)
from nezubot_discrod.server_log_tools import LogTools

JST = ZoneInfo("Asia/Tokyo")
PERSON = "30"
CALLER = "999"

INCIDENT_QUESTION = (
    "<@30> この人物が久しぶりにサーバーに顔出したんだけど、"
    "今日より前は最後いつ話してた？"
)


def _msg(number, when_utc, text, author_id=PERSON, username="itsuki_fake"):
    value = record(number, channel="10", text=text)
    value["id"] = str(1000 + number)
    value["author"] = {"id": author_id, "username": username, "bot": False}
    value["timestamp"] = when_utc
    return value


@pytest.fixture
def activity_store(tmp_path):
    """Person 30 spoke today, two days ago (the incident's ground
    truth shape), and long ago in May 2025 (the era the model
    wrongly answered from)."""
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

def test_activity_detector_matches_the_incident_question():
    assert _ACTIVITY_QUESTION.search(INCIDENT_QUESTION)
    assert _ACTIVITY_QUESTION.search("タヌキ最後いつ話してた？")
    assert _ACTIVITY_QUESTION.search("Xの最終発言っていつ？")
    assert _ACTIVITY_QUESTION.search("あの人、前回いつ来たっけ")


def test_activity_detector_ignores_plain_person_questions():
    assert not _ACTIVITY_QUESTION.search("タヌキについて教えて")
    assert not _ACTIVITY_QUESTION.search("Xってどんな人？")
    assert not _ACTIVITY_QUESTION.search("おれの呼ばれ方色々おしえて")


# --- the deterministic lookup ----------------------------------------------

def test_person_activity_reports_latest_and_before_today_in_jst(activity_store):
    now = datetime.now(JST)
    chunk = _person_activity(activity_store, ["10"], PERSON, now=now)
    assert chunk is not None
    text = str(chunk)
    yesterday = now - timedelta(days=1)
    # The before-today entry is the ground truth: yesterday 13:02 JST,
    # converted from the stored 04:02:31Z — not the May 2025 decoy.
    assert (f"{yesterday.year}年{yesterday.month}月{yesterday.day}日 13:02"
            in text)
    assert "より前の最新の発言" in text
    assert "最新の発言" in text
    assert "2025年5月26日" not in text.split("最新の発言")[1].split("\n")[0]


def test_person_activity_when_the_person_has_not_spoken_today(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        _msg(1, "2025-05-26T04:57:27+00:00", "むずそう"),
        _msg(2, "2026-11-23T04:02:31+00:00", "呼び捨てヤメロ"),
    ])
    now = datetime(2026, 11, 25, 20, 50, tzinfo=JST)
    chunk = _person_activity(store, ["10"], PERSON, now=now)
    text = str(chunk)
    assert "2026年11月23日 13:02" in text
    assert "今日より前の最新の発言も上記の発言" in text


def test_person_activity_unknown_person_returns_none(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([_msg(1, "2026-11-23T04:02:31+00:00", "呼び捨てヤメロ")])
    assert _person_activity(store, ["10"], "424242") is None


# --- pre_retrieve integration: the exact incident shape ---------------------

def test_pre_retrieve_delivers_activity_facts_for_the_incident_question(
        activity_store):
    log = LogTools(activity_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, INCIDENT_QUESTION, timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["activity_question"] is True
    assert result.diagnostics["activity_delivered"] is True
    text = str(result)
    assert "発言日時（確定情報）" in text
    now = datetime.now(JST)
    yesterday = now - timedelta(days=1)
    assert (f"{yesterday.year}年{yesterday.month}月{yesterday.day}日 13:02"
            in text)
    # The deterministic block rides ahead of the overview sample, so
    # the model meets the exact dates before any sampled message.
    assert text.index("発言日時（確定情報）") < text.index("Speaker overview")


def test_pre_retrieve_omits_activity_for_plain_person_questions(activity_store):
    log = LogTools(activity_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "<@30> について教えて", timeout=5, budget=6000,
    ))
    assert result.diagnostics["people"] == [PERSON]
    assert result.diagnostics["activity_question"] is False
    assert result.diagnostics["activity_delivered"] is False
    assert "発言日時（確定情報）" not in str(result)
