"""v1.31.0 regression tests: first-activity facts in follow-up turns.

The v1.28.0 fix armed the deterministic person_first_activity block
only when the latest message itself carried first-activity wording.
The 2026-11-26 recurrence came through the conversation instead:

1. 23:32, the owner replied 「ひま」 to the bot's own pre-fix wrong
   answer (「初参加は10月23日…」, 18:34, still in the transcript).
   Diagnostics (turn ef8e8741e47c): first_activity_question false,
   people [] (grounding hit its deadline under load), one lexical
   chunk, zero tool calls — the bot chatted on, restating 10月23日
   as the premise.
2. 01:00, the owner said 「しらべてそれを」. Diagnostics (turn
   975b1eda524b): first_activity_question false again, people [],
   pre_retrieve delivered NOTHING (chars 0); the rewritten query did
   carry the reply target's 「初参加」 wording but the detector never
   looked at it. The model's own tool searches returned no rows, and
   it reproduced the same wrong date from the transcript.

The fix: the detector also reads the rewritten query and the context
query (which already carry the reply target's words) for
context-dependent turns; when grounding resolves nobody, the block
takes its subject from the conversation's pointers (seed, then
carryover, then — for a context-dependent turn — the caller, whose
own first participation the incident's follow-ups continued); and
the guidance now states that the bot's own earlier statements are
not evidence for a first-participation date.
"""
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from test_server_log import record

from nezubot_discrod.devin_answer import build_devin_prompt
from nezubot_discrod.prompts import LOG_GUIDANCE, LOG_SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import _first_activity_signals, pre_retrieve
from nezubot_discrod.server_log_tools import LogTools

JST = ZoneInfo("Asia/Tokyo")
PERSON = "30"
CALLER = "999"

# The bot's pre-fix wrong answer (18:34) and its 23:32 restatement,
# as they sat in the reply chain / transcript of the follow-up turns.
BOT_WRONG_1 = (
    "ログで確認できる初参加は10月23日。最古の記録はその日の11:58 UTC"
    "（日本時間20:58）だけど、これはアーカイブ上の初確認日で、"
    "実際の参加日とは限らない。"
)
BOT_WRONG_2 = (
    "じゃあ、雑談するか。さっきの初参加ログ、10月23日って出てたけど"
    "実際いつからいるの？"
)


def _msg(number, when_utc, text, author_id=PERSON, username="itsuki_fake"):
    value = record(number, channel="10", text=text)
    value["id"] = str(1000 + number)
    value["author"] = {"id": author_id, "username": username, "bot": False}
    value["timestamp"] = when_utc
    return value


@pytest.fixture
def first_store(tmp_path):
    """The caller (person 30) has an earliest message on 2025-05-26 —
    the ground truth the block must deliver — while later messages
    are the decoys a relevance search surfaces."""
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


# --- detection signals -------------------------------------------------------

def test_signals_for_the_incident_followup_shapes():
    # 「ひま」: no wording of its own; the topic arrives via the
    # context query (reply target = the bot's wrong answer).
    direct, topic, dependent = _first_activity_signals(
        "ひま", None, BOT_WRONG_1)
    assert (direct, topic, dependent) == (False, True, True)
    # 「しらべてそれを」: the rewrite carries the reply target's words.
    direct, topic, dependent = _first_activity_signals(
        "しらべてそれを", BOT_WRONG_2 + " しらべてそれを", BOT_WRONG_2)
    assert (direct, topic, dependent) == (False, True, True)
    # The v1.28.0 direct shape still reads as a direct question.
    direct, topic, _ = _first_activity_signals("俺の初参加はいつ？")
    assert (direct, topic) == (True, True)


def test_signals_ignore_context_for_self_contained_messages():
    # A long, self-contained message about something else does not
    # inherit the topic from a first-activity message in the window.
    direct, topic, dependent = _first_activity_signals(
        "検索の仕組みを詳しく説明してほしい", None, BOT_WRONG_1)
    assert (direct, topic, dependent) == (False, False, False)


# --- the two incident turns, replayed at the pre_retrieve level ---------------

def test_followup_casual_continuation_delivers_the_true_first_activity(
        first_store):
    # Incident turn 1: 「ひま」 as a reply to the bot's wrong answer.
    # Nothing in the text resolves a person (as in production, where
    # grounding additionally hit its deadline), so the caller
    # fallback supplies the subject: the conversation is about the
    # caller's own first participation.
    log = LogTools(first_store, [10], caller_id=PERSON)
    result = asyncio.run(pre_retrieve(
        log, "ひま", timeout=5, budget=6000,
        context_query=BOT_WRONG_1 + " 俺の過去ログさかのぼって初参加の日にち探して",
    ))
    diag = result.diagnostics
    assert diag["people"] == []
    assert diag["first_activity_question"] is False
    assert diag["first_activity_topic"] is True
    assert diag["first_activity_subject"] == "caller"
    assert diag["first_activity_delivered"] is True
    text = str(result)
    assert "最初の発言（確定情報）" in text
    # The ground truth, not the transcript's 10月23日.
    assert "2025年5月26日 13:57" in text
    # The fact is attributed: the person's summary rides along.
    assert diag["summary_delivered"] is True


def test_followup_elliptical_instruction_delivers_the_true_first_activity(
        first_store):
    # Incident turn 2: 「しらべてそれを」, with the rewrite carrying
    # the replied-to bot message (its 23:32 restatement).
    log = LogTools(first_store, [10], caller_id=PERSON)
    result = asyncio.run(pre_retrieve(
        log, "しらべてそれを", timeout=5, budget=6000,
        search_text=BOT_WRONG_2 + " しらべてそれを",
        context_query=BOT_WRONG_2,
    ))
    diag = result.diagnostics
    assert diag["people"] == []
    assert diag["first_activity_question"] is False
    assert diag["first_activity_topic"] is True
    assert diag["first_activity_subject"] == "caller"
    assert diag["first_activity_delivered"] is True
    assert "2025年5月26日 13:57" in str(result)


def test_followup_seed_equal_to_caller_reaches_the_fallback(first_store):
    # The conversation grounded the caller himself: resolve_people
    # deliberately skips a seed equal to the caller, and the bare
    # follow-up text resolves nobody, so grounding returns [] — the
    # fallback is the only path to a subject, and it takes the seed.
    log = LogTools(first_store, [10], caller_id=PERSON)
    result = asyncio.run(pre_retrieve(
        log, "しらべてそれを", timeout=5, budget=6000,
        seed_person=PERSON, context_query=BOT_WRONG_2,
    ))
    diag = result.diagnostics
    assert diag["people"] == []
    assert diag["first_activity_subject"] == "seed"
    assert diag["first_activity_delivered"] is True
    assert "2025年5月26日 13:57" in str(result)


# --- shapes that must still NOT deliver ---------------------------------------

def test_direct_third_party_question_gets_no_caller_fallback(first_store):
    # A direct question about an unknown third party: the caller has
    # messages of his own, but the block must not silently become a
    # fact about the caller (the v1.28.0 no-subject behaviour).
    log = LogTools(first_store, [10], caller_id=PERSON)
    result = asyncio.run(pre_retrieve(
        log, "知らない誰かの初発言はいつ？", timeout=5, budget=6000,
    ))
    diag = result.diagnostics
    assert diag["people"] == []
    assert diag["first_activity_question"] is True
    assert diag["first_activity_subject"] is None
    assert diag["first_activity_delivered"] is False
    assert "最初の発言（確定情報）" not in str(result)


def test_followup_context_does_not_override_an_ambiguous_resolution(
        first_store):
    # Two people resolve: whose first message would the block
    # describe? Nothing is delivered, context topic or not.
    first_store.upsert_many([
        _msg(9, "2025-06-01T00:00:00+00:00", "別人の発言",
             author_id="31", username="otherfake"),
    ])
    log = LogTools(first_store, [10], caller_id=CALLER)
    result = asyncio.run(pre_retrieve(
        log, "<@30> と <@31> のそれをしらべて", timeout=5, budget=6000,
        context_query=BOT_WRONG_1,
    ))
    diag = result.diagnostics
    assert diag["people"] == [PERSON, "31"]
    assert diag["first_activity_topic"] is True
    assert diag["first_activity_delivered"] is False
    assert "最初の発言（確定情報）" not in str(result)


def test_self_contained_question_ignores_first_activity_context(first_store):
    log = LogTools(first_store, [10], caller_id=PERSON)
    result = asyncio.run(pre_retrieve(
        log, "検索の仕組みを詳しく説明してほしい", timeout=5, budget=6000,
        context_query=BOT_WRONG_1,
    ))
    diag = result.diagnostics
    assert diag["first_activity_topic"] is False
    assert diag["first_activity_delivered"] is False
    assert "最初の発言（確定情報）" not in str(result)


# --- guidance reaches both answer paths ----------------------------------------

def test_guidance_marks_own_past_statements_as_non_evidence():
    assert "Your own earlier statements are not evidence" in LOG_GUIDANCE
    assert "the block wins" in LOG_GUIDANCE
    assert "do not assert a date at all" in LOG_GUIDANCE
    prompt = build_devin_prompt([
        {"role": "system", "content": LOG_SYSTEM},
        {"role": "user", "content": "x"},
    ])
    assert "Your own earlier statements are not evidence" in prompt



# --- v1.33.0: the draft-inspection half for first-activity facts ---
#
# The 2026-11-27 01:24 turn (058da82defa1, Codex path): the block WAS
# delivered with the correct date, yet the draft repeated the
# transcript's wrong date and dressed it as re-verified from the
# records, with zero tool calls. The prompt rule ("the block wins")
# demonstrably did not hold, so the finished draft is now checked
# mechanically against the delivered block and revised once.

from nezubot_discrod.assistant import (  # noqa: E402
    draft_contradicts_first_activity,
    first_activity_fact,
)

_BLOCK = (
    "## 最初の発言（確定情報）\n"
    "対象: テストユーザー\n"
    "最初の発言: 2026年7月18日 23:48（JST） #ウェルカム\n"
    "これはアーカイブで確認できる最古の発言であり、Discord の"
    "サーバー参加日時そのものではない。"
)


def test_first_activity_fact_parses_the_block():
    fact = first_activity_fact(_BLOCK)
    assert fact is not None
    jst_date, utc_date = fact
    assert (jst_date.year, jst_date.month, jst_date.day) == (2026, 7, 18)
    # 23:48 JST is 14:48 UTC the same day.
    assert (utc_date.year, utc_date.month, utc_date.day) == (2026, 7, 18)
    assert first_activity_fact("ブロックなしの文脈") is None
    assert first_activity_fact("") is None


def test_first_activity_fact_utc_date_rolls_back():
    block = _BLOCK.replace("2026年7月18日 23:48", "2026年7月18日 0:30")
    _jst, utc_date = first_activity_fact(block)
    assert (utc_date.year, utc_date.month, utc_date.day) == (2026, 7, 17)


def test_draft_contradicts_first_activity_incident_shape():
    fact = first_activity_fact(_BLOCK)
    # The 01:24 draft, verbatim shape: a different full date, the
    # block's date never mentioned.
    incident = ("アーカイブの実レコードで確認し直した。テストユーザーの最古の発言は"
                "2026年10月23日20:58（日本時間）で、実際のサーバー参加日"
                "とは別だよ。")
    assert draft_contradicts_first_activity(incident, fact) is True


def test_draft_contradicts_first_activity_negative_cases():
    fact = first_activity_fact(_BLOCK)
    # States the block's own date: engaged with the block, left alone.
    assert draft_contradicts_first_activity(
        "最古の発言は2026年7月18日 23:48（JST）だよ。", fact) is False
    # Correct date alongside a discussion of the wrong one: the draft
    # mentions the block's date, so the gate stays out of the way.
    assert draft_contradicts_first_activity(
        "前に言った2026年10月23日は誤りで、正しくは2026年7月18日です。",
        fact) is False
    # No full date asserted at all.
    assert draft_contradicts_first_activity(
        "アーカイブで確認できる最古の発言だよ。", fact) is False
    # No delivered block: the gate is unarmed.
    assert draft_contradicts_first_activity(
        "最古の発言は2026年10月23日です。", None) is False
