"""v1.24.0 regression tests: the evidence-verification gate.

Reproduces the vex-duel surrender pattern (benchmark cases #19/#30):
a speaker disputes a fact about the record — what was said, who said
it — and the bot used to concede without checking, even when the
archive proved the claim wrong (case #30: the speaker insisted the
「好きぴ」 in a profile of her was someone else's, while her own
message 「好きぴの方とも双方ffになった」 sat in the archive). A
prompt-only instruction to verify first (v1.21.0) did not fix it, so
the gate is mechanical: detection in claim_check.py, a targeted
verification search in pre_retrieve (the speaker's own messages
first), a 主張の検証 chunk, and an explicit unverified marker in the
turn payload.
"""
import asyncio
import json

import pytest
from test_assistant import client_for, message
from test_server_log import record

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.claim_check import (
    claim_denies_authorship,
    claim_keywords,
    compose_claim_floor,
    detect_claim_dispute,
    draft_concedes,
)
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import pre_retrieve
from nezubot_discrod.server_log_tools import LogTools

CALLER = "999"

# Duel case #30 (raw pair #433), verbatim: the claim is false — the
# speaker herself wrote the 好きぴ message the profile cited.
NAMI_CLAIM = (
    "？？？\n記憶が上手くできて無さそう。\n"
    "わたしがいったはつげんじゃないやつもあるし、\n"
    "好きぴに関しては直前に貼られたリンクの投稿主の好きぴだし、"
)
# Same-shape stand-ins for duel case #19 (the duel's sample list is
# lost; these are the preserved raw pairs of the identical shape).
AKI_CLAIM = "お前のとこに瑞稀を嫁に出した記憶は無いぞ"
MERRY_CLAIM = "別にこの文章の中でこのサーバの人間で一番危険なんて言ってないでしょ？"


def _msg(number, when, text, author_id=CALLER, username="nami"):
    value = record(number, channel="10", text=text)
    value["id"] = str(1000 + number)
    value["author"] = {"id": author_id, "username": username, "bot": False}
    value["timestamp"] = when
    return value


@pytest.fixture
def claim_store(tmp_path):
    """The caller (999) has a past message that contradicts the
    claim she is about to make, plus unrelated chatter."""
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        _msg(1, "2026-01-01T00:01:00+00:00", "好きぴの方とも双方ffになった"),
        _msg(2, "2026-01-01T00:02:00+00:00", "同じ学校で同じ趣味だったから行けた"),
        _msg(3, "2026-01-01T00:03:00+00:00", "今日はカレーを作った",
             author_id="20", username="alice"),
        _msg(4, "2026-01-01T00:04:00+00:00", "全然関係ない雑談"),
    ])
    return store


# --- detector -------------------------------------------------------------

def test_detector_fires_on_the_duel_cases():
    assert detect_claim_dispute(NAMI_CLAIM)
    assert detect_claim_dispute(AKI_CLAIM)
    assert detect_claim_dispute(MERRY_CLAIM)


def test_detector_fires_on_record_denial_variants():
    assert detect_claim_dispute("そんなこと言ってない")
    assert detect_claim_dispute("俺は言った覚えがない")
    assert detect_claim_dispute("学習しすぎって前言ってなかった？")
    assert detect_claim_dispute("その発言、記録が間違ってる")
    assert detect_claim_dispute("いや、そのまとめデマだよ")


def test_detector_ignores_plain_questions_and_chatter():
    assert not detect_claim_dispute("タヌキについて教えて")
    assert not detect_claim_dispute("Xってどんな人？")
    assert not detect_claim_dispute("おはよう、今日も暑いね")
    assert not detect_claim_dispute("Rustの新しいクレート教えて")
    # An opinion with no record reference is not a record dispute.
    assert not detect_claim_dispute("それは違うと思う")
    assert not detect_claim_dispute("何が違うの")
    # Unrelated negations stay out of scope on purpose.
    assert not detect_claim_dispute("宿題まだ書いてない")
    assert not detect_claim_dispute("その話はまだ話してない")
    # A conditional 言ったら is not a claim about the record.
    assert not detect_claim_dispute(
        "人間の脳ってパラメータで言ったらどんくらいになるかな、MoEとは違うだろうし")


# --- keyword extraction -----------------------------------------------------

def test_keywords_carry_the_disputed_topic():
    assert "好きぴ" in claim_keywords(NAMI_CLAIM)
    assert "瑞稀" in claim_keywords(AKI_CLAIM)
    merry = claim_keywords(MERRY_CLAIM)
    assert "一番危険" in merry or "危険" in merry


def test_keywords_drop_sentence_fragments():
    keywords = claim_keywords(NAMI_CLAIM)
    assert "上手くできて" not in keywords
    assert "無さそう" not in keywords
    assert all(len(word) >= 2 for word in keywords)
    assert len(keywords) <= 5


# --- pre_retrieve integration ----------------------------------------------

def test_pre_retrieve_delivers_the_contradicting_own_message(claim_store):
    log = LogTools(claim_store, [10], caller_id=int(CALLER))
    result = asyncio.run(pre_retrieve(
        log, NAMI_CLAIM, message_id="2000", timeout=5, budget=6000,
    ))
    assert result.diagnostics["claim_dispute"] is True
    assert result.diagnostics["claim_check_delivered"] is True
    text = str(result)
    assert "主張の検証" in text
    # The decisive evidence: the speaker's OWN past message, which
    # contradicts the claim she is making in this turn.
    assert "好きぴの方とも双方ffになった" in text
    # The verification block leads the delivered context.
    assert text.index("主張の検証") < text.index("好きぴの方とも双方ffになった")


def test_pre_retrieve_no_gate_for_plain_questions(claim_store):
    log = LogTools(claim_store, [10], caller_id=int(CALLER))
    result = asyncio.run(pre_retrieve(
        log, "カレーについて教えて", message_id="2000", timeout=5, budget=6000,
    ))
    assert result.diagnostics["claim_dispute"] is False
    assert result.diagnostics["claim_check_delivered"] is False
    assert "主張の検証" not in str(result)


# --- assistant payload -------------------------------------------------------

def test_assistant_payload_marks_the_claim_unverified(claim_store):
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=claim_store)
    asyncio.run(ai.reply(
        10, NAMI_CLAIM, member_id=int(CALLER),
        speaker_names={"username": "nami", "display_name": "ねこ",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert payload["claim_dispute"]["status"] == "unverified"
    assert "UNVERIFIED" in payload["claim_dispute"]["note"]
    assert "主張の検証" in payload["retrieved_server_context"]["excerpts"]


# --- draft inspection --------------------------------------------------------

# The duel #30 surrender drafts (before the gate, and after its
# pre-retrieval half alone): both concede with an apology/agreement.
SURRENDER_DRAFTS = (
    "ごめん、発言の主語を取り違えてる。貼られたリンクの投稿主の"
    "「好きぴ」の話まで、ねこ自身の発言として扱ったのは誤りだね。",
    "その指摘は合ってる。貼られたプロフィールは別人の発言やリンク先の"
    "投稿主の「好きぴ」まで、ねこ本人の話みたいに混ぜてた。",
    "そうだね、申し訳ない、そこは分けて扱う。",
)
# Evidence-based answers that must NOT be treated as concessions:
# a rebuttal citing the record, and a basis-stating acknowledgment
# of a claim that was actually correct (the Merry shape).
GROUNDED_DRAFTS = (
    "いや、出どころは本人の発言だよ。11月に「好きぴの方とも双方ffに"
    "なった」って自分で書いてる。",
    "その文では「警戒リストの上位」とは言ってるけど、「このサーバで"
    "一番危険」とは言ってないね。そこはVexの言い換えが強めだった。",
    "記録上は、16:48にvexが「瑞稀ちゃん嫁」って言ってるだけだな。"
    "嫁に出した話までは確認できない。",
)


def test_draft_concedes_matches_surrender_drafts():
    for draft in SURRENDER_DRAFTS:
        assert draft_concedes(draft), draft


def test_draft_concedes_ignores_grounded_answers():
    for draft in GROUNDED_DRAFTS:
        assert not draft_concedes(draft), draft


def test_concession_draft_is_revised_once(claim_store):
    client, sent = client_for(
        message(SURRENDER_DRAFTS[0]),
        message("いや、出どころは本人の発言だよ。11月に「好きぴの方とも"
                "双方ffになった」って自分で書いてる。"),
    )
    ai = Assistant(client, "test", SandboxSettings(), server_log=claim_store)
    answer = asyncio.run(ai.reply(
        10, NAMI_CLAIM, member_id=int(CALLER),
        speaker_names={"username": "nami", "display_name": "ねこ",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    assert answer.startswith("いや、出どころは本人の発言だよ")
    assert len(sent) == 2
    # The revision call carries the draft in its nudge, plus the
    # verified hits themselves (the speaker's own messages).
    nudge = sent[1]["messages"][-1]["content"]
    assert "下書きの見直し" in nudge
    assert "検証で見つかった話者本人の発言" in nudge
    assert "好きぴの方とも双方ff" in nudge


def test_grounded_draft_is_not_revised(claim_store):
    client, sent = client_for(message(GROUNDED_DRAFTS[0]))
    ai = Assistant(client, "test", SandboxSettings(), server_log=claim_store)
    answer = asyncio.run(ai.reply(
        10, NAMI_CLAIM, member_id=int(CALLER),
        speaker_names={"username": "nami", "display_name": "ねこ",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    assert answer.startswith("いや、出どころは本人の発言だよ")
    assert len(sent) == 1


# --- deterministic floor -----------------------------------------------------

def test_floor_helpers():
    assert claim_denies_authorship(NAMI_CLAIM)
    assert not claim_denies_authorship(AKI_CLAIM)
    assert not claim_denies_authorship(MERRY_CLAIM)
    floor = compose_claim_floor([
        {"keyword": "好きぴ", "date": "2025-08-10",
         "text": "おいらもはや好きぴ強くて死ぬ"},
        {"keyword": "好きぴ", "date": "2026-11-06",
         "text": "好きぴの方とも双方ffになった"},
    ])
    assert floor.startswith("記録を確認したけど")
    assert "『好きぴ』は本人の発言にある" in floor
    assert "2026-11-06「好きぴの方とも双方ffになった」" in floor
    assert "私の発言ではない" in floor


def test_floor_replaces_a_twice_conceded_draft(claim_store):
    # Draft concedes, the revision concedes too — the floor then
    # answers from the verified hits alone.
    client, sent = client_for(
        message(SURRENDER_DRAFTS[0]),
        message("ごめん、その指摘のとおりだった。"),
    )
    ai = Assistant(client, "test", SandboxSettings(), server_log=claim_store)
    answer = asyncio.run(ai.reply(
        10, NAMI_CLAIM, member_id=int(CALLER),
        speaker_names={"username": "nami", "display_name": "ねこ",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    assert answer.startswith("記録を確認したけど")
    assert "好きぴの方とも双方ff" in answer
    assert len(sent) == 2


def test_floor_stays_off_for_event_denials(tmp_path):
    # The aki shape: the speaker denies an EVENT, not authorship.
    # Even with the caller's own keyword hits delivered and two
    # concession drafts, the floor must not fire — a keyword's
    # presence in his messages proves nothing about the event.
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        _msg(1, "2026-10-20T07:48:00+00:00", "瑞稀ちゃんかわいい",
             author_id="999", username="aki"),
        _msg(2, "2026-10-21T07:48:00+00:00", "瑞稀の話はもういい",
             author_id="999", username="aki"),
    ])
    client, sent = client_for(
        message("ごめん、そうだったかも。"),
        message("ごめん、記憶が曖昧だった。"),
    )
    ai = Assistant(client, "test", SandboxSettings(), server_log=store)
    answer = asyncio.run(ai.reply(
        10, AKI_CLAIM, member_id=999,
        speaker_names={"username": "aki", "display_name": "aki",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    assert answer == "ごめん、記憶が曖昧だった。"
    assert len(sent) == 2


def test_no_revision_without_delivered_evidence(tmp_path):
    # A dispute turn whose verification search finds nothing: the
    # gate's payload note still fires, but there is no evidence to
    # re-answer from, so a concession draft stands (single call).
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        _msg(1, "2026-01-01T00:01:00+00:00", "今日はカレーを作った",
             author_id="20", username="alice"),
    ])
    client, sent = client_for(message(SURRENDER_DRAFTS[0]))
    ai = Assistant(client, "test", SandboxSettings(), server_log=store)
    answer = asyncio.run(ai.reply(
        10, NAMI_CLAIM, member_id=int(CALLER),
        speaker_names={"username": "nami", "display_name": "ねこ",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    assert answer.startswith("ごめん")
    assert len(sent) == 1


def test_assistant_payload_has_no_claim_field_for_plain_turns(claim_store):
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=claim_store)
    asyncio.run(ai.reply(
        10, "カレーについて教えて", member_id=int(CALLER),
        speaker_names={"username": "nami", "display_name": "ねこ",
                       "server_nickname": None},
        channel_ids=[10],
        message_id=2000,
    ))
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert "claim_dispute" not in payload
