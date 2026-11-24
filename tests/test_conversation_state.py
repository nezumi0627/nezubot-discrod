import time

from nezubot_discrod.conversation_state import (
    STATE_TTL_SECONDS,
    ConversationStateStore,
    active_topic_view,
    build_state_update,
    classify_intent,
    state_context_view,
)
from nezubot_discrod.server_log import ServerLog

CLAIM = "キャッシュを全件走査する時点でヒットしないキーが絶対にでてきて非効率だ"


def anchor(mid="100", author="301", name="sora_server", content="これ解決できないの"):
    return {"id": mid, "author_id": author, "author_name": name, "content": content}


def claim_conversation():
    return {
        "reply_chain": [
            {"author": "ゆるふわモカ", "author_id": "302", "message_id": "90",
             "content": CLAIM},
        ],
        "reply_target": {
            "author": "ゆるふわモカ", "author_id": "302", "message_id": "90",
            "content": CLAIM,
        },
        "mentions": [],
    }


def test_classify_intent_shapes():
    assert classify_intent("これ解決できないの") == "request_solution"
    assert classify_intent("それ以外は？") == "followup"
    assert classify_intent("じゃあ逆に？") == "followup"
    assert classify_intent("タヌキってどんな人？") == "person_question"
    assert classify_intent("FTSって何？") == "question"
    assert classify_intent("なるほもね") == "statement"


def test_state_from_reply_claim_turn():
    state = build_state_update(
        None, now=1000.0, anchor=anchor(), question="これ解決できないの",
        conversation=claim_conversation(),
    )
    assert state["topic"] == CLAIM[:80]
    subject = state["active_subject"]
    assert subject["kind"] == "message" and subject["id"] == "90"
    assert subject["author_id"] == "302"
    (fact,) = state["established_facts"]
    assert fact["kind"] == "claim_under_discussion"
    assert fact["confirmed"] is False
    assert fact["message_id"] == "90"
    assert state["unresolved_questions"][0]["text"] == "これ解決できないの"
    assert state["last_intent"] == "request_solution"
    assert {p["id"] for p in state["participants"]} == {"301", "302"}
    assert state["turn_count"] == 1


def test_state_grounded_person_becomes_subject_and_attempted_accumulates():
    conversation = {
        "reply_chain": [],
        "mentions": [],
        "grounded_person": {"id": "555", "display_name": "ZephyrusZ",
                            "basis": "reply_target"},
    }
    state = build_state_update(
        None, now=1000.0, anchor=anchor(content="この人物の過去のメッセージさがして"),
        question="この人物の過去のメッセージさがして", conversation=conversation,
    )
    assert state["active_subject"]["kind"] == "person"
    assert state["active_subject"]["id"] == "555"
    follow = build_state_update(
        state, now=1010.0, anchor=anchor(mid="101", content="どこ住み？"),
        question="どこ住み？", conversation={"reply_chain": [], "mentions": []},
        answer_summary="ZephyrusZは北九州の話をよくしている",
    )
    # A signal-less follow-up keeps the subject and records the attempt.
    assert follow["active_subject"]["id"] == "555"
    assert follow["attempted_solutions"][0]["text"].startswith("ZephyrusZは")
    assert follow["turn_count"] == 2


def test_topic_switch_resets_topic_fields_but_keeps_participants():
    first = build_state_update(
        None, now=1000.0, anchor=anchor(), question="これ解決できないの",
        conversation=claim_conversation(),
    )
    switched = build_state_update(
        first, now=1010.0,
        anchor=anchor(mid="101", content="Rustの所有権について詳しく教えてほしい"),
        question="Rustの所有権について詳しく教えてほしい",
        conversation={"reply_chain": [], "mentions": []},
    )
    assert switched["topic"].startswith("Rustの所有権")
    assert switched["active_subject"] is None
    assert switched["established_facts"] == []
    # A short follow-up never counts as a switch.
    follow = build_state_update(
        first, now=1010.0, anchor=anchor(mid="101", content="それ以外は？"),
        question="それ以外は？", conversation={"reply_chain": [], "mentions": []},
    )
    assert follow["active_subject"] is not None
    assert follow["topic"] == first["topic"]


def test_store_roundtrip_ttl_and_clear(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    states = ConversationStateStore(store)
    assert states.load("10") is None
    state = build_state_update(
        None, now=time.time(), anchor=anchor(), question="これ解決できないの",
        conversation=claim_conversation(),
    )
    states.save("10", state)
    loaded = states.load("10")
    assert loaded["topic"] == state["topic"]
    assert loaded["active_subject"]["id"] == "90"
    stale = dict(state, updated_at=time.time() - STATE_TTL_SECONDS - 1)
    states.save("10", stale)
    assert states.load("10") is None
    states.save("10", state)
    states.clear("10")
    assert states.load("10") is None


def test_views_shape():
    state = build_state_update(
        None, now=1000.0, anchor=anchor(), question="これ解決できないの",
        conversation=claim_conversation(), answer_summary="分割すれば緩和できる",
    )
    view = state_context_view(state)
    assert view["established_facts"][0]["message_id"] == "90"
    assert view["attempted_solutions"] == ["分割すれば緩和できる"]
    assert view["last_intent"] == "request_solution"
    topic = active_topic_view(state)
    assert topic["subject"]["kind"] == "message"
    fallback = active_topic_view(None, {
        "grounded_person": {"id": "7", "display_name": "vex"},
    })
    assert fallback["subject"] == {"kind": "person", "id": "7", "label": "vex"}
    assert state_context_view(None) is None
    assert active_topic_view(None, None) is None
