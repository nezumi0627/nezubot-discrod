"""v1.20.0 conversation-quality regression tests.

Each test reproduces one failure shape from
CONVERSATION-QUALITY-REPORT.md with fake speakers standing in for
the real people, through the real code paths (resolution, adaptive
retrieval, payload assembly). No test posts anywhere.
"""
import asyncio
import json

import pytest
from test_assistant import client_for, message
from test_server_log import dense_store, record

from nezubot_discrod.ambient import ambient_excluded
from nezubot_discrod.appellations import AppellationIndex, mine_appellations
from nezubot_discrod.assistant import (
    Assistant,
    _compact_history_payload,
    _state_for_turn,
    server_fact_question,
)
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.conversation_state import build_state_update
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import _alias_snapshot, pre_retrieve
from nezubot_discrod.server_log_tools import LogTools


@pytest.fixture
def people_store(tmp_path):
    """Person 30 (username tanuki, display のの) is called タヌキ by
    caller 999 and others; person 40 is a prolific third party whose
    messages about タヌキ flood the lexical route."""
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    for index in range(6):
        value = record(1, channel="10", text=f"tanukiの発言 {index}")
        value["id"] = str(100 + index)
        value["author"] = {"id": "30", "username": "tanuki", "bot": False}
        rows.append(value)
    for index in range(5):
        value = record(1, channel="10", text=f"タヌキさん、それはどう？ {index}")
        value["id"] = str(200 + index)
        value["author"] = {"id": "999", "username": "testuser", "bot": False}
        rows.append(value)
    for index in range(30):
        value = record(1, channel="10", text=f"タヌキの話題 {index} みんなで話した")
        value["id"] = str(300 + index)
        value["author"] = {"id": "40", "username": "aki", "bot": False}
        rows.append(value)
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO message_people VALUES (?, '30', 'reply', 1.0)",
            [(str(200 + index),) for index in range(5)],
        )
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, display_name, nickname) "
            "VALUES ('30', '100', 'tanuki', 'のの', '')"
        )
    mine_appellations(store)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO appellations (target_id, caller_id, appellation, count) "
            "VALUES (?, ?, ?, ?)",
            [("30", "888", "のの", 4), ("30", "777", "たま", 3)],
        )
    return store


# --- P1: person-first material selection ---------------------------------

def test_p1_person_summary_leads_the_payload_on_a_tight_budget(people_store):
    log = LogTools(people_store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "タヌキについて教えて", timeout=5, budget=400,
    ))
    assert result.diagnostics["people"] == ["30"]
    assert result.diagnostics["summary_delivered"] is True
    # On the old dense/context-first order the identity line was the
    # first thing the shared budget cut; now it leads the payload.
    assert str(result).startswith("人物情報")


def test_p1_persons_own_words_reach_the_payload(people_store):
    store = dense_store(people_store)
    log = LogTools(store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "タヌキについて教えて", timeout=5, budget=1200,
    ))
    assert result.diagnostics["people"] == ["30"]
    assert "tanukiの発言" in str(result)


# --- P2: grounding ---------------------------------------------------------

def test_p2_alias_snapshot_survives_a_fresh_process_via_disk(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([record(1), record(2, author="bob")])
    with store.connect() as db:
        first = _alias_snapshot(store, ["10"], db)
    assert first.rows
    cache_dir = tmp_path / "logs"
    files = list(cache_dir.glob("server_log.db.alias-snapshots/*.json"))
    assert len(files) == 1
    # A new store object on the same path (a restarted process) reads
    # the same rows back, validated by the alias watermark.
    store2 = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    with store2.connect() as db:
        second = _alias_snapshot(store2, ["10"], db)
    assert second.rows == first.rows
    # A new alias row moves the watermark: the disk copy is stale and
    # the rebuilt snapshot carries the new author.
    store2.upsert_many([record(3, author="carol")])
    with store2.connect() as db:
        third = _alias_snapshot(store2, ["10"], db)
    assert any(author == "22" or name == "carol" for name, author in third.rows)


def test_p2_match_global_uses_server_wide_evidence_with_strict_floors():
    index = AppellationIndex([
        ("30", "888", "さくら", 4),
        ("30", "777", "さくら", 3),
        ("31", "888", "さくら", 1),
        ("30", "888", "共有名", 3),
        ("31", "777", "共有名", 3),
        ("30", "888", "少数名", 2),
        ("30", "777", "少数名", 1),
    ])
    # さくら: 7 of 8 uses name person 30 — resolves with no caller
    # evidence at all.
    assert index.match_global("さくらについて教えて") == ["30"]
    # A genuinely shared name and a thin name stay unresolved.
    assert index.match_global("共有名について教えて") == []
    assert index.match_global("少数名について教えて") == []


def test_p2_appellation_question_delivers_the_whole_list(people_store):
    log = LogTools(people_store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "タヌキの呼ばれ方を教えて", timeout=5,
    ))
    assert result.diagnostics["people"] == ["30"]
    assert result.diagnostics["chunks"].get("appellations", 0) == 1
    assert "呼び名一覧" in str(result)
    assert "のの" in str(result)
    assert "タヌキ" in str(result)


def test_p2_unresolved_person_question_is_flagged_to_the_model(people_store):
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=people_store)
    asyncio.run(ai.reply(
        10, "ゼルダについて教えて", member_id=999,
        speaker_names={"username": "testuser", "display_name": "テストユーザー",
                       "server_nickname": None},
        message_id=1,
    ))
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert payload["person_resolution"]["status"] == "unresolved"


# --- P3: recent-context dedupe and carryover -------------------------------

def test_p3_transcript_turn_does_not_duplicate_recent_messages(people_store):
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=people_store)
    conversation = {
        "transcript": "[00:01] testuser: タヌキさん、それはどう？ 0",
        "reply_chain": [],
        "participants": [],
        "mentions": [],
    }
    asyncio.run(ai.reply(
        10, "タヌキについて教えて", member_id=999,
        conversation=conversation,
        speaker_names={"username": "testuser", "display_name": "テストユーザー",
                       "server_nickname": None},
        message_id=500,
    ))
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert "recent_messages" not in payload
    assert "conversation" in payload


def test_p3_fresh_question_does_not_inherit_the_previous_person(
        people_store, monkeypatch):
    calls = []

    class FakeResult:
        def __init__(self, people):
            self.diagnostics = {"people": people}
            self.sources = []

        def __str__(self):
            return "evidence"

    async def fake_pre_retrieve(log, content, **kwargs):
        calls.append(kwargs)
        return FakeResult(["30"] if len(calls) == 1 else [])

    monkeypatch.setattr("nezubot_discrod.assistant.pre_retrieve", fake_pre_retrieve)
    client, _sent = client_for(message("one"), message("two"), message("three"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=people_store)
    names = {"username": "testuser", "display_name": "テストユーザー", "server_nickname": None}
    asyncio.run(ai.reply(10, "タヌキについて教えて", member_id=999,
                         speaker_names=names, message_id=1))
    # A self-contained new question: no carryover.
    asyncio.run(ai.reply(10, "Rustの所有権について詳しく教えて", member_id=999,
                         speaker_names=names, message_id=2))
    assert calls[1]["carryover_person"] is None
    # A bare follow-up: the settled person carries over.
    asyncio.run(ai.reply(10, "どこ住み？", member_id=999,
                         speaker_names=names, message_id=3))
    assert calls[2]["carryover_person"] == "30"


# --- P4: server facts --------------------------------------------------------

def test_p4_server_fact_question_shapes():
    assert server_fact_question("このサーバーのオーナーはだれ？")
    assert server_fact_question("サーバーのオーナーは誰ですか")
    assert server_fact_question("メンバーは何人いるの？")
    assert server_fact_question("このサーバーいつ作られたの")
    # About the owner as a person, not the fact itself: not guarded.
    assert not server_fact_question("オーナーって最近何してる？")
    assert not server_fact_question("タヌキについて教えて")


def test_p4_owner_question_is_answered_from_server_info_only(people_store):
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=people_store)
    guild_info = {
        "name": "Example Developers",
        "owner": {"id": "60", "display_name": "sora"},
        "created_at": "2024-06-15",
        "member_count": 321,
    }
    asyncio.run(ai.reply(
        10, "このサーバーのオーナーはだれ？", member_id=999,
        guild_info=guild_info,
        speaker_names={"username": "testuser", "display_name": "テストユーザー",
                       "server_nickname": None},
        message_id=1,
    ))
    payload = json.loads(sent[0]["messages"][-1]["content"])
    assert payload["server_fact"]["owner"]["display_name"] == "sora"
    # No archive retrieval at all: no joke log can outvote the fact.
    assert "retrieved_server_context" not in payload
    # server_info precedes the speaker block in the payload.
    keys = list(payload)
    assert keys.index("server_info") < keys.index("speaker")


# --- P5: ambient pre-filter --------------------------------------------------

def test_p5_ambient_excludes_messages_addressed_to_others():
    # D-1 shape: a question whose only mention is another bot.
    assert ambient_excluded({"mentions": [{"id": "777", "name": "mizuki"}]})
    # D-3 shape: a 「どう」 addressed to a different member.
    assert ambient_excluded({"mentions": [{"id": "888", "name": "vex"}]})
    assert not ambient_excluded({"mentions": []})
    assert not ambient_excluded({})
    assert not ambient_excluded(None)


def test_p5_assistant_skips_without_a_model_call(people_store):
    client, sent = client_for(message("answer"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=people_store)
    result = asyncio.run(ai.reply(
        10, "mizukiそれどうやるの？", member_id=999, ambient=True,
        conversation={"mentions": [{"id": "777", "name": "mizuki"}]},
    ))
    assert result == "(skip)"
    assert sent == []


# --- P6: history compaction --------------------------------------------------

def test_p6_history_payload_compaction_stubs_evidence():
    payload = {
        "message": "タヌキについて教えて",
        "retrieved_server_context": {"excerpts": "証拠" * 500},
        "recent_messages": "直近" * 300,
        "member_wiki": "wiki" * 200,
        "conversation": {"transcript": "\n".join(f"line {i}" for i in range(20))},
    }
    compacted = json.loads(_compact_history_payload(json.dumps(payload, ensure_ascii=False)))
    assert "省略" in compacted["retrieved_server_context"]["excerpts"]
    assert "省略" in compacted["recent_messages"]
    assert len(compacted["member_wiki"]) <= 241
    assert compacted["conversation"]["transcript"].split("\n")[-1] == "line 19"
    assert len(compacted["conversation"]["transcript"].split("\n")) == 6
    assert compacted["message"] == "タヌキについて教えて"


def test_p6_compaction_leaves_plain_and_small_payloads_untouched():
    assert _compact_history_payload("ただの文章") == "ただの文章"
    small = json.dumps({"message": "hi"}, ensure_ascii=False)
    assert _compact_history_payload(small) == small


# --- P7: speaker-scoped conversation state -----------------------------------

def test_p7_facts_carry_their_author_and_state_records_the_speaker():
    conversation = {
        "reply_chain": [{
            "author": "テストユーザー", "author_id": "20", "message_id": "4",
            "content": "福岡ね",
        }],
        "participants": [{"id": "20", "name": "テストユーザー"}],
        "mentions": [],
    }
    state = build_state_update(
        None, now=1000,
        anchor={"id": "5", "author_id": "50", "content": "どこ住みかわかる？"},
        question="どこ住みかわかる？",
        conversation=conversation,
    )
    fact = state["established_facts"][0]
    assert fact["author_id"] == "20"
    assert fact["author_name"] == "テストユーザー"
    assert state["last_speaker_id"] == "50"


def _person_state(last_speaker="999", subject_id="30", label="タヌキ"):
    return {
        "topic": "タヌキの話",
        "active_subject": {"kind": "person", "id": subject_id, "label": label,
                           "author_id": None},
        "established_facts": [],
        "last_speaker_id": last_speaker,
    }


def test_p7_person_subject_does_not_cross_speakers():
    state = _person_state()
    # Same speaker: untouched.
    assert _state_for_turn(state, 999, "どこ住み？") is state
    # Another speaker who names the person: untouched.
    assert _state_for_turn(state, 50, "タヌキってどこ住み？") is state
    # Another speaker asking about themselves: the previous person's
    # subject is dropped from this turn's view (A-7).
    filtered = _state_for_turn(state, 50, "どこ住みかわかる？")
    assert filtered["active_subject"] is None
    assert filtered["topic"] == "タヌキの話"
    # The subject being the asker themselves is kept.
    own = _person_state(subject_id="50", label="Ren")
    assert _state_for_turn(own, 50, "どこ住み？") is own
