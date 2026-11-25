import asyncio
import time
from types import SimpleNamespace

import pytest
from test_assistant import client_for, message
from test_server_log import dense_store, record

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import (
    _assess_sufficiency,
    _prefetch_covers,
    pre_retrieve,
)
from nezubot_discrod.server_log_tools import LogTools


@pytest.fixture
def archive(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    for index in range(48):
        value = record(1, channel=str(10 + index % 3),
                       text=f"Rust history {index} " + "x" * index)
        value["id"] = str(index + 1)
        rows.append(value)
    store.upsert_many(rows)
    return store


def _hit(hit_id, author_id):
    return SimpleNamespace(id=str(hit_id), author_id=str(author_id))


def _chunk(*hits):
    return SimpleNamespace(hits=list(hits))


def test_sufficiency_target_evidence_for_settled_person():
    available = {
        "author": [_chunk(_hit(1, "20"), _hit(2, "20"), _hit(3, "21"))],
        "person_summary": [_chunk()],
    }
    verdict = _assess_sufficiency(available, ["20"])
    assert verdict["sufficient"] is True
    assert verdict["target_reached"] is True
    assert verdict["weak"] == []


def test_sufficiency_person_question_needs_the_summary_line():
    # The person's own messages arrived but the identity line did
    # not: not sufficient, and the one weak route is the summary —
    # once. A summary that has already been retried (and evidently
    # does not exist for this person) stops blocking sufficiency.
    available = {"author": [_chunk(_hit(1, "20"), _hit(2, "20"), _hit(3, "21"))]}
    verdict = _assess_sufficiency(available, ["20"])
    assert verdict["sufficient"] is False
    assert verdict["weak"] == ["person_summary"]
    verdict = _assess_sufficiency(available, ["20"], summary_retried=True)
    assert verdict["sufficient"] is True
    assert verdict["weak"] == []


def test_sufficiency_missing_target_marks_author_weak():
    available = {"dense": [_chunk(_hit(1, "21"), _hit(2, "21"))]}
    verdict = _assess_sufficiency(available, ["20"])
    assert verdict["sufficient"] is False
    assert "author" in verdict["weak"]


def test_sufficiency_multi_route_evidence_without_person():
    available = {
        "dense": [_chunk(_hit(1, "21"), _hit(2, "22"))],
        "lexical": [_chunk(_hit(3, "23"), _hit(4, "24"))],
    }
    verdict = _assess_sufficiency(available, [])
    assert verdict["sufficient"] is True


def test_sufficiency_grounding_weak_only_when_cut():
    verdict = _assess_sufficiency({}, [], grounding_cut=False)
    assert "grounding" not in verdict["weak"]
    verdict = _assess_sufficiency({}, [], grounding_cut=True)
    assert "grounding" in verdict["weak"]


def test_sufficiency_thin_dense_counts_as_missing_until_extended():
    available = {"dense": [_chunk(_hit(1, "21"))]}
    verdict = _assess_sufficiency(available, [])
    assert "dense" in verdict["weak"]
    verdict = _assess_sufficiency(available, [], dense_extended=True)
    assert "dense" not in verdict["weak"]


def test_prefetch_covers_needs_two_hits_by_the_settled_person():
    hits = [_hit(1, "20"), _hit(2, "20"), _hit(3, "21")]
    assert _prefetch_covers(hits, "20", ["20"]) is True
    assert _prefetch_covers(hits, "20", ["21"]) is False
    assert _prefetch_covers([_hit(1, "20")], "20", ["20"]) is False
    assert _prefetch_covers(hits, None, ["20"]) is False


def test_extension_retries_grounding_that_was_cut(archive, monkeypatch):
    calls = []

    def fake_resolve(store, channel_ids, text, caller_id, message_id=None,
                     seed=None, carryover=None, deadline=None):
        calls.append(1)
        if len(calls) == 1:
            # Simulate the first pass being interrupted by its own
            # budget: slow, and it settles on nobody.
            time.sleep(0.45)
            return []
        return ["20"]

    monkeypatch.setattr(
        "nezubot_discrod.server_log_context.resolve_people", fake_resolve)
    log = LogTools(archive, [10, 11, 12], caller_id=999)
    budgets = {"grounding_ms": 400, "lexical_ms": 400,
               "dense_ms": 500, "aux_ms": 250}
    # A name no archive token matches, so the first pass stays thin:
    # only the grounding retry can produce evidence.
    result = asyncio.run(pre_retrieve(
        log, "zeldaについて", timeout=5, budgets=budgets,
    ))
    assert result.diagnostics["people"] == ["20"]
    assert "grounding" in result.diagnostics["extended_routes"]
    assert result.diagnostics["sources"] >= 1
    assert "Speaker overview" in result


def test_extension_reruns_dense_discarded_by_first_pass_budget(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        record(1, text="質問 Rust"),
        record(2, text="semantic concepts", author="bob"),
        record(3, text="plain stuff"),
    ])
    store = dense_store(store)
    store.embed_pending()
    store.enable_dense().refresh_snapshot()
    log = LogTools(store, [10], caller_id=999)
    budgets = {"grounding_ms": 400, "lexical_ms": 400,
               "dense_ms": 0, "aux_ms": 0}
    result = asyncio.run(pre_retrieve(
        log, "質問についておしえて", timeout=5, budgets=budgets,
    ))
    assert result.diagnostics["extension_rounds"] >= 1
    assert "dense" in result.diagnostics["extended_routes"]
    assert "semantic concepts" in result


def test_prefetch_hits_are_delivered_and_counted(archive):
    hits = archive.search("Rust", channel_ids=[10, 11, 12])[:4]
    assert len(hits) == 4
    log = LogTools(archive, [10, 11, 12], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "これどう思う？", timeout=5, prefetch_hits=hits,
    ))
    assert result.diagnostics["prefetch_hits"] == 4
    assert result.diagnostics["chunks"].get("prefetch", 0) == 1
    assert "Rust history" in result


def test_assistant_prefetch_roundtrip(archive, monkeypatch):
    monkeypatch.setattr("nezubot_discrod.assistant.load_ok", lambda: True)
    client, _sent = client_for(message("ok"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive)
    ai._channel_scopes["10"] = (archive, ("10", "11", "12"))
    ai.conversation_states.update(
        "10",
        anchor={"id": "1", "author_id": "20", "author_name": "alice",
                "content": "Rust history"},
        question="Rust history",
        conversation={"reply_chain": [], "mentions": []},
        resolved_people=["20"],
    )
    asyncio.run(ai._prefetch_run("10"))
    entry = ai.prefetch_cache.get(("10", "11", "12"), ("person", "20"))
    assert entry is not None
    assert len(entry["hits"]) >= 1
    found = ai._prefetch_lookup(("10", "11", "12"), {"active_subject": {
        "kind": "person", "id": "20"}}, {})
    assert found is not None
    hits, person = found
    assert person == "20"
    assert hits == entry["hits"]
    # A different audience scope is a different cache entry.
    assert ai.prefetch_cache.get(("10",), ("person", "20")) is None
