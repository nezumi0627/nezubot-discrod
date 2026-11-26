"""v1.29.0 Phase-2 restructure: candidate pool / aux always-on +
instrumentation, contextual (ctx) embedding in the worker, and
reply-graph expansion in pre_retrieve."""
import asyncio
from threading import Event

import pytest
from test_server_log import dense_store, record

from nezubot_discrod.server_log import SearchHits, ServerLog, embedding_text, text_hash
from nezubot_discrod.server_log_context import pre_retrieve
from nezubot_discrod.server_log_tools import LogTools

# ---- contextual embedding (server_log_dense) -------------------------------

class CapturingEmbedder:
    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        for text in texts:
            self.inputs.append(text)
            yield [1.0, 0.0]


def _rec(number, *, author="alice", text="hello world", reply_to=None,
         channel="10"):
    value = record(number, channel=channel, author=author, text=text)
    if reply_to is not None:
        value["message_reference"] = {"message_id": str(reply_to)}
    return value


def test_contextual_embed_pending_uses_parent_text_and_raw_hash(tmp_path):
    pytest.importorskip("numpy")
    embedder = CapturingEmbedder()
    store = ServerLog(
        tmp_path / "ctx" / "server_log.db",
        model="cl-nagoya/ruri-v3-70m-ctx",
        embedder=embedder,
        guild_id="100",
    )
    parent_text = "p" * 150
    store.upsert_many([
        _rec(1, author="alice", text=parent_text),
        _rec(2, author="bob", text="child reply text", reply_to=1),
        _rec(3, author="alice", text="standalone message"),
        _rec(4, author="bob", text="orphan reply text", reply_to=999),
    ])
    dense = store.enable_dense()
    assert dense.contextual is True
    assert dense.embed_pending(32, None, Event()) == 4

    # The child is encoded with its parent's (truncated) text prepended,
    # in exactly the stage-3 format; the ruri document prefix is added
    # by the encoder as usual.
    expected_child = (
        "検索文書: "
        f"alice: {parent_text[:100]}\nbob: child reply text"
    )
    assert expected_child in embedder.inputs
    # Parent-less and unresolvable-parent rows fall back to the plain
    # embedding_text, unchanged.
    assert "検索文書: alice: standalone message" in embedder.inputs
    assert "検索文書: bob: orphan reply text" in embedder.inputs
    assert f"検索文書: alice: {parent_text}" in embedder.inputs

    # Stored hashes stay on the raw-text convention: text_hash(row) of
    # the message itself, not of the contextual encoding text.
    with store.connect() as db:
        rows = {
            row["id"]: row
            for row in db.execute("SELECT * FROM messages").fetchall()
        }
        vectors = {
            row["message_id"]: row["text_hash"]
            for row in db.execute(
                "SELECT * FROM message_vectors WHERE model = ?",
                ("cl-nagoya/ruri-v3-70m-ctx",),
            ).fetchall()
        }
    assert set(vectors) == {"1", "2", "3", "4"}
    for message_id, stored_hash in vectors.items():
        assert stored_hash == text_hash(rows[message_id])
    assert vectors["2"] != __import__("hashlib").sha256(
        expected_child.removeprefix("検索文書: ").encode()
    ).hexdigest()
    # And embedding_text itself is untouched.
    assert embedding_text(rows["2"]) == "bob: child reply text"


def test_non_contextual_index_is_not_contextual(tmp_path):
    pytest.importorskip("numpy")
    store = ServerLog(
        tmp_path / "plain" / "server_log.db",
        model="cl-nagoya/ruri-v3-70m",
        embedder=CapturingEmbedder(),
        guild_id="100",
    )
    assert store.enable_dense().contextual is False


# ---- aux instrumentation on the fusion path (server_log) -------------------

def _hybrid_store(tmp_path):
    from test_server_log_hybrid import FtEmbedder, TeacherEmbedder, hybrid_store

    store = hybrid_store(tmp_path, teacher=TeacherEmbedder(), ft=FtEmbedder())
    store.upsert_many([
        record(1, text="semantic topic"),
        record(2, text="hybrid topic"),
        record(3, text="plain topic"),
    ])
    store.embed_pending()
    store.embed_pending_hybrid()
    store.enable_dense()
    return store


def test_search_fusion_result_carries_aux_attributes(tmp_path):
    store = _hybrid_store(tmp_path)
    hits = store.search("semantic hybrid", channel_ids=[10], limit=10)
    assert isinstance(hits, SearchHits)
    assert hits.aux_fired is True
    # Every fused hit came through the aux list in this fixture.
    assert hits.aux_hits >= 1
    assert hits.aux_hits == len({hit.id for hit in hits} & {"1", "2", "3"})

    gated = store.search(
        "semantic hybrid", channel_ids=[10], limit=10, allow_hybrid=False,
    )
    assert isinstance(gated, SearchHits)
    assert gated.aux_fired is False
    assert gated.aux_hits == 0


def test_search_early_return_paths_are_plain_lists(tmp_path):
    store = _hybrid_store(tmp_path)
    lexical = store.search("semantic", channel_ids=[10], lexical_only=True)
    assert type(lexical) is list
    assert getattr(lexical, "aux_fired", False) is False
    assert getattr(lexical, "aux_hits", 0) == 0


# ---- pre_retrieve: aux gate modes ------------------------------------------

def _person_archive(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    for index in range(12):
        value = record(1, channel="10", text=f"Rust history {index}")
        value["id"] = str(index + 1)
        value["timestamp"] = f"2025-01-{(index % 28) + 1:02d}T00:00:00Z"
        rows.append(value)
    store.upsert_many(rows)
    store = dense_store(store)
    store.embed_pending()
    store.enable_dense().refresh_snapshot()
    return store


def test_pre_retrieve_aux_gate_defaults_to_always(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVER_LOG_AUX_GATE", raising=False)
    store = _person_archive(tmp_path)
    log = LogTools(store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(log, "aliceについて", timeout=5))
    assert result.diagnostics["people"] == ["20"]
    assert result.diagnostics["aux_gate_mode"] == "always"
    # The person resolved, yet the gate stayed open.
    assert result.diagnostics["aux_gate_allowed"] is True


def test_pre_retrieve_aux_gate_grounding_restores_old_behaviour(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("SERVER_LOG_AUX_GATE", "grounding")
    store = _person_archive(tmp_path)
    log = LogTools(store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(log, "aliceについて", timeout=5))
    assert result.diagnostics["people"] == ["20"]
    assert result.diagnostics["aux_gate_mode"] == "grounding"
    assert result.diagnostics["aux_gate_allowed"] is False


# ---- pre_retrieve: reply graph ---------------------------------------------

def _timed(value, message_id, *, day=1, hour=0, minute=0):
    total = hour * 60 + minute
    day += total // 1440
    hour, minute = (total % 1440) // 60, total % 60
    value["id"] = str(message_id)
    value["timestamp"] = (
        f"2026-01-{day:02d}T{hour:02d}:{minute:02d}:00+00:00"
    )
    return value


def _graph_store(tmp_path, *, children=3, name="graph"):
    store = ServerLog(tmp_path / name / "server_log.db", guild_id="100")
    rows = []
    # Parent: plain text, so it is not itself a search seed.
    rows.append(_timed(
        _rec(1, author="alice", text="plain parent context"), 100, minute=0))
    # Seed: the semantic hit the graph expands from.
    rows.append(_timed(
        _rec(1, author="bob", text="semantic seed topic", reply_to=100),
        101, minute=1))
    for index in range(children):
        rows.append(_timed(
            _rec(1, author="alice", text=f"plain child reply {index}",
                 reply_to=101),
            102 + index, minute=2 + index))
    # A child in a channel outside the caller's scope.
    rows.append(_timed(
        _rec(1, author="alice", text="plain child elsewhere", reply_to=101,
             channel="11"),
        200, minute=10))
    # A deleted child of the seed.
    rows.append(_timed(
        _rec(1, author="alice", text="plain deleted child", reply_to=101),
        201, minute=11))
    # The question message itself, as a child of the seed.
    rows.append(_timed(
        _rec(1, author="alice", text="plain question child", reply_to=101),
        202, minute=12))
    # Semantic fillers push every plain message out of the seed top-30.
    for index in range(35):
        rows.append(_timed(
            _rec(1, author="alice", text=f"semantic filler {index}"),
            300 + index, day=2, minute=index))
    store.upsert_many(rows)
    store.mark_deleted("201")
    store = dense_store(store)
    store.embed_pending()
    store.enable_dense().refresh_snapshot()
    return store


def test_reply_graph_delivers_parent_and_children_with_scope(tmp_path, monkeypatch):
    monkeypatch.delenv("SERVER_LOG_REPLY_GRAPH", raising=False)
    store = _graph_store(tmp_path)
    log = LogTools(store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "semantic seed topic", message_id=202, timeout=5))
    assert result.diagnostics["reply_graph_seeds"] == 30
    assert result.diagnostics["reply_graph_neighbors"] == 4
    assert result.diagnostics["reply_graph_delivered"] is True
    assert result.diagnostics["chunks"].get("reply_graph", 0) >= 1
    assert "Reply graph expansion" in result
    assert "plain parent context" in result
    for index in range(3):
        assert f"plain child reply {index}" in result
    # Scope, tombstone and self exclusions all hold.
    assert "plain child elsewhere" not in result
    assert "plain deleted child" not in result
    assert "plain question child" not in result


def test_reply_graph_capped_at_fifty(tmp_path):
    store = _graph_store(tmp_path, children=60, name="graphcap")
    log = LogTools(store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "semantic seed topic", message_id=202, timeout=5))
    # 1 parent + 60 children would be 61; the expansion caps at +50.
    assert result.diagnostics["reply_graph_neighbors"] == 50


def test_reply_graph_can_be_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVER_LOG_REPLY_GRAPH", "off")
    store = _graph_store(tmp_path, name="graphoff")
    log = LogTools(store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(
        log, "semantic seed topic", message_id=202, timeout=5))
    assert result.diagnostics["reply_graph_seeds"] == 0
    assert result.diagnostics["reply_graph_neighbors"] == 0
    assert result.diagnostics["reply_graph_delivered"] is False
    assert "plain parent context" not in result
