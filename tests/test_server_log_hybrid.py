"""Hybrid (three-path) retrieval: fusion math, the aux dense index,
fallback equivalence, and hybrid embedding of pending messages."""
import pytest

from nezubot_discrod.server_log import ServerLog, _fusion, rrf
from nezubot_discrod.server_log_dense import embedding_prefix


def record(number=1, *, channel="10", author="alice", text="Rust 検索エンジン",
           bot=False):
    return {
        "id": str(number),
        "guild_id": "100",
        "channel_id": channel,
        "_channel": "general" if channel == "10" else "private",
        "_parent": "development",
        "author": {"id": "20" if author == "alice" else "21",
                   "username": author, "bot": bot},
        "timestamp": f"2026-01-01T00:{number:02d}:00+00:00",
        "content": text,
    }


class TeacherEmbedder:
    """Query/passage vectors keyed on 'semantic'."""

    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        for text in texts:
            self.inputs.append(text)
            yield [0.0, 1.0] if "semantic" in text else [1.0, 0.0]


class FtEmbedder:
    """Query/passage vectors keyed on 'hybrid' (disjoint from the teacher)."""

    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        for text in texts:
            self.inputs.append(text)
            yield [0.0, 1.0] if "hybrid" in text else [1.0, 0.0]


def hybrid_store(tmp_path, *, teacher=None, ft=None, name="data"):
    pytest.importorskip("numpy")
    return ServerLog(
        tmp_path / name / "server_log.db",
        model="intfloat/multilingual-e5-large",
        embedder=teacher or TeacherEmbedder(),
        hybrid_model="ft-e5-test",
        hybrid_embedder=ft or FtEmbedder(),
    )


def _row(message_id, text, author="alice"):
    return {"id": message_id, "author_name": author, "text": text}


# ---- _fusion ---------------------------------------------------------------

def test_fusion_three_way_matches_rrf_with_head():
    lexical = [_row("a", "probe alpha"), _row("b", "probe beta")]
    dense = ["c", "a", "d"]
    extra = ["d", "b", "e"]
    result = _fusion("probe", lexical, dense, 50, extra_dense=extra)
    tail = rrf([["a", "b"], dense, extra], 50)
    # Head: lexical anchor (a's text covers the query and contains it),
    # then the teacher top-1 (c) — the ft top-1 never enters the head.
    expected = ["a", "c"] + [item for item in tail if item not in {"a", "c"}]
    assert result == expected
    assert set(result) == {"a", "b", "c", "d", "e"}


def test_fusion_extra_none_is_legacy_result():
    lexical = [_row("a", "probe alpha"), _row("b", "probe beta")]
    dense = ["c", "a", "d"]
    legacy = _fusion("probe", lexical, dense, 50)
    assert _fusion("probe", lexical, dense, 50, extra_dense=None) == legacy
    assert _fusion("probe", lexical, dense, 50, extra_dense=[]) == legacy
    tail = rrf([["a", "b"], dense], 50)
    assert legacy == ["a", "c"] + [i for i in tail if i not in {"a", "c"}]


def test_fusion_partial_branch_ignores_extra():
    lexical = [_row(str(i), f"text {i}") for i in range(8)]
    dense = ["100", "101"]
    extra = ["200", "201"]
    result = _fusion("q", lexical, dense, 50, partial=True,
                     dense_coverage=0.5, extra_dense=extra)
    assert "200" not in result and "201" not in result
    assert result == _fusion("q", lexical, dense, 50, partial=True,
                              dense_coverage=0.5)


# ---- aux dense index ---------------------------------------------------------

def _seeded_store(tmp_path):
    store = hybrid_store(tmp_path)
    store.upsert_many([
        record(1, text="semantic topic"),
        record(2, text="hybrid topic"),
        record(3, text="plain topic"),
        record(4, channel="11", text="hybrid elsewhere"),
    ])
    assert store.embed_pending() == 4
    assert store.embed_pending_hybrid() == 4
    assert store.embed_pending_hybrid() == 0
    return store


def test_aux_search_ranks_by_ft_scores_within_scope(tmp_path):
    store = _seeded_store(tmp_path)
    index = store.enable_dense()
    with store.connect() as db:
        teacher = index.search(db, "semantic hybrid", [10],
                               "m.channel_id IN (?)", ["10"], 10)
        aux = index.search_aux(db, "semantic hybrid", [10],
                               "m.channel_id IN (?)", ["10"], 10)
    assert teacher[0] == "1"
    # The ft path ranks the ft-matching message first and respects scope
    # (message 4 matches ft but lives in channel 11).
    assert list(aux) == ["2", "1", "3"]


def test_aux_search_snapshot_mode(tmp_path):
    store = _seeded_store(tmp_path)
    index = store.enable_dense()
    index.refresh_snapshot()
    with store.connect() as db:
        aux = index.search_aux_snapshot(db, "semantic hybrid", [10],
                                        "m.channel_id IN (?)", ["10"], 10)
    assert aux is not None and aux[0] == "2"


def test_aux_unavailable_without_vectors_or_config(tmp_path):
    store = hybrid_store(tmp_path)
    store.upsert_many([record(1, text="semantic topic")])
    assert store.embed_pending() == 1  # teacher only; no ft vectors stored
    index = store.enable_dense()
    with store.connect() as db:
        assert index.search_aux(db, "semantic hybrid", [10],
                                "m.channel_id IN (?)", ["10"], 10) is None

    plain = ServerLog(tmp_path / "plain" / "server_log.db",
                      model="intfloat/multilingual-e5-large",
                      embedder=TeacherEmbedder())
    plain.upsert_many([record(1, text="semantic topic")])
    assert plain.embed_pending() == 1
    assert plain.embed_pending_hybrid() == 0  # not configured
    plain_index = plain.enable_dense()
    assert plain_index.aux_model is None
    with plain.connect() as db:
        assert plain_index.search_aux(db, "semantic", [10],
                                      "m.channel_id IN (?)", ["10"], 10) is None


# ---- end-to-end search wiring -------------------------------------------------

def test_search_uses_ft_encoder_and_falls_back_when_it_breaks(tmp_path):
    teacher, ft = TeacherEmbedder(), FtEmbedder()
    store = hybrid_store(tmp_path, teacher=teacher, ft=ft)
    store.upsert_many([
        record(1, text="semantic topic"),
        record(2, text="hybrid topic"),
        record(3, text="plain topic"),
    ])
    store.embed_pending()
    store.embed_pending_hybrid()
    store.enable_dense()
    hits = store.search("semantic hybrid", channel_ids=[10], limit=10)
    assert {hit.id for hit in hits} == {"1", "2", "3"}
    # The ft query was encoded by the ft embedder with the e5 query prefix.
    assert "query: semantic hybrid" in ft.inputs

    class BrokenFt(FtEmbedder):
        def embed(self, texts):
            raise RuntimeError("ft encoder down")

    broken = hybrid_store(tmp_path, ft=BrokenFt(), name="broken")
    broken.upsert_many([
        record(1, text="semantic topic"),
        record(2, text="hybrid topic"),
        record(3, text="plain topic"),
    ])
    broken.embed_pending()
    broken.embed_pending_hybrid()
    broken.enable_dense()
    fallback_hits = broken.search("semantic hybrid", channel_ids=[10], limit=10)

    plain = ServerLog(tmp_path / "plain" / "server_log.db",
                      model="intfloat/multilingual-e5-large",
                      embedder=TeacherEmbedder())
    plain.upsert_many([
        record(1, text="semantic topic"),
        record(2, text="hybrid topic"),
        record(3, text="plain topic"),
    ])
    plain.embed_pending()
    plain.enable_dense()
    plain_hits = plain.search("semantic hybrid", channel_ids=[10], limit=10)
    assert [hit.id for hit in fallback_hits] == [hit.id for hit in plain_hits]


def test_aux_skipped_when_search_is_author_scoped(tmp_path):
    teacher, ft = TeacherEmbedder(), FtEmbedder()
    store = hybrid_store(tmp_path, teacher=teacher, ft=ft)
    store.upsert_many([
        record(1, text="semantic lesson"),
        record(2, text="hybrid discovery", author="bob"),
    ])
    assert store.embed_pending() == 2
    assert store.embed_pending_hybrid() == 2
    store.enable_dense().refresh_snapshot()
    ft.inputs.clear()
    hits = store.search("hybrid", channel_ids=[10], author="alice",
                        require_dense=True, snapshot_only=True)
    assert hits and ft.inputs == []
    hits = store.search("hybrid", channel_ids=[10],
                        require_dense=True, snapshot_only=True)
    assert hits and any("hybrid" in text for text in ft.inputs)
    ft.inputs.clear()
    store.search("hybrid", channel_ids=[10], require_dense=True,
                 snapshot_only=True, allow_hybrid=False)
    assert ft.inputs == []


# ---- embedding prefixes per model lineage -------------------------------------

def test_embedding_prefix_lineages():
    # e5 keeps its historical instructions on both sides.
    assert embedding_prefix("intfloat/multilingual-e5-small", query=True) == "query: "
    assert embedding_prefix("intfloat/multilingual-e5-small") == "passage: "
    assert embedding_prefix("intfloat/multilingual-e5-small-ft1", query=True) == "query: "
    assert embedding_prefix("intfloat/multilingual-e5-small-ft1") == "passage: "
    assert embedding_prefix("intfloat/multilingual-e5-small-static") == "passage: "
    # ruri-v3 uses its own Japanese instructions.
    assert embedding_prefix("cl-nagoya/ruri-v3-70m", query=True) == "検索クエリ: "
    assert embedding_prefix("cl-nagoya/ruri-v3-70m") == "検索文書: "
    assert embedding_prefix("cl-nagoya/ruri-v3-130m", query=True) == "検索クエリ: "
    # Unknown lineages get no prefix: never guess an instruction format.
    assert embedding_prefix("some/other-model", query=True) == ""
    assert embedding_prefix("some/other-model") == ""
    assert embedding_prefix(None, query=True) == ""
    assert embedding_prefix("") == ""


def test_e5_hybrid_document_prefix_unchanged(tmp_path):
    teacher, ft = TeacherEmbedder(), FtEmbedder()
    store = hybrid_store(tmp_path, teacher=teacher, ft=ft)
    store.upsert_many([record(1, text="semantic topic")])
    store.embed_pending()
    assert store.embed_pending_hybrid() == 1
    assert "passage: alice: semantic topic" in ft.inputs


def test_ruri_hybrid_prefixes_on_documents_and_queries(tmp_path):
    teacher, ruri = TeacherEmbedder(), FtEmbedder()
    store = ServerLog(
        tmp_path / "ruri" / "server_log.db",
        model="intfloat/multilingual-e5-large",
        embedder=teacher,
        hybrid_model="cl-nagoya/ruri-v3-70m",
        hybrid_embedder=ruri,
    )
    store.upsert_many([
        record(1, text="semantic topic"),
        record(2, text="hybrid topic"),
    ])
    store.embed_pending()
    assert store.embed_pending_hybrid() == 2
    # Documents are embedded under the ruri document instruction, over
    # the same "{author}: {text}" body the stored production vectors use.
    assert "検索文書: alice: semantic topic" in ruri.inputs
    assert "検索文書: alice: hybrid topic" in ruri.inputs
    store.enable_dense()
    ruri.inputs.clear()
    store.search("semantic hybrid", channel_ids=[10], limit=10)
    # Queries go through the ruri query instruction, not the e5 one.
    assert "検索クエリ: semantic hybrid" in ruri.inputs
    assert not any(text.startswith("query: ") for text in ruri.inputs)
