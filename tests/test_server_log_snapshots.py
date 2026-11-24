from threading import Event
from time import monotonic

import pytest
from test_server_log import dense_store, record

from nezubot_discrod.server_log import DenseSnapshotUnavailable, ServerLog


@pytest.fixture
def store(tmp_path):
    store = dense_store(ServerLog(tmp_path / "archive" / "server_log.db"))
    store.upsert_many([record(1, text="semantic original"), record(2, text="Rust original")])
    store.embed_pending()
    store._dense.refresh_snapshot()
    return store


def hits(store, snapshot=None):
    with store.connect() as db:
        db.execute("BEGIN")
        where, args = store._scope([10])
        if snapshot is not None:
            return snapshot.search(db, "質問", [10], where, args, 10, synchronize=False)
        return store._dense.search_snapshot(db, "質問", [10], where, args, 10)


def test_current_snapshot_search_does_not_wait_for_background_refresh(store, monkeypatch):
    dense = store._dense
    original = dense._sync
    entered, release = Event(), Event()

    def slow(db):
        entered.set()
        assert release.wait(3)
        return original(db)

    store.upsert(record(3, text="new unembedded message"))
    monkeypatch.setattr(dense, "_sync", slow)
    dense.request_refresh()
    try:
        assert entered.wait(1)
        started = monotonic()
        result = hits(store)
        assert set(result) == {"1", "2"} and result.partial and result.coverage is None
        assert monotonic() - started < 0.3
        store.mark_deleted("1")
        store.upsert(record(2, text="edited without a vector"))
        assert hits(store) == []
    finally:
        release.set()
        dense.refresh_thread.join(3)
    assert not dense.refresh_thread.is_alive()


@pytest.mark.parametrize("change", [
    "deleted", "tombstone", "edited", "removed", "replaced", "corrupt", "moved",
])
def test_snapshot_revalidates_current_vectors_and_scope(store, monkeypatch, change):
    dense = store._dense
    monkeypatch.setattr(dense, "request_refresh", lambda: None)
    if change == "deleted":
        store.mark_deleted("1")
    elif change == "edited":
        store.upsert(record(1, text="edited content"))
    elif change == "moved":
        store.upsert(record(1, channel="99", text="semantic original"))
    else:
        with store.connect() as db:
            if change == "tombstone":
                db.execute("INSERT INTO deleted_messages VALUES ('1')")
            elif change == "removed":
                db.execute("DELETE FROM message_vectors WHERE message_id='1'")
            else:
                vector = (b"invalid" if change == "corrupt" else
                          dense.np.asarray([1, 0], dtype="<f2").tobytes())
                db.execute("UPDATE message_vectors SET vector=? WHERE message_id='1'", (vector,))
    assert hits(store) == ["2"]


def test_refresh_appends_immutable_versions_and_publishes_after_explicit_search(store, monkeypatch):
    dense = store._dense
    monkeypatch.setattr(dense, "request_refresh", lambda: None)
    previous = dense.snapshot
    before = previous.matrix[:previous.size].copy()
    store.upsert(record(1, text="Rust edited"))
    store.embed_pending()
    store.search("質問", channel_ids=[10])
    assert dense.snapshot is previous
    dense.refresh_snapshot()
    assert dense.snapshot is not previous and dense.snapshot.snapshot is None
    assert dense.np.array_equal(previous.matrix[:previous.size], before)
    assert hits(store, previous) == ["2"]
    assert set(hits(store)) == {"1", "2"}


def test_snapshot_fusion_retains_unembedded_lexical_evidence(store, monkeypatch):
    monkeypatch.setattr(store._dense, "request_refresh", lambda: None)
    store.upsert(record(3, text="unembedded exact"))
    result = store.search("unembedded", channel_ids=[10], snapshot_only=True)
    assert "3" in {hit.id for hit in result}


def test_require_dense_snapshot_probe_skips_fts_without_snapshot(tmp_path, monkeypatch):
    store = dense_store(ServerLog(tmp_path / "archive" / "server_log.db"))
    store.upsert(record(1, text="Rust original"))
    store.enable_dense()  # no refresh_snapshot: nothing is published
    calls = 0
    original = ServerLog._lexical_query

    def counting(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(ServerLog, "_lexical_query", staticmethod(counting))
    with pytest.raises(DenseSnapshotUnavailable):
        store.search("Rust", channel_ids=[10], require_dense=True, snapshot_only=True)
    assert calls == 0


def test_empty_snapshot_and_failed_refresh_do_not_discard_published_snapshot(tmp_path, monkeypatch):
    store = dense_store(ServerLog(tmp_path / "archive" / "server_log.db"))
    dense = store.enable_dense()
    monkeypatch.setattr(dense, "request_refresh", lambda: None)
    assert hits(store) is None
    dense.refresh_snapshot()
    assert hits(store) is None
    store.upsert(record(1, text="semantic"))
    store.embed_pending()
    dense.refresh_snapshot()
    previous = dense.snapshot
    store.upsert(record(2, text="new pending message"))

    def failed(db):
        raise ValueError("invalid vector")

    monkeypatch.setattr(dense, "_sync", failed)
    with pytest.raises(ValueError, match="invalid vector"):
        dense.refresh_snapshot()
    assert dense.snapshot is previous and hits(store) == ["1"]


def test_disk_cache_restores_snapshot_without_rebuild(tmp_path):
    path = tmp_path / "archive" / "server_log.db"
    store = dense_store(ServerLog(path))
    store.upsert_many([record(1, text="semantic original"), record(2, text="Rust original")])
    store.embed_pending()
    store._dense.refresh_snapshot()
    cache = path.parent / (path.name + ".dense-cache")
    assert (cache / "meta.json").exists()

    store2 = dense_store(ServerLog(path))
    dense2 = store2.enable_dense()
    with store2.connect() as db:
        db.execute("BEGIN")
        assert dense2._sync(db) is True
    # Restored from disk: no full rebuild was needed.
    assert dense2.rebuilds == 0 and dense2.size == 2
    dense2.refresh_snapshot()
    with store2.connect() as db:
        db.execute("BEGIN")
        where, args = store2._scope([10])
        result = dense2.search_snapshot(db, "質問", [10], where, args, 10)
    assert result is not None and set(result) == {"1", "2"}


def test_disk_cache_corruption_falls_back_to_rebuild(tmp_path):
    path = tmp_path / "archive" / "server_log.db"
    store = dense_store(ServerLog(path))
    store.upsert_many([record(1, text="semantic original"), record(2, text="Rust original")])
    store.embed_pending()
    store._dense.refresh_snapshot()
    matrix = path.parent / (path.name + ".dense-cache") / "matrix.npy"
    data = bytearray(matrix.read_bytes())
    data[-1] ^= 0xFF
    matrix.write_bytes(bytes(data))

    store2 = dense_store(ServerLog(path))
    dense2 = store2.enable_dense()
    with store2.connect() as db:
        db.execute("BEGIN")
        assert dense2._sync(db) is True
    assert dense2.rebuilds == 1 and dense2.size == 2
