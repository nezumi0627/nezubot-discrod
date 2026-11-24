import json
import os
import random
import sqlite3
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from nezubot_discrod.config import Settings
from nezubot_discrod.server_log import (
    LEXICAL_MEMO,
    LogHit,
    ServerLog,
    _fusion,
    bigrams,
    format_hits,
    main,
    message_record,
    new_lexical_memo,
    record_text,
    rrf,
    text_hash,
)
from nezubot_discrod.server_log_dense import DenseHits, DenseIndex, embeddable_text
from nezubot_discrod.server_log_tools import LogTools
from nezubot_discrod.server_log_vectors import export_vectors, import_vectors


def record(number=1, *, channel="10", author="alice", text="Rust 検索エンジン", bot=False):
    return {
        "id": str(number),
        "guild_id": "100",
        "channel_id": channel,
        "_channel": "general" if channel == "10" else "private",
        "_parent": "development",
        "author": {"id": "20" if author == "alice" else "21", "username": author, "bot": bot},
        "timestamp": f"2026-01-01T00:{number:02d}:00+00:00",
        "content": text,
    }


@pytest.fixture
def store(tmp_path):
    return ServerLog(tmp_path / "data" / "server_log.db")


def test_upsert_edit_delete_and_cursor_persist(store):
    value = record()
    value.update(
        {
            "embeds": [{"title": "title", "description": "description"}],
            "attachments": [{"filename": "example.py"}],
            "sticker_items": [{"name": "wave"}],
            "message_reference": {"message_id": "42"},
        }
    )
    store.upsert(value)
    store.upsert(value)
    (hit,) = store.search("検索", channel_ids=[10])
    assert hit.id == "1" and hit.author == "alice" and hit.channel_id == "10"
    assert hit.url == "https://discord.com/channels/100/10/1"
    assert hit.text == "Rust 検索エンジン\ntitle\ndescription\n[file] example.py\n[sticker] wave"
    with store.connect() as db:
        (row,) = db.execute("SELECT * FROM messages").fetchall()
        assert row["parent_name"] == "development" and row["reply_to"] == "42"
        assert db.execute("SELECT count(*) FROM message_fts").fetchone()[0] == 1
    value.update(
        content="Python 非同期処理",
        embeds=[],
        attachments=[],
        sticker_items=[],
        edited_timestamp="2026-01-02T01:00:00Z",
    )
    store.upsert(value)
    assert store.search("Rust", channel_ids=[10]) == []
    assert store.search("Python", channel_ids=[10])[0].text == "Python 非同期処理"
    store.set_cursor(10, "99")
    assert ServerLog(store.path).get_cursor("10") == "99"
    store.set_cursor(10, "98")
    assert store.get_cursor(10) == "98"
    assert store.get_cursor(11) is None
    store.mark_deleted("1")
    assert store.search("Python", channel_ids=[10]) == []
    assert store.read("1", channel_ids=[10]) == []
    assert store.recent(10, 5) == []
    store.upsert(value)
    assert store.recent(10, 5) == []
    store.mark_deleted("2")
    store.upsert(record(2))
    assert store.recent(10, 5) == []
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.path.parent.stat().st_mode & 0o777 == 0o700
    store.upsert_many([record(7), record(8)], cursor_channel_id=10)
    assert store.get_cursor(10) == "8"
    store.upsert_many([], cursor_channel_id=10)
    assert store.get_cursor(10) == "8"


def test_scoping_and_filters(store):
    store.upsert_many([record(1), record(2, author="bob"), record(3, channel="11")])
    assert store.search("Rust", channel_ids=[]) == []
    assert store.search("Rust", channel_ids=[10], channel="11") == []
    assert {h.id for h in store.search("Rust", channel_ids=[10])} == {"1", "2"}
    assert [h.id for h in store.search("Rust", channel_ids=[10], author="bob")] == ["2"]
    assert [h.id for h in store.search("Rust", channel_ids=[10], author=20)] == ["1"]
    assert len(store.search("Rust", channel_ids=[10, 11], channel="#private")) == 1
    assert [
        h.id
        for h in store.search(
            "Rust",
            channel_ids=[10, 11],
            after="2026-01-01T00:01:00Z",
            before="2026-01-01T00:03:00+00:00",
        )
    ] == ["2"]
    assert store.search("Rust", channel_ids=[10], author="' OR 1=1 --") == []
    assert store.read("3", channel_ids=[10]) == []
    assert [h.id for h in store.recent(10, 1)] == ["2"]
    assert store.search("", channel_ids=[10]) == []
    assert store.search("😀", channel_ids=[10]) == []
    assert store.search('"Rust" OR NEAR(foo)', channel_ids=[10])
    assert store.search("Rust", channel_ids=[10], limit=0) == []
    with pytest.raises(ValueError):
        store.search("Rust", channel_ids=[10], after="not a date")


def test_read_chronology_and_boundaries(store):
    store.upsert_many([record(i) for i in range(1, 7)] + [record(7, channel="11")])
    store.mark_deleted("2")
    assert [h.id for h in store.read("4", channel_ids=[10], limit=3)] == ["3", "4", "5"]
    assert [h.id for h in store.read("4", channel_ids=[10], direction="before", limit=2)] == [
        "1",
        "3",
    ]
    assert [h.id for h in store.read("4", channel_ids=[10], direction="after", limit=2)] == [
        "5",
        "6",
    ]
    assert [h.id for h in store.read("1", channel_ids=[10, 11], limit=3)] == ["1", "3", "4"]
    assert [h.id for h in store.read("6", channel_ids=[10], limit=3)] == ["4", "5", "6"]
    assert [h.id for h in store.read("3", channel_ids=[10], limit=1)] == ["3"]
    assert store.read("missing", channel_ids=[10]) == []
    assert store.read("1", channel_ids=[]) == []
    assert store.read("1", channel_ids=[10], limit=0) == []
    with pytest.raises(ValueError, match="direction"):
        store.read("1", channel_ids=[10], direction="other")


def test_tokenization_and_format(store):
    assert bigrams("ＡＢＣ 日本語 a_b １２") == ["abc", "日本", "本語", "a_b", "12"]
    assert bigrams("日") == ["日"]
    assert bigrams("😀") == []
    assert record_text({}) == ""
    store.upsert(record(text="Rust\n" + "長" * 400))
    rendered = format_hits(store.recent(10, 1))
    assert rendered.startswith("1. 2026-01-01T00:01:00+00:00 #general alice (id: 20): Rust ")
    assert "... https://discord.com/channels/100/10/1" in rendered
    assert "長" * 301 not in rendered
    assert "\n" not in rendered
    assert format_hits([]) == "No messages found."


def test_lexical_query_is_shared_within_a_memo_context(store, monkeypatch):
    store.upsert(record(text="Rust 検索エンジン"))
    calls = 0
    original = ServerLog._lexical_query

    def counting(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(ServerLog, "_lexical_query", staticmethod(counting))
    first = store.search("Rust", channel_ids=[10], lexical_only=True)
    second = store.search("Rust", channel_ids=[10], lexical_only=True)
    assert [hit.id for hit in first] == [hit.id for hit in second] == ["1"]
    assert calls == 2
    calls = 0
    token = LEXICAL_MEMO.set(new_lexical_memo())
    try:
        first = store.search("Rust", channel_ids=[10], lexical_only=True)
        second = store.search("Rust", channel_ids=[10], lexical_only=True)
    finally:
        LEXICAL_MEMO.reset(token)
    assert [hit.id for hit in first] == [hit.id for hit in second] == ["1"]
    assert calls == 1


def test_log_hit_preserves_legacy_positional_constructor():
    hit = LogHit("1", "10", "general", "alice", "2026-01-01", "Rust", "https://example.com")
    assert hit.date == "2026-01-01" and hit.text == "Rust" and hit.url == "https://example.com"
    assert hit.author_id == ""
    identified = LogHit(
        "1", "10", "general", "alice", "2026-01-01", "Rust", "https://example.com",
        author_id="20",
    )
    assert identified.author_id == "20"


def test_rrf_order():
    assert rrf([["1", "2"], ["2", "3"]]) == ["2", "1", "3"]
    assert rrf([["1", "2"], ["2", "3"]], limit=1) == ["2"]
    assert rrf([]) == []


def test_dense_head_fusion_lexical_gate_and_recall_tail():
    lexical = [{"id": "lex", "author_name": "alice", "text": "exact Rust identifier"},
               {"id": "tail", "author_name": "alice", "text": "historical context"}]
    assert _fusion("昔の話", lexical, ["dense", "tail"], 3) == ["dense", "tail", "lex"]
    assert _fusion("Rust identifier", lexical, ["dense", "tail"], 3) == ["lex", "dense", "tail"]
    assert _fusion("昔の話", [], ["dense"], 3) == ["dense"]
    assert _fusion("昔の話", lexical, [], 3) == ["lex", "tail"]
    assert _fusion("", [], [], 3) == []


def test_speaker_alias_queries_and_returned_identity(store):
    value = record(1, text="Rust 検索エンジン")
    value["author"].update(
        username="alice_dev", global_name="Ａｌｉｃｅ", nick="テストユーザーさん"
    )
    store.upsert(value)
    store.upsert(record(2, author="bob", text="Rust 検索エンジン"))
    for name in ("20", "<@20>", "<@!20>", "alice_dev", "ALICE", "@alice_dev", "テストユーザーさん"):
        hit, = store.search("Rust", channel_ids=[10], author=name)
        assert hit.id == "1" and hit.author_id == "20" and hit.author == "テストユーザーさん"
        assert [h.id for h in store.search(name, channel_ids=[10])] == ["1"]
    for query in ("テストユーザーさんが言ってたRust", "Aliceが言っていた「検索エンジン」",
                  "Aliceさんが話してた検索エンジン", "alice_dev Rust", "<@!20> Rust",
                  "from:alice_dev Rust"):
        assert [h.id for h in store.search(query, channel_ids=[10])] == ["1"]
    output = format_hits(store.search("Rust", channel_ids=[10], author="20"))
    assert "テストユーザーさん (id: 20)" in output
    assert store.search("Rust", channel_ids=[10], author="missing") == []
    assert store.search("Rust", channel_ids=[10], author="' OR 1=1 --") == []
    assert [hit.id for hit in store.search("", channel_ids=[10], author="20")] == ["1"]


@pytest.mark.parametrize("query", [
    "Alice Smith", "Alice Smith Rust", "from:Alice Smith", "from:Alice Smith Rust",
    "  Alice  Smith Rust  ", "ＡＬＩＣＥ　ＳＭＩＴＨ Rust", "Alice Smithさん Rust",
    "Alice Smithが言ってたRust",
])
def test_longest_speaker_alias_prefix_wins(store, query):
    shorter = record(1, text="Smith Rust")
    shorter["author"]["global_name"] = "Alice"
    longer = record(2, author="bob", text="Rust")
    longer["author"]["global_name"] = "Alice Smith"
    store.upsert_many([shorter, longer])
    assert [h.id for h in store.search(query, channel_ids=[10])] == ["2"]
    assert [h.id for h in store.search("Alice Rust", channel_ids=[10])] == ["1"]
    assert store.search(query, channel_ids=[11]) == []
    with store.connect() as db:
        assert store._speaker_query(db, "nobody Rust", None, ["10"]) == ("nobody Rust", None)


@pytest.mark.parametrize(("query", "text"), [
    ("Alice", ""), ("Aliceさん", ""), ("Alice氏", ""),
    ("Alice Rust", "Rust"), ("Aliceさん Rust", "Rust"), ("Alice氏 Rust", "Rust"),
    ("Aliceさんが言ってたRust", "Rust"), ("from:Aliceさんが言ってたRust", "Rust"),
    ("from:Alice氏", ""), ("from:Aliceさん Rust", "Rust"),
    ("<@20>", ""), ("<@!20>", ""), ("<@20>さん", ""), ("<@!20>氏", ""),
    ("<@20>Rust", "Rust"), ("<@!20>Rust", "Rust"), ("<@20> Rust", "Rust"),
    ("<@20>さん Rust", "Rust"), ("<@!20>氏 Rust", "Rust"),
    ("from:<@20>Rust", "Rust"), ("from:<@!20>Rust", "Rust"), ("from: <@20>Rust", "Rust"),
    ("from:<@20>", ""), ("from:<@!20>氏", ""), ("from:<@20>さん Rust", "Rust"),
    ("<@20>さんが言ってたRust", "Rust"), ("from:<@20>さんが言ってたRust", "Rust"),
    ("<@20>氏名", "氏名"), ("<@20>さんま", "さんま"),
])
def test_speaker_forms_preserve_query_and_strip_honorifics(store, query, text):
    store.upsert_many([
        record(1, text="Rust 氏名 さんま"), record(2, author="bob", text="<@999> Rust"),
    ])
    assert [h.id for h in store.search(query, channel_ids=[10])] == ["1"]
    assert store.search(query, channel_ids=[]) == []
    assert store.search("<@999>Rust", channel_ids=[10]) == []
    with store.connect() as db:
        assert store._speaker_query(db, query, None, ["10"]) == (text, ("20",))
        assert store._speaker_query(db, "<@20Rust", None, ["10"]) == ("<@20Rust", None)


def test_speaker_alias_history_ambiguity_and_permissions(store):
    value = record(1, text="Rust")
    value["author"].update(global_name="old name", nick="nickname")
    store.upsert(value)
    changed = record(2, text="Rust")
    changed["author"].update(global_name="new name", username="renamed")
    store.upsert(changed)
    other = record(3, text="Rust", author="bob")
    other["author"]["global_name"] = "nickname"
    store.upsert(other)
    hidden = record(4, text="Rust", author="bob", channel="11")
    hidden["author"]["nick"] = "private alias"
    store.upsert(hidden)
    assert {h.id for h in store.search("Rust", author="old name", channel_ids=[10])} == {"1", "2"}
    assert {h.id for h in store.search("Rust", author="nickname", channel_ids=[10])} == {
        "1", "2", "3"
    }
    assert store.search("Rust", author="private alias", channel_ids=[10]) == []
    assert [h.id for h in store.search("Rust", author="private alias", channel_ids=[11])] == ["4"]
    assert store.search("nickname", channel_ids=[]) == []
    store.mark_deleted("2")
    assert [h.id for h in store.search("renamed", channel_ids=[10])] == ["1"]


def test_speaker_log_tool_filter_and_identity_format(store):
    value = record(1, text="Rust")
    value["author"].update(global_name="Alice", nick="テストユーザー")
    store.upsert(value)
    tool = LogTools(store, [10])
    for author in ("<@20>", "alice", "テストユーザー"):
        output = tool.call("search_server_log", {"query": "Rust", "author": author})
        assert "テストユーザー (id: 20)" in output
        assert "2026-01-01T00:01:00+00:00" in output
    assert "No messages" in LogTools(store, [11]).call(
        "search_server_log", {"query": "Rust", "author": "<@20>"}
    )
    with pytest.raises(ValueError, match="Invalid server log tool arguments"):
        tool.call("search_server_log", {"query": "Rust", "author": 20})


def test_dense_pending_fallback_with_resolved_aliases(store):
    store = dense_store(store)
    value = record(1, text="semantic")
    value["author"].update(global_name="Alice", nick="テストユーザー")
    store.upsert(value)
    store.embed_pending()
    store.upsert(record(2, author="bob", text="unembedded"))
    assert [h.id for h in store.search("テストユーザーが言ってた質問", channel_ids=[10])] == ["1"]
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]
    store.upsert(record(3, text="unembedded"))
    assert [h.id for h in store.search("質問", author="テストユーザー", channel_ids=[10])] == ["1"]


def test_speaker_schema_migrates_old_database_and_raw_backfill(store):
    store.upsert(record(1, text="Rust"))
    with store.connect() as db:
        row = db.execute("SELECT * FROM messages").fetchone()
        vector = b"\x00\x3c\x00\x00"
        db.execute("INSERT INTO message_vectors VALUES (?, ?, ?, ?, ?)",
                   ("1", store.model, text_hash(row), 2, vector))
        db.execute("DELETE FROM log_migrations WHERE name = 'speaker_identity'")
        db.execute("DELETE FROM author_aliases")
        for column in ("author_username", "author_global_name", "author_nickname"):
            db.execute(f"ALTER TABLE messages DROP COLUMN {column}")
    reopened = ServerLog(store.path)
    assert reopened.search("alice", channel_ids=[10])[0].author_id == "20"
    value = record(1, text="Rust")
    value["author"].update(username="actual_username", global_name="alice")
    assert reopened.backfill_authors([value, record(100)]) == 1
    assert reopened.backfill_authors([value]) == 1
    assert reopened.search("actual_username", channel_ids=[10])[0].author_id == "20"
    with reopened.connect() as db:
        assert db.execute("SELECT author_username FROM messages").fetchone()[0] == "actual_username"
        assert db.execute("SELECT vector FROM message_vectors").fetchone()[0] == vector


def test_author_backfill_preserves_edits_tombstones_and_vectors(store, tmp_path, capsys):
    value = record(1, text="new content")
    value["edited_timestamp"] = "2026-02-01T00:00:00Z"
    store.upsert(value)
    stale = record(1, text="stale content")
    stale["author"].update(global_name="Old Alice")
    stale["member"] = {"nick": "historical nick"}
    store.mark_deleted("1")
    source = tmp_path / "authors.jsonl"
    source.write_text(json.dumps(stale) + "\n")
    main(["--db", str(store.path), "backfill-authors", str(source)])
    assert "Processed 1 existing author records" in capsys.readouterr().out
    with store.connect() as db:
        row = db.execute("SELECT * FROM messages").fetchone()
        assert row["text"] == "new content" and row["deleted"] == 1
        assert row["author_global_name"] == "Old Alice"
        assert row["author_nickname"] == "historical nick"
    assert store.search("historical nick", channel_ids=[10]) == []
    assert store.backfill_authors([{**stale, "author": {"id": "other", "username": "other"}}]) == 0


def test_export_guild_default_and_transactions(store, monkeypatch):
    monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)
    value = record()
    value.pop("guild_id")
    default = ServerLog(store.path)
    default.upsert(value)
    assert default.recent(10, 1)[0].url.endswith(f"/{Settings.guild_id}/10/1")
    monkeypatch.setenv("DISCORD_GUILD_ID", "200")
    default = ServerLog(store.path)
    default.upsert(value)
    assert default.recent(10, 1)[0].url.endswith("/200/10/1")
    assert default.upsert_many([]) == 0
    with pytest.raises(KeyError):
        default.upsert_many([record(2), {"id": "3"}])
    assert [hit.id for hit in default.recent(10, 5)] == ["1"]


class FakeEmbedder:
    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        for text in texts:
            self.inputs.append(text)
            yield [0.0, 1.0] if "semantic" in text or "質問" in text else [1.0, 0.0]


def dense_store(store, embedder=None):
    pytest.importorskip("numpy")
    return ServerLog(
        store.path, model="intfloat/multilingual-e5-large", embedder=embedder or FakeEmbedder()
    )


def test_dense_vectors_fusion_filters_and_incomplete_fallback(store):
    backend = FakeEmbedder()
    store = dense_store(store, backend)
    store.upsert_many(
        [
            record(1, text="質問 Rust"),
            record(2, text="semantic concepts", author="bob"),
            record(3, channel="11", text="semantic secret"),
            record(4, text="bot", bot=True),
            record(5, text="a"),
        ]
    )
    progress = []
    assert store.embed_pending(batch_size=1, progress=progress.append) == 3
    assert progress == [1, 2, 3]
    assert all(text.startswith("passage: ") for text in backend.inputs)
    with store.connect() as db:
        vectors = db.execute("SELECT * FROM message_vectors").fetchall()
        assert len(vectors) == 3
        assert all(len(row["vector"]) == row["dimensions"] * 2 for row in vectors)
    assert {h.id for h in store.search("質問", channel_ids=[10])} == {"1", "2"}
    assert backend.inputs[-1] == "query: 質問"
    assert [h.id for h in store.search("質問", channel_ids=[10], author="bob")] == ["2"]
    assert store.search("質問", channel_ids=[10], channel="private") == []
    assert [
        h.id
        for h in store.search(
            "質問",
            channel_ids=[10],
            after="2026-01-01T00:01:00Z",
        )
    ] == ["2"]
    store.upsert(record(2, text="changed", author="bob"))
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]
    assert store.embed_pending() == 1
    store.mark_deleted("2")
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]
    assert ServerLog(store.path, model="different", embedder=FakeEmbedder()).embed_pending() == 2


def test_dense_static_cascade_fills_gaps_and_teacher_wins(store):
    np = pytest.importorskip("numpy")
    backend = FakeEmbedder()
    store = dense_store(store, backend)
    store.static_model = "static-test"
    store.upsert_many(
        [
            record(1, text="質問 Rust"),
            record(2, text="semantic concepts", author="bob"),
            record(3, text="semantic uncovered", author="carol"),
        ]
    )

    def put_vector(db, message_id, model, vector):
        row = db.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        db.execute(
            "INSERT OR REPLACE INTO message_vectors VALUES (?, ?, ?, ?, ?)",
            (message_id, model, text_hash(row), len(vector),
             np.array(vector, dtype="<f2").tobytes()),
        )

    with store.connect() as db:
        # Message 1 has both: the teacher vector must win over the static one.
        put_vector(db, "1", "intfloat/multilingual-e5-large", [0.0, 1.0])
        put_vector(db, "1", "static-test", [1.0, 0.0])
        # Message 2 is covered by the static model only.
        put_vector(db, "2", "static-test", [0.0, 1.0])
        db.commit()

    dense = store.enable_dense()
    assert {h.id for h in store.search("質問", channel_ids=[10])} == {"1", "2"}
    assert set(dense.ids) == {"1", "2"}
    # Teacher direction [0, 1] for message 1, not the static [1, 0].
    assert dense.matrix[dense.ids["1"]].tolist() == pytest.approx([0.0, 1.0])
    assert dense.matrix[dense.ids["2"]].tolist() == pytest.approx([0.0, 1.0])
    # Message 3 has no vector anywhere: it stays pending for the teacher.
    pending = dense.pending.execute("SELECT id FROM messages").fetchall()
    assert [row[0] for row in pending] == ["3"]


@pytest.mark.parametrize("text,expected", [
    ("Rust 検索エンジン", True),
    ("ok", True),
    ("１２", True),
    ("ｱｲ", True),
    ("あ", False),
    ("a", False),
    ("a😀", False),
    ("😀😀😀", False),
    ("！！！", False),
    ("  ", False),
    ("", False),
])
def test_embeddable_text_noise_gate(text, expected):
    assert embeddable_text(text) is expected


def test_dense_embedding_skips_noise_and_keeps_log_rows(store):
    backend = FakeEmbedder()
    store = dense_store(store, backend)
    store.upsert_many([
        record(1, text="semantic concepts"),
        record(2, text="😀😀😀"),
        record(3, text="！！！"),
        record(4, text="あ"),
        record(5, text="  "),
        record(6, text="ok"),
        record(7, text="１２"),
        record(8, text="a😀"),
        record(9, text="ｱｲ"),
    ])
    assert store.embed_pending() == 4
    with store.connect() as db:
        embedded = {row[0] for row in db.execute("SELECT message_id FROM message_vectors")}
        stored = db.execute("SELECT count(*) FROM messages").fetchone()[0]
    assert embedded == {"1", "6", "7", "9"}
    assert stored == 9
    assert store.embed_pending() == 0
    hit_ids = {h.id for h in store.search("semantic", channel_ids=[10])}
    assert "1" in hit_ids
    assert hit_ids.isdisjoint({"2", "3", "4", "5", "8"})
    pending = {row[0] for row in store._dense.pending.execute("SELECT id FROM messages")}
    assert pending == set()


def test_edit_during_embedding_does_not_publish_stale_vector(store):
    class EditingEmbedder(FakeEmbedder):
        def embed(self, texts):
            if not self.inputs:
                store.upsert(record(text="edited text"))
            yield from super().embed(texts)

    store.upsert(record())
    dense = dense_store(store, EditingEmbedder())
    assert dense.embed_pending() == 1
    assert [h.id for h in dense.search("edited", channel_ids=[10])] == ["1"]
    with dense.connect() as db:
        assert db.execute("SELECT text_hash FROM message_vectors").fetchone()[0] == text_hash(
            db.execute("SELECT * FROM messages").fetchone(),
        )


def test_incremental_cache_insert_edit_delete_and_cross_instance(store, monkeypatch):
    store = dense_store(store)
    store.upsert_many([record(1, text="semantic"), record(2, text="Rust")])
    store.embed_pending()
    store.search("質問", channel_ids=[10])
    dense = store._dense
    matrix = dense.matrix
    slot = dense.ids["1"]
    writer = ServerLog(store.path)
    writer.upsert(record(3, text="new semantic"))
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1", "2"]
    assert dense.matrix is matrix and dense.rebuilds == 1
    assert store.embed_pending() == 1
    assert {h.id for h in store.search("質問", channel_ids=[10])} == {"1", "2", "3"}
    writer.upsert(record(1, text="edited Rust"))
    assert {h.id for h in store.search("質問", channel_ids=[10])} == {"2", "3"}
    assert not dense.active[slot]
    assert store.embed_pending() == 1
    store.search("質問", channel_ids=[10])
    assert dense.ids["1"] == slot
    assert list(dense.matrix[slot]) == [1.0, 0.0]
    writer.mark_deleted("1")
    assert "1" not in {h.id for h in store.search("質問", channel_ids=[10])}
    assert not dense.active[slot]
    writer.upsert(record(1, text="resurrect semantic"))
    assert "1" not in {h.id for h in store.search("質問", channel_ids=[10])}
    assert dense.matrix is matrix and dense.rebuilds == 1

    def forbidden_hash(row):
        raise AssertionError("warm query hashed a row")

    monkeypatch.setattr("nezubot_discrod.server_log_dense.text_hash", forbidden_hash)
    statements = []
    connect = store.connect

    @contextmanager
    def traced_connect():
        with connect() as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(store, "connect", traced_connect)
    assert store.search("質問", channel_ids=[10])
    message_reads = [sql for sql in statements if sql.lstrip().startswith("SELECT")
                     and "FROM messages m" in sql and "JOIN" not in sql]
    assert message_reads and all("m.id IN" in sql for sql in message_reads)


def test_pending_completeness_obeys_scope_and_filters(store):
    store = dense_store(store)
    store.upsert_many([record(1, text="semantic", author="bob"),
                       record(2, channel="11", text="semantic")])
    store.embed_pending()
    store.search("質問", channel_ids=[10])
    store.upsert(record(3, text="unembedded", author="alice"))
    assert [h.id for h in store.search("質問", channel_ids=[10], author="bob")] == ["1"]
    assert [h.id for h in store.search(
        "質問", channel_ids=[10], before="2026-01-01T00:02Z"
    )] == ["1"]
    assert [h.id for h in store.search("質問", channel_ids=[10, 11], channel="private")] == ["2"]
    assert store.search("質問", channel_ids=[10], after="2026-01-01T00:02Z") == []
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]


def test_dense_filter_overfetch_widens_and_is_permission_scoped(store):
    store = dense_store(store)
    records = []
    for number in range(450):
        value = record(1, text="semantic", author="bob" if number >= 440 else "alice")
        value["id"] = str(1000 + number)
        value["timestamp"] = "2026-01-01T00:02:00Z" if number >= 440 else value["timestamp"]
        records.append(value)
    store.upsert_many(records)
    store.embed_pending()
    assert len(store.search("質問", channel_ids=[10], author="bob")) == 10
    assert len(store.search("質問", channel_ids=[10], after="2026-01-01T00:01:30Z")) == 10
    assert store.search("質問", channel_ids=[]) == []
    assert store.search("質問", channel_ids=[11]) == []
    assert all(hit.author == "bob" for hit in store.search("質問", channel_ids=[10], author="bob"))


def test_query_lru_is_bounded_and_never_caches_scope(store):
    backend = FakeEmbedder()
    store = dense_store(store, backend)
    store.upsert_many([record(1, text="semantic"), record(2, channel="11", text="semantic")])
    store.embed_pending()
    dense = store._dense
    dense.query_cache_size = 2
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]
    assert [h.id for h in store.search("質問", channel_ids=(c for c in [11]))] == ["2"]
    assert backend.inputs.count("query: 質問") == 1
    store.search("other", channel_ids=[10])
    store.search("third", channel_ids=[10])
    assert len(dense.queries) == 2
    store.search("質問", channel_ids=[10])
    assert backend.inputs.count("query: 質問") == 2


def test_empty_dense_index_and_failed_query_cache(store):
    class FailingQuery(FakeEmbedder):
        fail = True

        def embed(self, texts):
            if self.fail and texts[0].startswith("query:"):
                self.fail = False
                raise RuntimeError("query unavailable")
            return super().embed(texts)

    store = dense_store(store, FailingQuery())
    assert store.embed_pending() == 0
    assert store.search("質問", channel_ids=[10]) == []
    assert not store._dense.queries
    store.upsert(record(1, text="semantic"))
    store.embed_pending()
    assert store.search("質問", channel_ids=[10]) == []
    assert not store._dense.queries
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]


def test_cached_tombstone_and_physical_deletion_mask_old_vectors(store):
    store = dense_store(store)
    store.upsert_many([record(1, text="semantic"), record(2, text="semantic")])
    store.embed_pending()
    assert len(store.search("質問", channel_ids=[10])) == 2
    dense = store._dense
    with store.connect() as db:
        db.execute("INSERT INTO deleted_messages VALUES ('1')")
        db.execute("DELETE FROM messages WHERE id = '2'")
    assert store.search("質問", channel_ids=[10]) == []
    assert not dense.active[dense.ids["1"]]
    assert not dense.active[dense.ids["2"]]
    assert dense.rebuilds == 1


def test_query_matmul_limits_blas_to_one_thread(store, monkeypatch):
    pytest.importorskip("threadpoolctl")
    from threadpoolctl import threadpool_info, threadpool_limits

    store = dense_store(store)
    store.upsert(record(1, text="semantic"))
    store.embed_pending()
    calls = []

    @contextmanager
    def checked_limits(**kwargs):
        calls.append(kwargs)
        with threadpool_limits(**kwargs):
            pools = [p for p in threadpool_info() if p["user_api"] == "blas"]
            assert pools and all(p["num_threads"] == 1 for p in pools)
            yield

    monkeypatch.setattr(store._dense, "threadpool_limits", checked_limits)
    assert store.search("質問", channel_ids=[10])
    assert calls == [{"limits": 1, "user_api": "blas"}]


def test_blas_fallback_without_threadpoolctl(store, monkeypatch):
    pytest.importorskip("threadpoolctl")
    from threadpoolctl import threadpool_info

    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    store = dense_store(store)
    store.upsert(record(1, text="semantic"))
    store.embed_pending()
    previous = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
    with store._dense.threadpool_limits(limits=1, user_api="blas"):
        assert all(p["num_threads"] == 1 for p in threadpool_info() if p["user_api"] == "blas")
    assert [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"] == previous
    assert store.search("質問", channel_ids=[10])


def test_openblas_controller_restores_threads_and_missing_backend(monkeypatch):
    from nezubot_discrod import server_log_dense

    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    state = [8]

    def get_threads():
        return state[0]

    def set_threads(value):
        state[0] = value

    library = SimpleNamespace(scipy_openblas_get_num_threads64_=get_threads,
                              scipy_openblas_set_num_threads64_=set_threads)
    monkeypatch.setattr(server_log_dense.ctypes, "CDLL", lambda path: library)
    np = SimpleNamespace(
        _core=SimpleNamespace(_multiarray_umath=SimpleNamespace(__file__=__file__))
    )
    limiter = server_log_dense._blas_limiter(np)
    with pytest.raises(RuntimeError, match="matmul failed"):
        with limiter(limits=1, user_api="blas"):
            assert state == [1]
            raise RuntimeError("matmul failed")
    assert state == [8]
    with pytest.raises(ValueError, match="Only BLAS"):
        with limiter(limits=1, user_api="other"):
            pass
    monkeypatch.setattr(server_log_dense.ctypes, "CDLL", lambda path: SimpleNamespace())
    with pytest.raises(ImportError, match="Install threadpoolctl"):
        server_log_dense._blas_limiter(np)


def test_e5_max_length_configures_tokenizer_and_keeps_existing_vectors(store, monkeypatch):
    store.upsert(record())
    dense_store(store).embed_pending()
    backend = FakeEmbedder()
    lengths = []
    backend.model = SimpleNamespace(tokenizer=SimpleNamespace(enable_truncation=
                                                             lambda **kw: lengths.append(kw)))
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=lambda *a, **k:
                                                               backend))
    monkeypatch.setenv("SERVER_LOG_MAX_LENGTH", "64")
    configured = ServerLog(store.path)
    assert configured.embed_pending() == 0
    assert lengths == [{"max_length": 64}]
    assert configured.search("Rust", channel_ids=[10])
    assert backend.inputs == ["query: Rust"]
    assert ServerLog(store.path, max_length=256).max_length == 256
    with pytest.raises(ValueError, match="must be positive"):
        ServerLog(store.path, max_length=0)
    monkeypatch.setenv("SERVER_LOG_MAX_LENGTH", "invalid")
    with pytest.raises(ValueError):
        ServerLog(store.path)


def test_stale_stored_hash_and_corrupt_dimensions_fall_back(store):
    store = dense_store(store)
    store.upsert_many([record(1, text="semantic"), record(2, text="Rust")])
    store.embed_pending()
    store.search("質問", channel_ids=[10])
    with store.connect() as db:
        db.execute("UPDATE message_vectors SET text_hash = 'stale' WHERE message_id = '1'")
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["2"]
    with store.connect() as db:
        db.execute("DELETE FROM message_vectors WHERE message_id = '1'")
    store.embed_pending()
    with store.connect() as db:
        db.execute("UPDATE message_vectors SET dimensions = 3 WHERE message_id = '1'")
    assert [h.id for h in store.search("Rust", channel_ids=[10])] == ["2"]


def test_search_rrf_order_and_cached_scope(store):
    store = dense_store(store)
    store.upsert_many(
        [
            record(1, text="Rust"),
            record(2, text="Rust semantic"),
            record(3, text="semantic"),
            record(4, channel="11", text="semantic"),
        ]
    )
    assert store.embed_pending() == 4
    assert [hit.id for hit in store.search("Rust 質問", channel_ids=[10])] == ["1", "2", "3"]
    assert set(store._dense.ids) == {"1", "2", "3", "4"}
    assert [hit.id for hit in store.search("質問", channel_ids=[11])] == ["4"]
    assert set(store._dense.ids) == {"1", "2", "3", "4"}
    store.upsert(record(5, channel="12", text="pending"))
    assert [hit.id for hit in store.search("Rust 質問", channel_ids=[10])] == ["1", "2", "3"]
    assert [hit.id for hit in store.search("Rust 質問", channel_ids=[10, 12])] == ["1", "2", "3"]


@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 1.0]])
def test_invalid_vectors_leave_lexical_available(store, vector):
    class InvalidEmbedder:
        def embed(self, texts):
            return [vector for _ in texts]

    store = dense_store(store, InvalidEmbedder())
    store.upsert(record())
    with pytest.raises(ValueError, match="embedding output"):
        store.embed_pending()
    assert [hit.id for hit in store.search("Rust", channel_ids=[10])] == ["1"]


def test_optional_dependency_and_backend_failure_fall_back(store, monkeypatch):
    store.upsert(record())
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert store.embed_pending() == 0
    assert store.search("Rust", channel_ids=[10])[0].id == "1"

    class BrokenEmbedder:
        def embed(self, texts):
            raise RuntimeError("unavailable")

    dense = dense_store(store)
    assert dense.embed_pending() == 1
    dense._dense.embedder = BrokenEmbedder()
    assert dense.search("Rust", channel_ids=[10])[0].id == "1"
    with pytest.raises(RuntimeError, match="dense search failed"):
        dense.search("Rust", channel_ids=[10], require_dense=True)


def test_background_embedder(store):
    from threading import Event

    finished = Event()

    class SignallingEmbedder(FakeEmbedder):
        def embed(self, texts):
            yield from super().embed(texts)
            finished.set()

    store = dense_store(store, SignallingEmbedder())
    store.upsert(record())
    worker = store.start_embedder(poll_interval=0.01)
    try:
        assert store.start_embedder() is worker
        assert finished.wait(5)
    finally:
        store.stop_embedder(timeout=5)
    assert not worker.is_alive()
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM message_vectors").fetchone()[0] == 1


def test_explicit_dense_initialization_does_not_embed_or_start_worker(store):
    store = dense_store(store)
    store.upsert(record(1, text="unembedded exact"))
    assert store._dense is None
    dense = store.enable_dense()
    assert store.enable_dense() is dense
    assert store._worker is None and store._embedder.inputs == []
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM message_vectors").fetchone()[0] == 0
    with pytest.raises(RuntimeError, match="no usable scoped vectors"):
        store.search("exact", channel_ids=[10], require_dense=True)
    assert store.search("exact", channel_ids=[10])[0].id == "1"


def test_search_logs_loading_fallback_and_uses_dense_when_ready(store, monkeypatch, caplog):
    store = dense_store(store)
    store.upsert_many([record(1, text="semantic target"), record(2, text="質問 lexical")])
    store.embed_pending()
    reader = dense_store(ServerLog(store.path))
    assert reader.search("質問 解析", channel_ids=[10])[0].id == "2"
    loading, release = Event(), Event()

    def delayed(*args):
        loading.set()
        assert release.wait(5)
        return DenseIndex(*args)

    monkeypatch.setattr("nezubot_discrod.server_log_dense.DenseIndex", delayed)
    caplog.set_level("INFO", logger="nezubot_discrod.server_log")
    worker = Thread(target=reader.enable_dense)
    worker.start()
    try:
        assert loading.wait(5)
        assert reader.search("質問 解析", channel_ids=[10])[0].id == "2"
        assert "mode=lexical reason=loading" in caplog.text
        with pytest.raises(RuntimeError, match="loading"):
            reader.search("質問 解析", channel_ids=[10], require_dense=True)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert reader.search("質問 解析", channel_ids=[10], require_dense=True)[0].id == "1"
    assert "mode=dense" in caplog.text


def test_dense_initialization_failure_can_retry(store, monkeypatch):
    store = dense_store(store)

    def broken(*args):
        raise ImportError("missing backend")

    monkeypatch.setattr("nezubot_discrod.server_log_dense.DenseIndex", broken)
    with pytest.raises(ImportError, match="missing backend"):
        store.enable_dense()
    assert store._dense is None and store._dense_state == "initialization failed"
    monkeypatch.setattr("nezubot_discrod.server_log_dense.DenseIndex", DenseIndex)
    assert store.enable_dense() is store._dense
    assert store._dense_state == "ready"


@pytest.mark.parametrize("lexical_only,first", [(False, "1"), (True, "2")])
def test_search_cli_explicit_backend_and_permission_scope(store, monkeypatch, capsys,
                                                        lexical_only, first):
    store = dense_store(store)
    store.upsert_many([record(1, text="semantic target"), record(2, text="質問 lexical"),
                       record(3, channel="11", text="semantic secret")])
    assert store.embed_pending() == 3
    reader = dense_store(ServerLog(store.path))
    monkeypatch.setattr("nezubot_discrod.server_log.ServerLog", lambda *a, **kw: reader)
    args = ["search", "質問 解析", "--channel-ids", "10", "--limit", "1"]
    main(args + (["--lexical-only"] if lexical_only else []))
    output = capsys.readouterr().out.strip()
    assert output.endswith("/" + first) and "secret" not in output
    assert reader._worker is None
    assert (reader._dense is None) == lexical_only
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM message_vectors").fetchone()[0] == 3


def test_search_cli_all_channels_filters_and_empty_archive(store, monkeypatch, capsys):
    reader = dense_store(store)
    monkeypatch.setattr("nezubot_discrod.server_log.ServerLog", lambda *a, **kw: reader)
    main(["search", "Rust", "--all-channels"])
    output = capsys.readouterr()
    assert output.out.strip() == "No messages found."
    assert "NOT live Discord caller permissions" in output.err
    reader.upsert_many([record(1, text="Rust public"),
                        record(2, channel="11", text="Rust private")])
    reader.embed_pending()
    main(["search", "Rust", "--all-channels", "--author", "alice", "--channel", "general",
          "--after", "2025-01-01", "--before", "2027-01-01", "--limit", "1"])
    output = capsys.readouterr().out.strip()
    assert output.endswith("/1") and "private" not in output


def test_search_cli_escapes_terminal_controls_without_changing_bot_output(
    store, monkeypatch, capsys,
):
    text = "Rust \x1b[31mred\x1b[0m\x07\x7f\x9b31m\x9d52;c;payload"
    store.upsert(record(1, text=text))
    hits = store.search("Rust", channel_ids=[10])
    assert "\x1b[31m" in format_hits(hits)
    monkeypatch.setattr("nezubot_discrod.server_log.ServerLog", lambda *a, **kw: store)
    args = ["search", "Rust", "--channel-ids", "10", "--lexical-only"]
    main(args)
    output = capsys.readouterr().out
    assert all(char in "\n\t" or not (ord(char) < 32 or 127 <= ord(char) <= 159)
               for char in output)
    assert all(escape in output for escape in ("\\x1b", "\\x07", "\\x7f", "\\x9b", "\\x9d"))
    monkeypatch.setattr("nezubot_discrod.server_log.format_hits", lambda hits: "line\n\ttab\x00end")
    main(args)
    assert capsys.readouterr().out == "line\n\ttab\\x00end\n"


@pytest.mark.parametrize("embedded,missing,bound", [(1, 2, "after"), (2, 1, "before")])
def test_strict_search_cli_rejects_unembedded_date_scope(
    store, monkeypatch, capsys, embedded, missing, bound,
):
    store = dense_store(store)
    store.upsert(record(embedded, text="semantic"))
    assert store.embed_pending() == 1
    store.upsert(record(missing, text="Rust unembedded"))
    cutoff = "2026-01-01T00:01:30Z"
    filters = {bound: cutoff}
    assert [hit.id for hit in store.search("Rust", channel_ids=[10], **filters)] == [str(missing)]
    with pytest.raises(RuntimeError, match="no usable scoped vectors"):
        store.search("Rust", channel_ids=[10], require_dense=True, **filters)
    monkeypatch.setattr("nezubot_discrod.server_log.ServerLog", lambda *a, **kw: store)
    with pytest.raises(SystemExit) as error:
        main(["search", "Rust", "--channel-ids", "10", f"--{bound}", cutoff])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert not output.out and "no usable scoped vectors" in output.err


def test_strict_search_rejects_zero_coverage_even_with_candidates(store, monkeypatch):
    store = dense_store(store)
    store.upsert(record(1, text="Rust"))
    dense = store.enable_dense()
    monkeypatch.setattr(dense, "search", lambda *a, **kw: DenseHits(
        ["1"], partial=True, coverage=0,
    ))
    with pytest.raises(RuntimeError, match="no usable scoped vectors"):
        store.search("Rust", channel_ids=[10], require_dense=True)
    assert store.search("Rust", channel_ids=[10])[0].id == "1"


@pytest.mark.parametrize("author", ["unknown author", "999999999", "<@999999999>", "bob"])
def test_strict_search_cli_returns_no_results_for_empty_author_scope(
    store, monkeypatch, capsys, author,
):
    store = dense_store(store)
    store.upsert_many([record(1, text="Rust"), record(2, channel="11", author="bob", text="Rust")])
    store.embed_pending()
    monkeypatch.setattr("nezubot_discrod.server_log.ServerLog", lambda *a, **kw: store)
    monkeypatch.setattr(store._dense, "search", lambda *a, **kw: pytest.fail(
        "empty scope must not invoke dense search",
    ))
    main(["search", "Rust", "--channel-ids", "10", "--author", author])
    output = capsys.readouterr()
    assert output.out.strip() == "No messages found."
    assert "Search failed" not in output.err


def test_search_cli_dense_failure_is_not_silent_lexical(store, monkeypatch, capsys):
    store.upsert(record())
    monkeypatch.setattr("nezubot_discrod.server_log.ServerLog", lambda *a, **kw: store)
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(SystemExit) as error:
        main(["search", "Rust", "--channel-ids", "10"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert not output.out and "Search failed:" in output.err
    assert "--lexical-only" in output.err


@pytest.mark.parametrize("command", ["search", "embed", "import", "backfill-authors",
                                   "export-vectors", "import-vectors"])
@pytest.mark.parametrize("inherited_model", [None, "service-model"])
def test_cli_loads_checkout_dotenv_without_overriding_service_env(
    tmp_path, monkeypatch, command, inherited_model,
):
    module = sys.modules["nezubot_discrod.server_log"]
    monkeypatch.setattr(module, "__file__", str(tmp_path / "nezubot_discrod" / "server_log.py"))
    (tmp_path / ".env").write_text(
        "SERVER_LOG_EMBEDDING_MODEL=checkout-model\nSERVER_LOG_MAX_LENGTH=256\n"
    )
    monkeypatch.setenv("SERVER_LOG_EMBEDDING_MODEL", inherited_model or "placeholder")
    if inherited_model is None:
        monkeypatch.delenv("SERVER_LOG_EMBEDDING_MODEL")
    monkeypatch.setenv("SERVER_LOG_MAX_LENGTH", "512")
    monkeypatch.delenv("SERVER_LOG_MAX_LENGTH")
    stores = []

    def construct(*args, **kwargs):
        reader = ServerLog(*args, **kwargs)
        stores.append(reader)
        monkeypatch.setattr(reader, "enable_dense", lambda: None)
        monkeypatch.setattr(reader, "search", lambda *a, **kw: [])
        monkeypatch.setattr(reader, "embed_pending", lambda **kw: 0)
        monkeypatch.setattr(reader, "_dense", object())
        monkeypatch.setattr(reader, "upsert_many", lambda records: len(list(records)))
        monkeypatch.setattr(reader, "backfill_authors", lambda records: len(list(records)))
        return reader

    monkeypatch.setattr(module, "ServerLog", construct)
    monkeypatch.setattr("nezubot_discrod.server_log_vectors.transfer", lambda *a: "{}")
    source = tmp_path / "messages.jsonl"
    source.write_text("{}\n")
    arguments = {
        "search": ["Rust", "--channel-ids", "10"], "embed": [],
        "import": [str(source)], "backfill-authors": [str(source)],
        "export-vectors": [str(source)], "import-vectors": [str(source)],
    }
    main([command, *arguments[command]])
    assert stores[0].model == (inherited_model or "checkout-model")
    assert stores[0].max_length == 256


@pytest.mark.parametrize("args", [["search", "Rust"],
                                  ["search", "Rust", "--all-channels", "--limit", "0"]])
def test_search_cli_requires_explicit_scope_and_positive_limit(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_private_sqlite_symlinks_rejected(tmp_path, suffix):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    (root / ("server_log.db" + suffix)).symlink_to(outside)
    with pytest.raises(PermissionError):
        ServerLog(root / "server_log.db").upsert(record())
    assert not outside.exists()


def test_private_directory_hardlinks_and_checkout_rejected(tmp_path):
    root = tmp_path / "data"
    root.mkdir(mode=0o755)
    with pytest.raises(PermissionError):
        ServerLog(root / "server_log.db").upsert(record())
    root.chmod(0o700)
    source = tmp_path / "source"
    source.touch(mode=0o600)
    os.link(source, root / "server_log.db")
    with pytest.raises(PermissionError, match="hard links"):
        ServerLog(root / "server_log.db").upsert(record())
    with pytest.raises(PermissionError):
        ServerLog(tmp_path / "checkout" / "db", checkout=tmp_path).upsert(record())
    (root / "server_log.db").unlink()
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(PermissionError):
        ServerLog(link / "db").upsert(record())


def test_message_conversion_and_import_cli(store, tmp_path, capsys):
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    message = SimpleNamespace(
        id=1,
        guild=SimpleNamespace(id=100),
        channel=SimpleNamespace(id=10, name="general", parent=SimpleNamespace(name="parent")),
        author=SimpleNamespace(id=20, name="alice", display_name="Alice", bot=False),
        created_at=stamp,
        edited_at=None,
        reference=SimpleNamespace(message_id=42),
        content="hello",
        embeds=[SimpleNamespace(to_dict=lambda: {"title": "title"})],
        attachments=[SimpleNamespace(filename="file.png")],
        stickers=[],
    )
    converted = message_record(message)
    assert converted["guild_id"] == "100" and converted["_parent"] == "parent"
    assert converted["message_reference"] == {"message_id": "42"}
    assert record_text(converted) == "hello\ntitle\n[file] file.png"
    message.author.global_name = "global name"
    message.author.nick = "server nickname"
    message.author.display_name = "server nickname"
    identity = message_record(message)["author"]
    assert identity["username"] == "alice"
    assert identity["global_name"] == "global name" and identity["nick"] == "server nickname"
    message.guild = None
    assert message_record(message)["guild_id"] == "@me"
    source = tmp_path / "messages.jsonl"
    converted.pop("guild_id")
    source.write_text(json.dumps(converted) + "\n")
    main(["--db", str(store.path), "--guild-id", "100", "import", str(source)])
    assert "Imported 1 messages" in capsys.readouterr().out
    assert store.search("title", channel_ids=[10])[0].url.endswith("/100/10/1")
    with pytest.raises(ValueError, match="batch_size"):
        store.upsert_many([], batch_size=0)


def test_tombstone_hides_row_even_if_a_racing_write_clears_deleted(store):
    store.upsert(record())
    store.mark_deleted("1")
    with store.connect() as db:
        db.execute("UPDATE messages SET deleted = 0 WHERE id = '1'")
    assert store.search("検索", channel_ids=[10]) == []
    assert store.recent("10", 5) == []


@pytest.mark.parametrize("coverage", ["random", "oldest"])
def test_fourteen_percent_coverage_preserves_unembedded_lexical_rank(store, coverage):
    backend = FakeEmbedder()
    store = dense_store(store, backend)
    for i in range(1, 101):
        value = record(i, text="質問 lexical target" if i == 100 else "semantic dense-only")
        value["timestamp"] = f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00"
        store.upsert(value)
    assert store.embed_pending() == 100
    kept = random.Random(42).sample(range(1, 100), 14) if coverage == "random" else range(1, 15)
    with store.connect() as db:
        db.execute(
            f"DELETE FROM message_vectors WHERE message_id NOT IN ({','.join('?' for _ in kept)})",
            [str(i) for i in kept],
        )
    before = list(backend.inputs)
    hits = store.search("質問", channel_ids=[10])
    assert hits[0].id == "100"
    assert len(hits) == 10
    assert {hit.id for hit in hits[1:]} <= {str(i) for i in kept}
    assert backend.inputs == before + ["query: 質問"]


def vector_bundle(tmp_path, records=None):
    records = records if records is not None else [record(1, text="semantic original")]
    source = dense_store(ServerLog(tmp_path / "source" / "server_log.db"))
    source.upsert_many(records)
    source.embed_pending()
    path = source.path.parent / "vectors.db"
    assert export_vectors(source, path) == len(records)
    return path


def test_vector_bundle_roundtrip_cache_journal_and_idempotence(store, tmp_path):
    value = record(1, text="semantic original")
    store = dense_store(store)
    assert store.embed_pending() == 0
    store.upsert(value)
    assert store.search("質問", channel_ids=[10]) == []
    cache = store._dense
    rebuilds = cache.rebuilds
    bundle = vector_bundle(tmp_path, [value])
    assert bundle.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(bundle) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {
            "metadata", "message_vectors"
        }
        expected = db.execute("SELECT * FROM message_vectors").fetchone()
    with store.connect() as db:
        generation = db.execute("SELECT generation FROM log_state").fetchone()[0]
        sequence = db.execute("SELECT max(version) FROM dense_changes").fetchone()[0]
    assert import_vectors(store, bundle)["imported"] == 1
    with store.connect() as db:
        assert tuple(db.execute("SELECT * FROM message_vectors").fetchone()) == expected
        assert db.execute("SELECT generation FROM log_state").fetchone()[0] > generation
        after = db.execute("SELECT max(version) FROM dense_changes").fetchone()[0]
        assert after > sequence
    assert [h.id for h in store.search("質問", channel_ids=[10])] == ["1"]
    assert cache.rebuilds == rebuilds
    assert import_vectors(store, bundle)["unchanged"] == 1
    with store.connect() as db:
        assert db.execute("SELECT max(version) FROM dense_changes").fetchone()[0] == after
        db.execute("UPDATE message_vectors SET vector = ?", (b"\x00\x3c\x00\x00",))
    store.search("質問", channel_ids=[10])
    assert list(cache.matrix[cache.ids["1"]]) == [1, 0]
    assert import_vectors(store, bundle)["imported"] == 1
    store.search("質問", channel_ids=[10])
    assert list(cache.matrix[cache.ids["1"]]) == [0, 1]
    assert cache.rebuilds == rebuilds
    store.upsert(record(1, text="unrelated edited"))
    assert import_vectors(store, bundle)["stale"] == 1
    assert store.search("質問", channel_ids=[10]) == []


def test_vector_import_skips_stale_missing_deleted_tombstoned_and_ineligible(store, tmp_path):
    records = [record(i, text="semantic original") for i in range(1, 9)]
    bundle = vector_bundle(tmp_path, records)
    store.upsert_many(records[:-1])
    store.upsert(record(2, text="edited content"))
    store.upsert(record(3, author="renamed", text="semantic original"))
    store.mark_deleted("4")
    store.mark_deleted("5")
    with store.connect() as db:
        db.execute("UPDATE messages SET deleted = 0 WHERE id = '5'")
        db.execute("UPDATE messages SET is_bot = 1 WHERE id = '6'")
        db.execute("UPDATE messages SET text = 'x' WHERE id = '7'")
    assert import_vectors(store, bundle) == {
        "imported": 1, "unchanged": 0, "missing": 1, "stale": 2, "deleted": 2,
        "ineligible": 2, "model": 0, "invalid": 0,
    }
    with store.connect() as db:
        assert [r[0] for r in db.execute("SELECT message_id FROM message_vectors")] == ["1"]


@pytest.mark.parametrize("field,value", [
    ("model", "other/model"), ("dimensions", 3), ("dimensions", 0),
    ("vector", b"bad"), ("vector", b"\x00\x7c\x00\x00"),
    ("vector", b"\x00\x7e\x00\x00"), ("vector", b"\x00\x00\x00\x00"),
])
def test_vector_import_skips_invalid_rows(store, tmp_path, field, value):
    store.upsert(record(1, text="semantic original"))
    bundle = vector_bundle(tmp_path)
    with sqlite3.connect(bundle) as db:
        db.execute(f"UPDATE message_vectors SET {field} = ?", (value,))
    counts = import_vectors(store, bundle)
    assert counts["imported"] == 0
    assert counts["model" if field == "model" else "invalid"] == 1


@pytest.mark.parametrize("key,value", [
    ("complete", "0"), ("format", "future"), ("document_format", "other"),
    ("dtype", "float32"), ("max_length", "512"), ("model", "other/model"),
    ("dimensions", "-1"),
])
def test_vector_import_rejects_incompatible_metadata(store, tmp_path, key, value):
    store.upsert(record(1, text="semantic original"))
    bundle = vector_bundle(tmp_path)
    with sqlite3.connect(bundle) as db:
        db.execute("UPDATE metadata SET value = ? WHERE key = ?", (value, key))
    with pytest.raises(ValueError):
        import_vectors(store, bundle)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM message_vectors").fetchone()[0] == 0


def test_vector_bundle_empty_cli_and_existing_output_protection(store, tmp_path, capsys):
    store.upsert(record())
    bundle = store.path.parent / "empty-vectors.db"
    main(["--db", str(store.path), "export-vectors", str(bundle)])
    assert capsys.readouterr().out.strip() == '{"exported": 0}'
    main(["--db", str(store.path), "import-vectors", str(bundle)])
    assert all(value == 0 for value in json.loads(capsys.readouterr().out).values())
    with pytest.raises(FileExistsError):
        export_vectors(store, bundle)
    with pytest.raises(FileExistsError):
        export_vectors(store, store.path)
    assert store.search("検索", channel_ids=[10])[0].id == "1"
    with pytest.raises(ValueError, match="separate"):
        import_vectors(store, store.path)


def test_vector_import_dimension_mismatch_and_private_file_guards(store, tmp_path):
    bundle = vector_bundle(tmp_path)
    store.upsert(record(1, text="semantic original"))
    with store.connect() as db:
        db.execute("INSERT INTO message_vectors VALUES (?, ?, ?, ?, ?)",
                   ("1", store.model, "old", 3, b"\x00\x3c" * 3))
    with pytest.raises(ValueError, match="dimensions differ"):
        import_vectors(store, bundle)
    bundle.chmod(0o644)
    with pytest.raises(PermissionError):
        import_vectors(store, bundle)


@pytest.mark.parametrize("lexical,dense,expected", [
    ([], [], []), ([], ["d"], ["d"]), (["a"], [], ["a"]),
    (list("abcdefgh"), ["d1", "a", "d2", "d3"],
     ["a", "b", "c", "d", "e", "d1", "f", "d2", "g", "d3", "h"]),
])
def test_partial_fusion_preserves_top_five_and_lexical_order(lexical, dense, expected):
    rows = [{"id": value} for value in lexical]
    assert _fusion("query", rows, dense, 20, partial=True) == expected
    assert _fusion("query", rows, dense, 3, partial=True) == expected[:3]


def test_partial_dense_keeps_lexical_order_and_restores_full_fusion(store):
    store = dense_store(store)
    store.upsert_many([record(i, text="質問 lexical") for i in range(10, 17)])
    store.upsert_many([record(1, text="semantic first"), record(2, text="semantic second")])
    lexical = [h.id for h in store.search("質問", channel_ids=[10], limit=20)]
    store.embed_pending()
    full = [h.id for h in store.search("質問", channel_ids=[10], limit=20)]
    with store.connect() as db:
        db.execute("DELETE FROM message_vectors WHERE message_id = ?", (lexical[-1],))
    partial = [h.id for h in store.search("質問", channel_ids=[10], limit=20)]
    assert partial[:5] == lexical[:5]
    assert [value for value in partial if value in lexical] == lexical
    assert partial[5] == "1" and partial[7] == "2"
    assert store.embed_pending() == 1
    assert [h.id for h in store.search("質問", channel_ids=[10], limit=20)] == full


@pytest.mark.parametrize("coverage,position", [(None, 5), (0.14, 5), (0.899, 5),
                                              (0.9, 0), (0.96, 0)])
def test_high_coverage_partial_fusion_promotes_dense_head(coverage, position):
    rows = [{"id": str(i), "author_name": "alice", "text": "トランザクション"}
            for i in range(8)]
    result = _fusion("Twitterのリクエストヘッダーの解析", rows, ["semantic"], 8,
                     partial=True, dense_coverage=coverage)
    assert result[position] == "semantic"
    assert len(result) == len(set(result)) == 8


@pytest.mark.parametrize("rows,dense,expected", [
    ([], [], []), ([], ["semantic"], ["semantic"]),
    ([{"id": "exact", "author_name": "alice", "text": "Rust"}], [], ["exact"]),
    ([{"id": "exact", "author_name": "alice", "text": "Rust"}],
     ["semantic"], ["exact", "semantic"]),
])
def test_high_coverage_partial_fusion_keeps_unembedded_exact_hits(rows, dense, expected):
    assert _fusion("Rust", rows, dense, 10, partial=True, dense_coverage=0.96) == expected


def test_high_coverage_partial_search_retains_unembedded_lexical_hits(store):
    store = dense_store(store)
    store.upsert(record(1, text="semantic header analysis"))
    store.upsert_many([record(i, text="質問 lexical") for i in range(2, 8)])
    store.upsert_many([record(i, text="background filler") for i in range(8, 20)])
    assert store.embed_pending() == 19
    store.upsert(record(20, text="Rust unembedded exact"))
    assert store.search("質問 解析", channel_ids=[10], limit=8)[0].id == "1"
    assert store.search("Rust", channel_ids=[10], limit=8)[0].id == "20"
    assert list(store._dense.scope_coverage.values()) == [0.95]


def test_partial_coverage_respects_all_scope_filters_and_has_bounded_cache(store):
    store = dense_store(store)
    store.upsert_many([record(i, text="semantic") for i in range(10)])
    store.upsert(record(20, author="bob", text="semantic"))
    store.upsert_many([record(i, channel="11", text="semantic") for i in range(40, 60)])
    assert store.embed_pending() == 31
    store.upsert(record(10, text="unembedded exact"))
    store.upsert_many([record(i, author="bob", text="unembedded exact") for i in range(21, 30)])
    dense = store._dense
    dense.query_cache_size = 2

    def coverage(channels, *, author=None, **filters):
        with store.connect() as db:
            ids = store._author_ids(db, author, channels, explicit=True) if author else None
            where, args = store._scope(channels, author=ids, **filters)
            hits = dense.search(db, "質問", channels, where, args, 50,
                                author_ids=ids, **filters)
        assert hits is not None
        assert hits.partial == (hits.coverage < 1)
        return hits.coverage

    assert coverage([10]) == pytest.approx(11 / 21)
    assert coverage([10], author="alice") == pytest.approx(10 / 11)
    assert coverage([10], author="bob") == pytest.approx(1 / 10)
    for channel in ("10", "general", "#general"):
        assert coverage([10, 11], channel=channel) == pytest.approx(11 / 21)
    assert coverage([10, 11], channel="private") == 1
    assert coverage([10, 11], channel="missing") == 1
    assert coverage([10], before="2026-01-01T00:20Z") == pytest.approx(10 / 11)
    assert coverage([10], after="2026-01-01T00:19Z") == pytest.approx(1 / 10)
    assert coverage([10], after="2026-01-01T00:20Z") == 0
    assert coverage([10], before="2025-01-01") == 1
    assert coverage(
        [10, 11], author="alice", channel="#general", after="2025-12-31T23:59Z",
        before="2026-01-01T00:20Z",
    ) == pytest.approx(10 / 11)
    renamed = record(0, text="semantic")
    renamed["_channel"] = "renamed"
    store.upsert(renamed)
    assert coverage([10], channel="general") == 0.5
    assert coverage([10], channel="renamed") == 1
    assert len(dense.scope_coverage) == 2


def test_partial_coverage_cache_tracks_edits_tombstones_and_new_vectors(store):
    store = dense_store(store)
    store.upsert_many([record(i, text="semantic") for i in range(1, 11)])
    assert store.embed_pending() == 10
    store.upsert(record(11, text="unembedded exact"))
    dense = store._dense

    def coverage():
        store.search("質問", channel_ids=[10])
        return list(dense.scope_coverage.values())[-1]

    assert coverage() == pytest.approx(10 / 11)
    matrix = dense.matrix
    writer = dense_store(ServerLog(store.path))
    writer.upsert(record(1, text="edited semantic"))
    assert coverage() == pytest.approx(9 / 11)
    writer.mark_deleted("11")
    assert coverage() == 0.9
    assert writer.embed_pending() == 1
    assert coverage() == 1
    writer.upsert_many([record(12, bot=True), record(13, text="x"),
                        record(14, text=" "), record(15, text="deleted semantic")])
    writer.mark_deleted("15")
    assert coverage() == 1
    with writer.connect() as db:
        db.execute("UPDATE message_vectors SET text_hash = 'stale' WHERE message_id = '2'")
    assert coverage() == 0.9
    assert dense.matrix is matrix and dense.rebuilds == 1


def test_embedding_lookahead_is_bounded_and_resumes(store):
    from threading import Event

    backend = FakeEmbedder()
    store = dense_store(store, backend)
    records = []
    for i in range(1025):
        value = record(i, text="long document " * 20 if i < 1024 else "aa")
        value["timestamp"] = "2024-01-01T00:00:00+00:00"
        value["id"] = f"{i:04d}"
        records.append(value)
    store.upsert_many(records)
    stop = Event()

    def progress(count):
        if count == 1024:
            stop.set()

    # Pending rows embed newest first (id DESC breaks the shared timestamp),
    # so the "aa" record (id 1024) is in the first page and id 0000 remains.
    assert store._embed_batches(32, progress, stop) == 1024
    assert sum("long document" in text for text in backend.inputs) == 1023
    assert "passage: alice: aa" in backend.inputs
    assert store.embed_pending() == 1
    assert "long document" in backend.inputs[-1]


def test_embedding_buckets_use_unpadded_token_lengths_and_keep_row_mapping(store):
    class TokenEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(tokenizer=SimpleNamespace(encode_batch=self.tokenize))
            self.tokenized = []

        def tokenize(self, texts):
            self.tokenized.extend(texts)
            return [SimpleNamespace(attention_mask=[1] * (2 if "semantic" in text else 4)
                                    + [0] * (4 if "semantic" in text else 2)) for text in texts]

    backend = TokenEmbedder()
    store = dense_store(store, backend)
    store.upsert_many([record(1, text="plain"), record(2, text="semantic " * 5)])
    assert store.embed_pending(batch_size=1) == 2
    assert "semantic" in backend.inputs[0] and "plain" in backend.inputs[1]
    assert all(text.startswith("passage: alice: ") for text in backend.tokenized)
    with store.connect() as db:
        vectors = {row["message_id"]: row["vector"] for row in
                   db.execute("SELECT * FROM message_vectors")}
    assert vectors == {"1": b"\x00\x3c\x00\x00", "2": b"\x00\x00\x00\x3c"}


def test_embedding_rechecks_tombstone_after_lookahead(store):
    class DeletingEmbedder(FakeEmbedder):
        def embed(self, texts):
            with store.connect() as db:
                db.execute("INSERT OR IGNORE INTO deleted_messages VALUES ('1')")
            yield from super().embed(texts)

    store.upsert(record(1, text="semantic"))
    store = dense_store(store, DeletingEmbedder())
    assert store.embed_pending() == 0
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM message_vectors").fetchone()[0] == 0


def test_embedding_preserves_vector_imported_during_inference(store, tmp_path):
    bundle = vector_bundle(tmp_path)
    state = []

    def snapshot():
        with store.connect() as db:
            return (
                tuple(db.execute("SELECT * FROM message_vectors").fetchone()),
                db.execute("SELECT sequence FROM dense_state").fetchone()[0],
                db.execute("SELECT generation FROM log_state").fetchone()[0],
                tuple(db.execute("SELECT * FROM dense_changes").fetchone()),
            )

    class ImportingEmbedder(FakeEmbedder):
        def embed(self, texts):
            assert import_vectors(store, bundle)["imported"] == 1
            state.append(snapshot())
            yield from ([1.0, 0.0] for _ in texts)

    store.upsert(record(1, text="semantic original"))
    store = dense_store(store, ImportingEmbedder())
    assert store.embed_pending() == 0
    assert snapshot() == state[0]
    assert snapshot()[0][-1] == b"\x00\x00\x00\x3c"


@pytest.mark.parametrize("model,hash_matches", [("same", False), ("other", True)])
def test_embedding_does_not_skip_stale_or_other_model_vector(store, model, hash_matches):
    class RacingEmbedder(FakeEmbedder):
        def embed(self, texts):
            with store.connect() as db:
                row = db.execute("SELECT * FROM messages WHERE id = '1'").fetchone()
                db.execute("INSERT INTO message_vectors VALUES (?, ?, ?, ?, ?)", (
                    "1", store.model if model == "same" else "other/model",
                    text_hash(row) if hash_matches else "stale", 2, b"\x00\x3c\x00\x00",
                ))
            yield from super().embed(texts)

    store.upsert(record(1, text="semantic original"))
    store = dense_store(store, RacingEmbedder())
    assert store.embed_pending() == 1
    with store.connect() as db:
        row = db.execute("SELECT * FROM message_vectors WHERE model = ?", (store.model,)).fetchone()
        assert row["vector"] == b"\x00\x00\x00\x3c"
        assert row["text_hash"] == text_hash(db.execute("SELECT * FROM messages").fetchone())


def test_chunked_lexical_matches_single_shot_results(store, monkeypatch):
    store.upsert_many([
        record(1, text="検索エンジンの高速化について議論した"),
        record(2, text="埋め込みベクトルの正規化と類似度計算の話"),
        record(3, text="検索エンジンと埋め込みの両方を使う設計"),
        record(4, text="全然関係ない料理の話"),
    ])
    query = "検索エンジンと埋め込みベクトルを組み合わせた検索の設計について教えて"
    assert len(set(bigrams(query))) > ServerLog._LEXICAL_CHUNK_TOKENS
    chunked = store.search(query, channel_ids=[10], lexical_only=True)
    monkeypatch.setattr(store, "lexical_deadline_ms", 0)
    single = store.search(query, channel_ids=[10], lexical_only=True)
    assert {hit.id for hit in chunked} == {hit.id for hit in single}
    assert {hit.id for hit in chunked} >= {"1", "3"}


def test_lexical_soft_deadline_aborts_to_partial_rows(store, monkeypatch):
    store.upsert_many([
        record(1, text="検索エンジンの高速化について議論した"),
        record(2, text="埋め込みベクトルの正規化と類似度計算の話"),
        record(3, text="検索エンジンと埋め込みの両方を使う設計"),
    ])
    original = ServerLog._lexical_query
    calls = {"count": 0}

    def flaky(db, expression, where, args, pool):
        calls["count"] += 1
        if calls["count"] > 1:
            # What the progress handler does when the deadline passes.
            raise sqlite3.OperationalError("interrupted")
        return original(db, expression, where, args, pool)

    monkeypatch.setattr(ServerLog, "_lexical_query", staticmethod(flaky))
    store.lexical_deadline_ms = 450
    hits = store.search(
        "検索エンジンと埋め込みベクトルを組み合わせた検索の設計について教えて",
        channel_ids=[10], lexical_only=True,
    )
    # The second chunk was aborted; the first chunk's rows still answer.
    assert calls["count"] > 1 and hits
