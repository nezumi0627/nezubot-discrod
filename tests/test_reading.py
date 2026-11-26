import json
import random
from datetime import datetime

import pytest

import nezubot_discrod.reading as reading
from nezubot_discrod.reading import (
    ReadingSession,
    ReadingStore,
    books_chunk,
    build_bookshelf_embed,
    choose_theme,
    collect_bookshelf,
    due_slot,
    is_dev_question,
    make_brain,
    parse_times,
    search_books,
)

WALL = datetime(2026, 11, 25, 21, 30, tzinfo=reading.JST)

URL_A = "https://example.com/sqlite-wal"
URL_B = "https://example.com/sqlite-checkpoint"
URL_X = "https://example.com/never-read"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeWeb:
    """Canned web: one search page with two results, readable pages."""

    def __init__(self, clock=None, advance=0.0):
        self.clock = clock
        self.advance = advance
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, dict(args),
                           self.clock() if self.clock else None))
        if self.clock:
            self.clock.advance(self.advance)
        if name == "web_search":
            return json.dumps({"results": [
                {"title": "SQLite WAL モード詳解", "url": URL_A,
                 "snippet": "WAL の仕組み"},
                {"title": "チェックポイント入門", "url": URL_B,
                 "snippet": "チェックポイントの動き"},
            ], "note": ""})
        if name == "read_web_page":
            return json.dumps({
                "url": args["url"],
                "text": "WAL モードでは読み取りと書き込みが並行できる。" * 10,
                "truncated": False,
            })
        raise AssertionError(name)


class FakeBrain:
    """Scripted reader brain: search, read both results in order,
    then finish. The finalize reply keeps one real book and also
    tries to smuggle in a page that was never read."""

    def __init__(self, clock=None, advance=0.0, keep_books=True):
        self.clock = clock
        self.advance = advance
        self.keep_books = keep_books
        self.step_calls = 0

    def __call__(self, system, payload, timeout):
        if self.clock:
            self.clock.advance(self.advance)
        if system == reading.FINALIZE_SYSTEM:
            books = []
            if self.keep_books:
                books = [
                    {"title": "SQLite WAL モード詳解", "url": URL_A,
                     "summary": "WAL の並行読み書きの要点",
                     "key_points": ["読み取りは書き込みをブロックしない"],
                     "interesting": True,
                     "interesting_reason": "具体的な仕組みが bot の DB 設計に応用できる"},
                    {"title": "読んでいない本", "url": URL_X,
                     "summary": "捏造", "key_points": [],
                     "interesting": True,
                     "interesting_reason": "読んでいないので残らない"},
                ]
            return json.dumps({"learned": "WAL について学んだ。", "books": books})
        assert system == reading.READER_SYSTEM
        self.step_calls += 1
        data = json.loads(payload)
        event = data["last_event"]
        read_urls = [note["url"] for note in data["notes"]]
        if event["type"] == "start":
            return json.dumps({"action": {"type": "search", "query": "SQLite WAL"}})
        if event["type"] == "search_results":
            for result in event["results"]:
                if result["url"] not in read_urls:
                    return json.dumps(
                        {"action": {"type": "read", "url": result["url"]}})
            return json.dumps({"action": {"type": "finish"}})
        if event["type"] == "page":
            digest = {
                "title": "ページタイトル",
                "summary": "このページの要約",
                "key_points": ["要点1"],
            }
            if len(read_urls) >= 2:
                return json.dumps(
                    {"digest": digest, "action": {"type": "finish"}})
            return json.dumps({"digest": digest, "action": {
                "type": "search", "query": "SQLite WAL checkpoint"}})
        raise AssertionError(event)


def make_session(tmp_path, brain, web, clock=None, **kwargs):
    store = ReadingStore(tmp_path / "reading.db")
    return ReadingSession(
        store, brain, web, tmp_path / "reports",
        backend="devin", model="swe-2-1",
        clock=clock, wall=lambda: WALL, rng=random.Random(0), **kwargs)


def test_session_reads_deep_and_keeps_only_read_books(tmp_path):
    web = FakeWeb()
    result = make_session(tmp_path, FakeBrain(), web).run()
    assert result.pages_read == 2
    # The smuggled, never-read book is rejected; only the real one stays.
    assert result.books_kept == 1
    store = ReadingStore(tmp_path / "reading.db")
    books = store.recent_books()
    assert [book["url"] for book in books] == [URL_A]
    assert books[0]["title"] == "SQLite WAL モード詳解"
    # The interesting flag and its reason survive the round trip.
    assert books[0]["interesting"] is True
    assert books[0]["interesting_reason"] == "具体的な仕組みが bot の DB 設計に応用できる"
    assert [book["url"] for book in store.interesting_books_after(0)] == [URL_A]
    report = (tmp_path / "reports" / result.report_path.split("/")[-1])
    text = report.read_text(encoding="utf-8")
    assert result.theme in text
    assert URL_A in text and URL_B in text
    assert "WAL について学んだ" in text
    assert "『SQLite WAL モード詳解』" in text
    # The report has a dedicated interesting section with the reason.
    assert "## 特に面白かった本（interesting）" in text
    assert "⭐ 『SQLite WAL モード詳解』" in text
    assert "具体的な仕組みが bot の DB 設計に応用できる" in text


def test_session_without_harvest_keeps_no_books(tmp_path):
    brain = FakeBrain(keep_books=False)
    result = make_session(tmp_path, brain, FakeWeb()).run()
    assert result.pages_read == 2
    assert result.books_kept == 0
    assert ReadingStore(tmp_path / "reading.db").recent_books() == []
    report = (tmp_path / "reports" / result.report_path.split("/")[-1])
    assert "なし" in report.read_text(encoding="utf-8")


def test_session_never_starts_exploration_past_the_reserve(tmp_path):
    clock = FakeClock()
    web = FakeWeb(clock=clock, advance=30.0)
    brain = FakeBrain(clock=clock, advance=30.0)
    session = make_session(tmp_path, brain, web, clock=clock,
                           session_seconds=100, reserve_seconds=20)
    result = session.run()
    # Brain and web calls each burn 30s: after the first search the
    # clock is at 60s, the next brain reply lands at 90s with only
    # 10s left — inside the reserve — so the read it asked for must
    # never be started.
    assert result.pages_read == 0
    assert all(name != "read_web_page" for name, _args, _t in web.calls)
    assert all(started < 80 for _name, _args, started in web.calls)
    assert result.report_path  # the session still wraps up and reports


def test_session_dedupes_fragment_variants_of_a_read_url(tmp_path):
    reads = []

    class Web:
        def call(self, name, args):
            assert name == "read_web_page"
            reads.append(args["url"])
            return json.dumps({"url": args["url"], "text": "本文",
                               "truncated": False})

    steps = iter([
        {"action": {"type": "read", "url": URL_A}},
        {"digest": {"title": "t", "summary": "s", "key_points": []},
         "action": {"type": "read", "url": URL_A + "#section"}},
        {"action": {"type": "finish"}},
    ])

    def brain(system, payload, timeout):
        if system == reading.FINALIZE_SYSTEM:
            return json.dumps({"learned": "学んだ", "books": []})
        return json.dumps(next(steps))

    result = make_session(tmp_path, brain, Web()).run()
    assert reads == [URL_A]
    assert result.pages_read == 1


def test_choose_theme_avoids_recent_sessions(tmp_path):
    store = ReadingStore(tmp_path / "reading.db")
    recent = list(reading.THEME_SEEDS[:5])
    for theme in recent:
        session_id = store.start_session(
            theme=theme, started_at="2026-11-25T00:00:00+09:00",
            backend="devin", model="swe-2-1")
        store.finish_session(session_id, ended_at="2026-11-25T00:15:00+09:00",
                             pages_read=1, books_kept=0, report_path="x")
    for seed in range(20):
        assert choose_theme(store, random.Random(seed)) not in recent


def test_book_search_and_chunk_gating(tmp_path):
    store = ReadingStore(tmp_path / "reading.db")
    store.add_book(
        title="SQLite WAL モード詳解", url=URL_A, theme="SQLite の WAL モード",
        summary="WAL では読み取りと書き込みが並行できる",
        key_points=["チェックポイントで WAL を戻す"],
        read_at="2026-11-25T21:45:00+09:00")
    store.add_book(
        title="React レンダリング最適化", url=URL_B, theme="React の最適化",
        summary="メモ化と再レンダリングの制御",
        key_points=["useMemo の使いどころ"],
        read_at="2026-11-25T12:45:00+09:00")
    hits = search_books(store, "SQLite の WAL ってどういう仕組み？")
    assert [book["url"] for book in hits] == [URL_A]
    chunk = books_chunk(store.path, "SQLite の WAL ってどういう仕組み？")
    assert chunk is not None and URL_A in chunk and "SQLite WAL" in chunk
    # Not a development question: the library stays out of the turn.
    assert not is_dev_question("今日の天気はどうなる？")
    assert books_chunk(store.path, "今日の天気はどうなる？") is None
    # A development question with no matching book stays silent too.
    assert books_chunk(store.path, "Rust の所有権がわからない") is None
    assert books_chunk(tmp_path / "missing.db", "SQLite の WAL は？") is None


def test_bookshelf_collect_and_embed(tmp_path):
    store = ReadingStore(tmp_path / "reading.db")
    empty = build_bookshelf_embed(collect_bookshelf(store.path))
    assert "まだ本がありません" in empty.fields[0].name
    store.add_book(
        title="SQLite WAL モード詳解", url=URL_A, theme="SQLite の WAL モード",
        summary="WAL の要点", key_points=[],
        read_at="2026-11-25T21:45:00+09:00")
    embed = build_bookshelf_embed(collect_bookshelf(store.path))
    assert "SQLite WAL モード詳解" in embed.fields[0].name
    assert URL_A in embed.fields[0].value


def test_schedule_slots():
    times = parse_times("12:30,21:30")
    assert times == [(12, 30), (21, 30)]
    assert parse_times("") == [(12, 30), (21, 30)]
    morning = datetime(2026, 11, 25, 12, 0, tzinfo=reading.JST)
    assert due_slot(morning, times, None) is None
    afternoon = datetime(2026, 11, 25, 13, 0, tzinfo=reading.JST)
    assert due_slot(afternoon, times, None) == "2026-11-25 12:30"
    assert due_slot(afternoon, times, "2026-11-25 12:30") is None
    night = datetime(2026, 11, 25, 22, 0, tzinfo=reading.JST)
    # Both slots passed with only the first marked: the latest wins.
    assert due_slot(night, times, "2026-11-25 12:30") == "2026-11-25 21:30"
    assert due_slot(night, times, "2026-11-25 21:30") is None


def test_pre_retrieve_delivers_books_for_dev_questions(tmp_path):
    import asyncio

    from nezubot_discrod.server_log import ServerLog
    from nezubot_discrod.server_log_context import pre_retrieve
    from nezubot_discrod.server_log_tools import LogTools

    store = ServerLog(tmp_path / "server_log.db", guild_id="123")
    library = ReadingStore(tmp_path / "reading.db")
    library.add_book(
        title="SQLite WAL モード詳解", url=URL_A, theme="SQLite の WAL モード",
        summary="WAL では読み取りと書き込みが並行できる",
        key_points=["チェックポイントで WAL を戻す"],
        read_at="2026-11-25T21:45:00+09:00")
    log = LogTools(store, ["100"], caller_id="999")
    result = asyncio.run(pre_retrieve(log, "SQLite の WAL モードって何？"))
    assert result.diagnostics["books_delivered"] is True
    assert "SQLite WAL モード詳解" in str(result)
    assert URL_A in str(result)
    # A non-development question never receives the books chunk.
    result = asyncio.run(pre_retrieve(log, "今日の天気はどうなる？"))
    assert result.diagnostics["books_delivered"] is False
    assert "読書の時間に読んだ本から" not in str(result)


def test_make_brain_rejects_paid_lightning_model():
    with pytest.raises(ValueError):
        make_brain("devin", model="swe-2-1-lightning")
    with pytest.raises(ValueError):
        make_brain("carrier-pigeon")


_AI_MARKERS = (
    "AI", "LLM", "RAG", "モデル", "エージェント", "学習", "推論",
    "プロンプト",
)


def test_theme_seeds_are_ai_only():
    assert len(reading.THEME_SEEDS) >= 40
    for seed in reading.THEME_SEEDS:
        assert any(marker in seed for marker in _AI_MARKERS), seed
    # None of the old general-development seeds survive.
    assert not any("SQLite" in seed or "Kotlin" in seed
                   for seed in reading.THEME_SEEDS)
    # The prompts frame the reading as AI study, not development study.
    assert "artificial intelligence" in reading.READER_SYSTEM
    assert "software-development" not in reading.READER_SYSTEM
    assert "interesting" in reading.FINALIZE_SYSTEM


def test_store_migrates_interesting_columns_idempotently(tmp_path):
    import sqlite3
    from contextlib import closing

    path = tmp_path / "reading.db"
    # A store created by v1.22.0: no interesting columns yet.
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE reading_books ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "title TEXT NOT NULL, url TEXT NOT NULL, theme TEXT NOT NULL, "
            "summary TEXT NOT NULL, key_points TEXT NOT NULL, "
            "read_at TEXT NOT NULL, session_id INTEGER)")
        db.execute(
            "INSERT INTO reading_books "
            "(title, url, theme, summary, key_points, read_at) "
            "VALUES ('旧い本', ?, '旧テーマ', '要約', '[]', "
            "'2026-11-26T00:15:00+09:00')", (URL_A,))
    store = ReadingStore(path)
    books = store.recent_books()
    # Pre-migration rows read back as not interesting.
    assert books[0]["interesting"] is False
    assert books[0]["interesting_reason"] == ""
    new_id = store.add_book(
        title="新しい本", url=URL_B, theme="AI の記憶", summary="要約",
        key_points=[], read_at="2026-11-26T12:45:00+09:00",
        interesting=True, interesting_reason="記憶の具体手法が応用できる")
    # A reason without the flag is dropped at the store boundary too.
    store.add_book(
        title="地味な本", url=URL_X, theme="AI ニュース", summary="要約",
        key_points=[], read_at="2026-11-26T12:46:00+09:00",
        interesting=False, interesting_reason="残らない理由")
    fresh = {book["id"]: book for book in store.all_books()}
    assert fresh[new_id]["interesting"] is True
    assert fresh[new_id]["interesting_reason"] == "記憶の具体手法が応用できる"
    assert fresh[new_id + 1]["interesting_reason"] == ""
    # The collection query sees only interesting books past the watermark,
    # and a second connect (migration re-run) changes nothing.
    assert [book["id"] for book in store.interesting_books_after(0)] == [new_id]
    assert store.interesting_books_after(new_id) == []
    assert ReadingStore(path).recent_books()[0]["title"] == "地味な本"


def test_valid_books_normalizes_interesting(tmp_path):
    notes = [{"url": URL_A, "title": "t", "summary": "s", "key_points": []}]
    raw = [
        {"title": "t", "url": URL_A, "summary": "s", "key_points": [],
         "interesting": True, "interesting_reason": "  新規性がある  "},
    ]
    (book,) = ReadingSession._valid_books(raw, notes)
    assert book["interesting"] is True
    assert book["interesting_reason"] == "新規性がある"
    # A false flag clears any stray reason; a non-boolean truthy value
    # does not count as interesting.
    for flag in (False, "yes", 1, None):
        raw[0]["interesting"] = flag
        raw[0]["interesting_reason"] = "理由だけ残っている"
        (book,) = ReadingSession._valid_books(raw, notes)
        assert book["interesting"] is False
        assert book["interesting_reason"] == ""


def test_bookshelf_marks_interesting_books(tmp_path):
    store = ReadingStore(tmp_path / "reading.db")
    store.add_book(
        title="普通の本", url=URL_A, theme="AI ニュース", summary="要約",
        key_points=[], read_at="2026-11-26T12:45:00+09:00")
    store.add_book(
        title="面白い本", url=URL_B, theme="AI の記憶", summary="要約",
        key_points=[], read_at="2026-11-26T21:45:00+09:00",
        interesting=True, interesting_reason="エージェントの記憶設計に使える")
    embed = build_bookshelf_embed(collect_bookshelf(store.path))
    marked = [field for field in embed.fields if "面白い本" in field.name]
    plain = [field for field in embed.fields if "普通の本" in field.name]
    assert marked and marked[0].name.startswith("⭐")
    assert "interesting: エージェントの記憶設計に使える" in marked[0].value
    assert plain and not plain[0].name.startswith("⭐")
    assert "interesting" not in plain[0].value
