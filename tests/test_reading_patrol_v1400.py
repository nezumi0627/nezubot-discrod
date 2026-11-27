"""v1.40.0: the second half of reading time patrols the
link-sharing channel. All fixtures are synthetic (dummy channel,
message and repository names)."""

import json
import random
from datetime import datetime

from nezubot_discrod import reading
from nezubot_discrod.reading import (
    PatrolSession,
    ReadingStore,
    collect_candidates,
    extract_urls,
    normalize_github_repo,
)

WALL = datetime(2026, 11, 27, 12, 45, tzinfo=reading.JST)
CHANNEL_ID = "123456789012345678"
REPO_URL = "https://github.com/tanuki-dev/agent-kit"
API_URL = "https://api.github.com/repos/tanuki-dev/agent-kit"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def msg(mid, content, *, author="巡回テスト", embeds=None):
    return {
        "id": str(mid), "content": content,
        "timestamp": "2026-11-27T00:00:00+00:00",
        "author": {"username": author, "global_name": author},
        "embeds": embeds or [],
    }


class FakeSource:
    """Canned Discord source. ``messages`` are stored newest-first,
    exactly as the REST API returns them."""

    def __init__(self, messages, channel_name="リンク共有"):
        self.messages = list(messages)
        self.channel_name = channel_name
        self.after_calls = []

    def fetch_channel(self, channel_id):
        return {"id": channel_id, "name": self.channel_name}

    def fetch_messages(self, channel_id, *, after=None, limit=100):
        self.after_calls.append(after)
        if after is None:
            return list(self.messages[:limit])
        newer = [m for m in self.messages if int(m["id"]) > int(after)]
        return newer[:limit]


class FakeWeb:
    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "read_web_page":
            return json.dumps({
                "url": args["url"],
                "text": "ページ本文: エージェント基盤の実装と解説。" * 5,
                "truncated": False,
            })
        raise AssertionError(name)


class FakePatrolBrain:
    """Reads the repo page, then its API record, then finishes.
    The finalize keeps the repo as interesting and also tries to
    smuggle in a candidate that was never opened."""

    def __call__(self, system, payload, timeout):
        if system == reading.PATROL_FINALIZE_SYSTEM:
            return json.dumps({
                "learned": "エージェント基盤の repo が共有されていた。",
                "books": [
                    {"title": "tanuki-dev/agent-kit", "url": REPO_URL,
                     "summary": "AI エージェント基盤の実装",
                     "key_points": ["スター 42", "最終更新は今週"],
                     "interesting": True,
                     "interesting_reason": "エージェントの文脈管理の具体的な"
                                           "手法が bot 開発に応用できる"},
                    {"title": "開いていない候補", "url": "https://github.com/"
                     "tanuki-dev/never-opened",
                     "summary": "捏造", "key_points": [],
                     "interesting": True,
                     "interesting_reason": "開いていないので残らない"},
                ],
            })
        assert system == reading.PATROL_SYSTEM
        data = json.loads(payload)
        event = data["last_event"]
        if event["type"] == "start":
            return json.dumps({"action": {"type": "read", "url": REPO_URL}})
        if event["type"] == "page" and event["url"] == REPO_URL:
            return json.dumps({
                "digest": {"title": "tanuki-dev/agent-kit",
                           "summary": "AI エージェント基盤の実装",
                           "key_points": ["文脈管理の実装がある"]},
                "action": {"type": "read", "url": API_URL},
            })
        if event["type"] == "page" and event["url"] == API_URL:
            return json.dumps({
                "digest": {"title": "tanuki-dev/agent-kit",
                           "summary": "",
                           "key_points": ["スター 42", "最終更新は今週",
                                          "言語は Python"]},
                "action": {"type": "finish"},
            })
        raise AssertionError(event)


def make_patrol(tmp_path, source, brain=None, web=None):
    store = ReadingStore(tmp_path / "reading.db")
    return PatrolSession(
        store, brain or FakePatrolBrain(), web or FakeWeb(),
        tmp_path / "reports", source=source, channel_id=CHANNEL_ID,
        backend="devin", model="swe-2-1",
        clock=FakeClock(), wall=lambda: WALL, rng=random.Random(0))


# ---------------------------------------------------------------------------
# URL / candidate extraction
# ---------------------------------------------------------------------------


def test_extract_urls_strips_trailing_punctuation():
    text = f"これ見て {REPO_URL} 。 あと https://example.com/a, 次も"
    assert extract_urls(text) == [REPO_URL, "https://example.com/a"]


def test_normalize_github_repo_folds_subpaths():
    assert normalize_github_repo(REPO_URL + "/issues/12") == (
        "tanuki-dev", "agent-kit", REPO_URL)
    assert normalize_github_repo(REPO_URL + ".git") == (
        "tanuki-dev", "agent-kit", REPO_URL)
    assert normalize_github_repo("https://github.com/settings/profile") is None
    assert normalize_github_repo("https://github.com/tanuki-dev") is None
    assert normalize_github_repo("https://example.com/tanuki-dev/agent-kit") is None


def test_collect_candidates_dedupes_content_and_embeds():
    messages = [
        msg(100, f"作った {REPO_URL}",
            embeds=[{"url": REPO_URL}]),
        msg(101, "動画 https://www.youtube.com/watch?v=xxxx"),
        msg(102, f"再掲 {REPO_URL}/tree/main"),
    ]
    candidates, consumed, stats = collect_candidates(messages, [])
    assert [c["url"] for c in candidates] == [REPO_URL]
    assert candidates[0]["kind"] == "repo"
    assert candidates[0]["api_url"] == API_URL
    assert consumed == "102"
    assert stats["messages"] == 3 and stats["repo_links"] == 3


def test_collect_candidates_skips_repos_already_in_library():
    messages = [msg(100, REPO_URL)]
    candidates, consumed, _stats = collect_candidates(messages, [REPO_URL])
    assert candidates == []
    assert consumed == "100"


def test_collect_candidates_falls_back_to_links_without_repos():
    messages = [msg(100, "記事 https://example.com/post/1")]
    candidates, _consumed, _stats = collect_candidates(messages, [])
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "link"
    assert candidates[0]["url"] == "https://example.com/post/1"


def test_collect_candidates_cap_leaves_the_rest_for_next_time():
    messages = [
        msg(100, "https://github.com/tanuki-dev/one"),
        msg(101, "https://github.com/tanuki-dev/two"),
        msg(102, "https://github.com/tanuki-dev/three"),
    ]
    candidates, consumed, _stats = collect_candidates(
        messages, [], max_candidates=2)
    assert [c["title"] for c in candidates] == [
        "tanuki-dev/one", "tanuki-dev/two"]
    assert consumed == "101"


# ---------------------------------------------------------------------------
# Store kind column
# ---------------------------------------------------------------------------


def test_store_records_book_kind_and_migrates(tmp_path):
    store = ReadingStore(tmp_path / "reading.db")
    store.add_book(
        title="通常の本", url="https://example.com/book", theme="テーマ",
        summary="要約", key_points=[], read_at="2026-11-27T12:00:00+09:00")
    store.add_book(
        title="tanuki-dev/agent-kit", url=REPO_URL,
        theme=reading.PATROL_THEME, summary="repo の要約", key_points=[],
        read_at="2026-11-27T12:45:00+09:00", kind="repo",
        interesting=True, interesting_reason="理由")
    books = store.recent_books(10)
    assert {book["url"]: book["kind"] for book in books} == {
        "https://example.com/book": "book", REPO_URL: "repo"}
    interesting = store.interesting_books_after(0)
    assert [book["url"] for book in interesting] == [REPO_URL]
    assert interesting[0]["kind"] == "repo"
    assert interesting[0]["theme"] == reading.PATROL_THEME


# ---------------------------------------------------------------------------
# Patrol session end to end
# ---------------------------------------------------------------------------


def test_patrol_first_run_keeps_repo_and_advances_watermark(tmp_path):
    # Newest-first, as the API returns them.
    source = FakeSource([
        msg(103, "音楽 https://www.youtube.com/watch?v=xxxx"),
        msg(102, f"新作 {REPO_URL}", embeds=[{"url": REPO_URL}]),
        msg(101, "おはよう"),
    ])
    patrol = make_patrol(tmp_path, source)
    result = patrol.run()
    assert result.channel_name == "リンク共有"
    assert result.messages_scanned == 3
    assert result.candidates == 1
    assert result.pages_read == 1
    assert result.books_kept == 1
    store = patrol.store
    assert store.get_state(reading.PATROL_STATE_KEY) == "103"
    books = store.recent_books(10)
    assert len(books) == 1
    book = books[0]
    assert book["kind"] == "repo"
    assert book["theme"] == reading.PATROL_THEME
    assert book["interesting"] is True
    assert "応用できる" in book["interesting_reason"]
    # The nightly interesting collection picks the repo up as-is.
    assert [b["id"] for b in store.interesting_books_after(0)] == [book["id"]]
    report = (tmp_path / "reports" / "20261127-124500-reading.md")
    text = report.read_text(encoding="utf-8")
    assert "リンク共有チャンネルの周回" in text
    assert "tanuki-dev/agent-kit" in text


def test_patrol_second_run_uses_watermark_and_finds_nothing(tmp_path):
    source = FakeSource([msg(102, f"新作 {REPO_URL}")])
    patrol = make_patrol(tmp_path, source)
    patrol.run()
    source2 = FakeSource([msg(102, f"新作 {REPO_URL}")])
    patrol2 = make_patrol(tmp_path, source2)
    result = patrol2.run()
    assert source2.after_calls == ["102"]
    assert result.candidates == 0
    assert result.books_kept == 0
    assert len(patrol2.store.recent_books(10)) == 1


def test_patrol_never_opens_urls_outside_the_candidates(tmp_path):
    class NosyBrain:
        def __call__(self, system, payload, timeout):
            if system == reading.PATROL_FINALIZE_SYSTEM:
                return json.dumps({"learned": "なし", "books": []})
            data = json.loads(payload)
            event = data["last_event"]
            if event["type"] == "start":
                return json.dumps({"action": {
                    "type": "read", "url": "https://example.com/evil"}})
            return json.dumps({"action": {"type": "finish"}})

    source = FakeSource([msg(102, f"新作 {REPO_URL}")])
    web = FakeWeb()
    patrol = make_patrol(tmp_path, source, brain=NosyBrain(), web=web)
    result = patrol.run()
    assert result.pages_read == 0
    assert web.calls == []
