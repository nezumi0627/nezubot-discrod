import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from nezubot_discrod.member_wiki import MAX_CONTENT, MAX_PAGES, MAX_REVISIONS, MemberWiki, WikiError


def test_persistent_wiki_supports_append_search_rewrite_and_revisions(tmp_path):
    path = tmp_path / "data" / "wiki.db"
    pages = MemberWiki(path).for_member(1, 10)
    assert json.loads(pages.call("wiki_list", {})) == []
    assert json.loads(pages.call("wiki_append", {"page": "profile", "text": "# 好きな言語"})) == {
        "page": "profile", "revision": 1,
    }
    pages.call("wiki_append", {"page": "profile", "text": "Rust"})
    pages = MemberWiki(path).for_member(1, 10)
    assert json.loads(pages.call("wiki_search", {"query": "RUST"})) == [
        {"page": "profile", "revision": 2}
    ]
    assert json.loads(pages.call("wiki_search", {"query": "好きな言語"}))
    assert json.loads(pages.call("wiki_list", {})) == [{"page": "profile", "revision": 2}]
    original = json.loads(pages.call("wiki_read", {"page": "profile"}))
    assert original["body"] == "# 好きな言語\n\nRust"
    pages.call("wiki_rewrite", {"page": "profile", "text": "# 好きな言語\nPython", "revision": 2})
    assert json.loads(pages.call("wiki_read", {"page": "profile", "revision": 2}))["body"] == (
        "# 好きな言語\n\nRust"
    )
    assert pages.context() == [{"page": "profile", "excerpt": "# 好きな言語\nPython"}]
    with pytest.raises(WikiError, match="revision changed"):
        pages.call("wiki_rewrite", {"page": "profile", "text": "stale", "revision": 2})
    assert path.stat().st_mode & 0o777 == 0o600


def test_members_guilds_and_dm_are_isolated(tmp_path):
    wiki = MemberWiki(tmp_path / "wiki.db")
    wiki.for_member(1, 10).call("wiki_append", {"page": "profile", "text": "Rust"})
    for pages in [wiki.for_member(2, 10), wiki.for_member(1, 11), wiki.for_member(1, None)]:
        assert pages.context() == []
        with pytest.raises(WikiError, match="not found"):
            pages.call("wiki_read", {"page": "profile"})


def test_structured_fact_updates_keep_other_content_and_revision_history(tmp_path):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    pages.call("wiki_append", {"page": "preferences", "text": "# Existing notes\n\nTea"})

    def remember(key, text):
        return json.loads(pages.call("wiki_remember", {
            "page": "preferences", "key": key, "text": text,
        }))

    assert remember("drink", "Tea")["changed"] is True
    assert remember("editor", "Uses Vim") == {
        "page": "preferences", "revision": 3, "changed": True,
    }
    assert remember("editor", "uses   Ｖｉｍ")["changed"] is False
    assert remember("favorite-editor", "Uses Vim")["changed"] is True
    assert remember("editor", "Uses Neovim")["revision"] == 5
    latest = json.loads(pages.call("wiki_read", {"page": "preferences"}))
    assert latest["body"] == (
        "# Existing notes\n\nTea\n\n- **drink**: Tea\n\n"
        "- **editor**: Uses Neovim\n\n- **favorite-editor**: Uses Vim"
    )
    earlier = json.loads(pages.call("wiki_read", {"page": "preferences", "revision": 3}))
    assert "- **editor**: Uses Vim" in earlier["body"]


def test_wiki_search_and_context_rank_exact_titles_and_relevant_japanese(tmp_path):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    pages.call("wiki_append", {"page": "notes", "text": "Profile and other notes."})
    pages.call("wiki_append", {"page": "profile", "text": "Introduction."})
    pages.call("wiki_append", {
        "page": "preferences", "text": "Filler. " * 80 + "\n\n- **hobby**: ボードゲームとRust",
    })
    assert [hit["page"] for hit in json.loads(pages.call("wiki_search", {"query": "PROFILE"}))] == [
        "profile", "notes",
    ]
    assert json.loads(pages.call("wiki_search", {"query": "ＲＵＳＴは好き？"}))[0]["page"] == (
        "preferences"
    )
    context = pages.context("最近はボードゲームをしますか？")
    assert context[0]["page"] == "preferences"
    assert context[0]["excerpt"].startswith("- **hobby**: ボードゲームとRust")
    assert all(len(entry["excerpt"]) <= 240 for entry in context)


@pytest.mark.parametrize("overrides", [
    {"member_id": 2}, {"page": "misc"}, {"key": "../profile"}, {"text": "one\ntwo"},
])
def test_structured_facts_reject_scope_and_format_overrides(tmp_path, overrides):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    params = {"page": "profile", "key": "hobby", "text": "Grows roses"}
    with pytest.raises(WikiError):
        pages.call("wiki_remember", params | overrides)
    assert pages.context() == []


@pytest.mark.parametrize("filename", ["wiki.db", "wiki.db-journal", "wiki.db-wal", "wiki.db-shm"])
def test_wiki_refuses_sqlite_links_outside_its_data_root(tmp_path, monkeypatch, filename):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".nezubot-discrod"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    (root / filename).symlink_to(outside)
    pages = MemberWiki().for_member(1, 10)

    with pytest.raises(PermissionError):
        pages.call("wiki_append", {"page": "profile", "text": "No write"})
    assert not outside.exists()
    assert {file.name for file in root.iterdir()} == {filename}


def test_wiki_refuses_a_data_root_inside_self_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pages = MemberWiki(checkout=tmp_path).for_member(1, 10)
    with pytest.raises(PermissionError):
        pages.context()
    assert not (tmp_path / ".nezubot-discrod").exists()


def test_wiki_refuses_a_hard_link_into_self_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".nezubot-discrod"
    root.mkdir(mode=0o700)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    protected = checkout / "protected"
    protected.touch(mode=0o600)
    os.link(protected, root / "wiki.db")
    pages = MemberWiki(checkout=checkout).for_member(1, 10)

    with pytest.raises(PermissionError, match="hard links"):
        pages.call("wiki_append", {"page": "profile", "text": "No write"})
    assert protected.stat().st_size == 0


@pytest.mark.parametrize(
    "args",
    [
        {"page": "../profile", "text": "Rust"},
        {"page": "profile", "text": "Rust", "member_id": 2},
        {"page": "profile", "text": " "},
        {"page": "profile", "text": "a" * (MAX_CONTENT + 1)},
    ],
)
def test_invalid_writes_do_not_create_pages(tmp_path, args):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    with pytest.raises(WikiError):
        pages.call("wiki_append", args)
    assert pages.context() == []


def test_concurrent_appends_do_not_overwrite_and_page_size_is_bounded(tmp_path):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(
            lambda i: pages.call("wiki_append", {"page": "work", "text": str(i)}), range(8)
        ))
    current = json.loads(pages.call("wiki_read", {"page": "work"}))
    assert set(current["body"].split()) == {str(i) for i in range(8)}
    assert current["revision"] == 8
    with pytest.raises(WikiError, match="size limit"):
        pages.call("wiki_append", {"page": "work", "text": "x" * MAX_CONTENT})
    assert json.loads(pages.call("wiki_read", {"page": "work"})) == current


def test_page_count_and_revision_retention_are_bounded(tmp_path):
    pages = MemberWiki(tmp_path / "wiki.db").for_member(1, 10)
    for i in range(MAX_PAGES):
        pages.call("wiki_append", {"page": f"p{i}", "text": "original"})
    with pytest.raises(WikiError, match="page limit"):
        pages.call("wiki_append", {"page": "overflow", "text": "new"})
    for revision in range(1, MAX_REVISIONS + 1):
        pages.call("wiki_rewrite", {"page": "p0", "text": "new", "revision": revision})
    with pytest.raises(WikiError, match="not found"):
        pages.call("wiki_read", {"page": "p0", "revision": 1})
    assert json.loads(pages.call("wiki_read", {"page": "p0", "revision": 2}))["body"] == "new"
