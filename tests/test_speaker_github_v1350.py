"""v1.35.0: speaker grounding, GitHub metadata, prompt premise.

The 2026-11-27 #雑談 incident: tanuki-dev asked 「俺のgithub出して」
right after the owner had received his own GitHub, and the bot
handed tanuki-dev the owner's account; asked 「俺誰かわかる？」 it
resolved nobody at all (the self-reference regex did not match
俺＋誰） and credited him with Vyline and iMons from transcript and
Server Wiki claims — while tanuki-dev/vyline is a fork of testuser0627/vyline.
These tests pin the three fixes: first-person resolution reaches
the caller for the 誰 shape too, GitHub authorship is delivered as
deterministic API metadata (owner / fork / source), and the system
prompt carries the multi-person premise plus the fork rule.
"""

import asyncio

import pytest
from test_server_log import record

from nezubot_discrod import github_meta
from nezubot_discrod.prompts import LOG_GUIDANCE, SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import pre_retrieve, resolve_people
from nezubot_discrod.server_log_tools import LogTools

FORK_FACTS = {
    "full_name": "bob/vyline", "owner": "bob", "fork": True,
    "parent": "alice/vyline", "source": "alice/vyline", "description": "",
}
TOOL_FACTS = {
    "full_name": "alice/tool", "owner": "alice", "fork": False,
    "parent": None, "source": None, "description": "",
}
OWN_FACTS = {
    "full_name": "bob/ruin", "owner": "bob", "fork": False,
    "parent": None, "source": None, "description": "",
}


@pytest.fixture
def store(tmp_path):
    return ServerLog(tmp_path / "data" / "server_log.db")


@pytest.fixture
def fake_fetch(monkeypatch):
    calls = []
    table = {
        ("bob", "vyline"): ("ok", FORK_FACTS),
        ("alice", "tool"): ("ok", TOOL_FACTS),
        ("bob", "ruin"): ("ok", OWN_FACTS),
        ("gone", "repo"): ("missing", None),
        ("down", "repo"): ("error", None),
    }

    def fetch(owner, repo):
        calls.append((owner, repo))
        return table.get((owner.casefold(), repo.casefold()), ("missing", None))

    monkeypatch.setattr(github_meta, "fetch_repo", fetch)
    return calls


def seed_github_messages(store):
    store.upsert(record(1, author="bob",
                        text="https://github.com/bob/vyline バグ探して"))
    store.upsert(record(2, author="bob",
                        text="https://github.com/bob/vyline 直した"))
    store.upsert(record(3, author="bob",
                        text="https://github.com/alice/tool これすごい"))
    store.upsert(record(4, author="alice", text="自分の repo は内緒"))


# --- speaker grounding -------------------------------------------------


def test_self_reference_who_question_resolves_to_caller(store):
    # The incident's 「俺誰かわかる？」: before the fix the
    # self-reference pattern required a particle after the pronoun,
    # so 誰 defeated it and resolution fell through to nobody.
    seed_github_messages(store)
    assert resolve_people(store, [10], "俺誰かわかる？", caller_id="21") == ["21"]
    assert resolve_people(store, [10], "俺のgithub出して", caller_id="21") == ["21"]
    assert resolve_people(store, [10], "わたしは何者？", caller_id="20") == ["20"]


def test_named_person_still_resolves_over_self_default(store):
    # A bare name question is not a self question: the caller default
    # must not swallow it (mentions/aliases keep their priority).
    seed_github_messages(store)
    assert resolve_people(store, [10], "aliceについて教えて", caller_id="21") == ["20"]


# --- GitHub reference parsing ------------------------------------------


def test_extract_github_refs():
    refs = github_meta.extract_github_refs(
        "見て https://github.com/tanuki-dev/vyline/issues/3 と "
        "github.com/testuser0627 あと github.com/settings/profile "
        "https://github.com/tanuki-dev/iMons_web/releases/tag/v1.9.07 "
        "github.com/tanuki-dev/vyline.git"
    )
    assert refs == [
        ("tanuki-dev", "vyline"),
        ("testuser0627", None),
        ("tanuki-dev", "iMons_web"),
    ]
    assert github_meta.extract_github_refs("リンクなし") == []
    assert github_meta.extract_github_refs(None) == []


# --- GitHub metadata block ----------------------------------------------


def test_person_block_traces_fork_to_maker(store, fake_fetch):
    seed_github_messages(store)
    block = github_meta.person_github_block(store, [10], "21")
    assert block is not None
    assert "GitHub情報（確定情報）" in block
    # The account comes from the person's own repo links (2 posts).
    assert "github.com/bob" in block
    # The fork is traced: bob holds it, alice made it.
    assert "元の repo は alice/vyline（製作者は alice）" in block
    assert "bob は fork の保持者であり、この repo の製作者ではない" in block
    # A repo bob merely pasted is alice's work, not bob's.
    assert "製作者は alice であってこの人物ではない" in block


def test_person_block_own_repo_is_not_a_fork(store, fake_fetch):
    store.upsert(record(1, author="bob", text="https://github.com/bob/ruin 作った"))
    store.upsert(record(2, author="bob", text="https://github.com/bob/ruin 更新"))
    block = github_meta.person_github_block(store, [10], "21")
    assert "repo bob/ruin: 所有者はこの人物（bob）。fork ではない。" in block


def test_text_block_marks_missing_and_error(store, fake_fetch):
    block = github_meta.text_github_block(
        store, "https://github.com/gone/repo と https://github.com/down/repo")
    assert "repo gone/repo: GitHub API では確認できなかった" in block
    assert "製作者は不明として扱うこと" in block
    assert "repo down/repo: メタデータを取得できなかった" in block
    assert "製作者は未確認として扱うこと" in block
    assert github_meta.text_github_block(store, "URL なし") is None


def test_repo_meta_caches_ok_and_missing_but_not_errors(store, fake_fetch):
    assert github_meta.repo_meta(store, "bob", "vyline") == ("ok", FORK_FACTS)
    assert github_meta.repo_meta(store, "bob", "vyline") == ("ok", FORK_FACTS)
    assert fake_fetch.count(("bob", "vyline")) == 1
    assert github_meta.repo_meta(store, "gone", "repo") == ("missing", None)
    assert github_meta.repo_meta(store, "gone", "repo") == ("missing", None)
    assert fake_fetch.count(("gone", "repo")) == 1
    # Errors are not cached: a transient failure retries next time.
    github_meta.repo_meta(store, "down", "repo")
    github_meta.repo_meta(store, "down", "repo")
    assert fake_fetch.count(("down", "repo")) == 2


# --- pre_retrieve integration: the incident shape ------------------------


def test_pre_retrieve_delivers_github_block_for_self_github_question(
        store, fake_fetch):
    seed_github_messages(store)
    log = LogTools(store, [10], caller_id=21)
    result = asyncio.run(pre_retrieve(log, "俺のgithub出して", timeout=5))
    assert result.diagnostics["people"] == ["21"]
    assert result.diagnostics["github_delivered"] is True
    text = str(result)
    assert "GitHub情報（確定情報）" in text
    assert "元の repo は alice/vyline（製作者は alice）" in text


def test_pre_retrieve_no_github_block_without_github_topic(store, fake_fetch):
    seed_github_messages(store)
    log = LogTools(store, [10], caller_id=21)
    result = asyncio.run(pre_retrieve(log, "俺は誰", timeout=5))
    assert result.diagnostics["github_delivered"] is False


# --- prompt premise -------------------------------------------------------


def test_prompt_carries_multi_person_premise_and_fork_rule():
    assert "multi-person group chat" in SYSTEM
    assert "First-person words in the latest message" in SYSTEM
    assert "when the speaker changes, re-anchor" in SYSTEM
    assert "GitHub情報（確定情報）" in LOG_GUIDANCE
    assert "the person holding the fork is not the maker" in LOG_GUIDANCE
    assert "never from the GitHub fact the previous speaker just received" \
        in LOG_GUIDANCE
