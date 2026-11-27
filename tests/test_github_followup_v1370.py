"""v1.37.0: the GitHub block follows the conversation topic.

The 2026-11-27 08:45 #コマンド incident: right after the owner's
「おれのGitHubおしえて」 was answered correctly, his elliptical
follow-up 「タヌキのは」 resolved tanuki-dev correctly — but the
GitHub block was never attempted, because v1.35.0's scheduling
keyed on the raw text alone and the follow-up carries no GitHub
wording. Left with no GitHub fact for the resolved person, the
model's (correct) anti-mixing caution collapsed into a refusal:
「このターンの記録からは確認できない。さっきのテストユーザーのアカウント
とは別だから、混ぜて答えるのはやめとく」.

These tests pin the fix: detection reads the rewritten query and
the context query like the first-activity signals do, the block
is fetched for the newly resolved person on a topic-carried
turn, a missed first pass gets one extension retry, and a long
self-contained message does not inherit the topic.
"""

import asyncio

import pytest
from test_server_log import record

from nezubot_discrod import github_meta
from nezubot_discrod.prompts import LOG_GUIDANCE
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import (
    ROUTE_BUDGETS_MS,
    _assess_sufficiency,
    _github_signals,
    pre_retrieve,
)
from nezubot_discrod.server_log_tools import LogResult, LogTools

FORK_FACTS = {
    "full_name": "tanuki-dev/vyline", "owner": "tanuki-dev", "fork": True,
    "parent": "testuser0627/vyline", "source": "testuser0627/vyline",
    "description": "",
}
GUNSHOT_FACTS = {
    "full_name": "tanuki-dev/gunshot", "owner": "tanuki-dev", "fork": False,
    "parent": None, "source": None, "description": "",
}
VYLINE_FACTS = {
    "full_name": "testuser0627/vyline", "owner": "testuser0627",
    "fork": False, "parent": None, "source": None, "description": "",
}


@pytest.fixture
def store(tmp_path):
    return ServerLog(tmp_path / "data" / "server_log.db")


@pytest.fixture
def fake_fetch(monkeypatch):
    table = {
        ("tanuki-dev", "vyline"): ("ok", FORK_FACTS),
        ("tanuki-dev", "gunshot"): ("ok", GUNSHOT_FACTS),
        ("testuser0627", "vyline"): ("ok", VYLINE_FACTS),
    }

    def fetch(owner, repo):
        return table.get(
            (owner.casefold(), repo.casefold()), ("missing", None))

    monkeypatch.setattr(github_meta, "fetch_repo", fetch)
    return table


def seed_people(store):
    # タヌキ (id 21) points at tanuki-dev's repos in his own messages;
    # the caller alice (id 20) points at testuser0627's — the
    # production shape, where the two accounts must not be mixed.
    store.upsert(record(1, author="タヌキ",
                        text="https://github.com/tanuki-dev/vyline 直した"))
    store.upsert(record(2, author="タヌキ",
                        text="https://github.com/tanuki-dev/vyline 見て"))
    store.upsert(record(3, author="タヌキ",
                        text="https://github.com/tanuki-dev/gunshot 作った"))
    store.upsert(record(4, author="alice",
                        text="https://github.com/testuser0627/vyline すごい"))
    store.upsert(record(5, author="alice",
                        text="https://github.com/testuser0627/vyline また見た"))


# --- detection signals ---------------------------------------------------


def test_signals_direct_wording_and_bare_url():
    direct, topic, _ = _github_signals("俺のgithub出して")
    assert direct and topic
    direct, topic, _ = _github_signals(
        "https://github.com/tanuki-dev/vyline 見て")
    assert direct and topic


def test_signals_followup_inherits_topic_from_context_query():
    # The incident shape: the follow-up names only the person; the
    # GitHub topic rides in on the context query (the speaker's own
    # previous line, exactly as production built it).
    direct, topic, dependent = _github_signals(
        "タヌキのは", None, "おれのGitHubおしえて タヌキのは")
    assert not direct
    assert dependent and topic


def test_signals_no_context_no_topic():
    direct, topic, _ = _github_signals("タヌキのは")
    assert not direct and not topic


def test_signals_long_self_contained_message_does_not_inherit():
    direct, topic, dependent = _github_signals(
        "タヌキの最近の開発活動について詳しく教えてほしい",
        None, "おれのGitHubおしえて")
    assert not direct and not dependent and not topic


# --- sufficiency: one retry for a missing block --------------------------


def test_sufficiency_forces_one_github_retry():
    verdict = _assess_sufficiency(
        {}, ["21"], github_topic=True)
    assert verdict["sufficient"] is False
    assert "github" in verdict["weak"]
    verdict = _assess_sufficiency(
        {}, ["21"], github_topic=True, github_retried=True)
    assert "github" not in verdict["weak"]
    verdict = _assess_sufficiency(
        {"github_facts": [LogResult("block")]}, ["21"],
        github_topic=True)
    assert "github" not in verdict["weak"]
    verdict = _assess_sufficiency({}, ["21"])
    assert "github" not in verdict["weak"]


# --- pre_retrieve: the incident shape ------------------------------------


def test_followup_turn_delivers_new_persons_github_block(
        store, fake_fetch):
    seed_people(store)
    log = LogTools(store, [10], caller_id=20)
    result = asyncio.run(pre_retrieve(
        log, "タヌキのは", timeout=5,
        context_query="おれのGitHubおしえて タヌキのは"))
    diagnostics = result.diagnostics
    assert diagnostics["people"] == ["21"]
    assert diagnostics["github_topic"] is True
    assert diagnostics["github_delivered"] is True
    text = str(result)
    # The block is the NEW person's, fetched this turn...
    assert "この人物の GitHub アカウント: github.com/tanuki-dev" in text
    assert "元の repo は testuser0627/vyline（製作者は testuser0627）" in text
    # ...and the caller's account is not presented as theirs.
    assert "この人物の GitHub アカウント: github.com/testuser0627" not in text


def test_followup_without_github_conversation_delivers_no_block(
        store, fake_fetch):
    seed_people(store)
    log = LogTools(store, [10], caller_id=20)
    result = asyncio.run(pre_retrieve(log, "タヌキのは", timeout=5))
    assert result.diagnostics["people"] == ["21"]
    assert result.diagnostics["github_topic"] is False
    assert result.diagnostics["github_delivered"] is False


def test_long_question_does_not_inherit_github_topic(store, fake_fetch):
    seed_people(store)
    log = LogTools(store, [10], caller_id=20)
    result = asyncio.run(pre_retrieve(
        log, "タヌキの最近の開発活動について詳しく教えてほしい",
        timeout=5, context_query="おれのGitHubおしえて"))
    assert result.diagnostics["github_topic"] is False
    assert result.diagnostics["github_delivered"] is False


def test_github_block_retried_after_first_pass_failure(
        store, monkeypatch):
    # The phase-1 build dies on a transient fetch failure; the
    # sufficiency gate must spend one extension retry on the block
    # instead of leaving the turn without its decisive evidence.
    seed_people(store)
    table = {
        ("tanuki-dev", "vyline"): ("ok", FORK_FACTS),
        ("tanuki-dev", "gunshot"): ("ok", GUNSHOT_FACTS),
        ("testuser0627", "vyline"): ("ok", VYLINE_FACTS),
    }
    calls = {"count": 0}

    def flaky_fetch(owner, repo):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient API failure")
        return table.get(
            (owner.casefold(), repo.casefold()), ("missing", None))

    monkeypatch.setattr(github_meta, "fetch_repo", flaky_fetch)
    log = LogTools(store, [10], caller_id=20)
    result = asyncio.run(pre_retrieve(
        log, "タヌキのgithub出して", timeout=5,
        budgets=ROUTE_BUDGETS_MS))
    assert calls["count"] >= 2
    assert result.diagnostics["github_delivered"] is True
    assert "github.com/tanuki-dev" in str(result)


# --- prompt: the lookup stays owed on a follow-up -------------------------


def test_prompt_keeps_followup_lookup_owed():
    assert "the lookup is still owed" in LOG_GUIDANCE
    assert "an unattempted one is not evidence" in LOG_GUIDANCE
    # The v1.35.0 anti-mixing rules stay in force.
    assert "the person holding the fork is not the maker" in LOG_GUIDANCE
    assert "never from the GitHub fact the previous speaker just received" \
        in LOG_GUIDANCE
