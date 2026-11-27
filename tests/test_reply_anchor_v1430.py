"""v1.43.0: the reply anchor — a demonstrative in a reply binds to
the replied-to message, not to the surrounding conversation.

The 2026-11-27 13:04 incident: zero_dev asked 「俺誰かわかる？」 and
the bot answered about zero_dev correctly. tanuki_dev then replied to the
bot's answer with 「ギットハブ探して」 — and the bot answered with
kuma_dev's GitHub, because grounding fell through the bot-authored
reply target to the transcript's most-discussed person (tanuki_dev and
kuma_dev had been chatting heavily), the seed carried kuma_dev into
resolution, and the deterministic GitHub block was built for him.

These tests pin the fix: when the reply target is the bot's own
answer, the anchor is the person that answer was about — resolved
through the question that produced it (the caller's durable record
first, the transcript window as fallback) — and a name the
replied-to person declares for themselves (「俺の名前はX」) rides
along as the lookup key for that person.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nezubot_discrod import github_meta
from nezubot_discrod.conversation import (
    LABEL,
    build_conversation_context,
    declared_name,
)
from nezubot_discrod.pending_turns import (
    init_pending_turns,
    mark_answered,
    register_turn,
    trigger_for_answer,
)
from nezubot_discrod.prompts import LOG_GUIDANCE
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import pre_retrieve, resolve_people
from nezubot_discrod.server_log_tools import LogTools

JST = timezone(timedelta(hours=9))
BASE = datetime(2026, 11, 27, 13, 0, tzinfo=JST)


def user(user_id, name, *, bot=False):
    return SimpleNamespace(id=user_id, name=name, display_name=name, bot=bot)


BOT_USER = user(99, "nezubot-discrod", bot=True)
DOLLAR = user(401, "zero_dev")
TAMANE = user(402, "tanuki_server")
KUMA = user(403, "kuma_dev")
PIKO = user(404, "piko")


def reference(message_id, resolved=None):
    return SimpleNamespace(message_id=message_id, resolved=resolved)


def message(mid, author, content, *, minutes=0, mentions=(), reply_to=None,
            channel=None):
    return SimpleNamespace(
        id=mid,
        author=author,
        content=content,
        created_at=BASE + timedelta(minutes=minutes),
        mentions=list(mentions),
        reference=reply_to,
        channel=channel,
    )


class FakeChannel:
    def __init__(self, messages=()):
        self.id = 10
        self._messages = list(messages)

    async def fetch_message(self, message_id):
        for item in self._messages:
            if item.id == message_id:
                return item
        raise LookupError(message_id)

    async def history(self, *, limit=None, before=None):
        items = [
            item for item in self._messages
            if before is None or item.created_at < before.created_at
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        for item in items[:limit]:
            yield item


def incident_scene(anchor_author=TAMANE, *, with_origin=True):
    """The production shape: kuma_dev dominates the chatter, zero_dev asks
    who he is, the bot answers about zero_dev, and a follow-up replies to
    the bot's answer asking for "the GitHub"."""
    chatter1 = message(1, KUMA, "カーネルの話の続き", minutes=0)
    chatter2 = message(2, TAMANE, "それはそう", minutes=1,
                       reply_to=reference(1, resolved=chatter1))
    chatter3 = message(3, KUMA, "で、結論としては", minutes=2)
    question = message(4, DOLLAR, "<@99> 俺誰かわかる？", minutes=3,
                       mentions=[BOT_USER])
    answer = message(5, BOT_USER, "zero_devで、2026年3月3日にサーバー参加"
                     "してる人だね", minutes=4)
    correct = message(6, DOLLAR, "正解✅", minutes=5)
    anchor = message(7, anchor_author, "ギットハブ探して", minutes=6,
                     reply_to=reference(5, resolved=answer))
    channel = FakeChannel(
        [chatter1, chatter2, chatter3, question, answer, correct, anchor])
    for item in (chatter1, chatter2, chatter3, question, answer, correct,
                 anchor):
        item.channel = channel
    origin = question if with_origin else None
    return anchor, origin


def build(anchor, origin=None):
    return asyncio.run(
        build_conversation_context(anchor, BOT_USER.id, origin_call=origin))


# --- conversation grounding: the bot-answer anchor -----------------------


def test_bot_answer_anchor_via_origin_record():
    anchor, origin = incident_scene()
    context = build(anchor, origin)
    grounded = context["grounded_person"]
    assert grounded["id"] == str(DOLLAR.id)
    assert grounded["basis"] == "bot_answer_subject"


def test_bot_answer_anchor_via_history_fallback():
    # No durable record: the producing call is recovered from the
    # transcript window (the bot-mentioning message before the
    # answer), and the anchor still beats the transcript's topic.
    anchor, _ = incident_scene(with_origin=False)
    context = build(anchor)
    grounded = context["grounded_person"]
    assert grounded["id"] == str(DOLLAR.id)
    assert grounded["basis"] == "bot_answer_subject"
    assert grounded["id"] != str(KUMA.id)


def test_bot_answer_anchor_can_be_the_speaker():
    # The asker follows up on the bot's answer about himself: the
    # subject is the speaker, and the anchor still resolves to him
    # (the transcript fallbacks would never ground to the speaker).
    anchor, origin = incident_scene(anchor_author=DOLLAR)
    context = build(anchor, origin)
    grounded = context["grounded_person"]
    assert grounded["id"] == str(DOLLAR.id)
    assert grounded["basis"] == "bot_answer_subject"


def test_direct_reply_target_unchanged():
    intro = message(1, DOLLAR, "自分のプロジェクト見て", minutes=0)
    anchor = message(2, TAMANE, "この人物のGitHub探して", minutes=1,
                     reply_to=reference(1, resolved=intro))
    channel = FakeChannel([intro, anchor])
    intro.channel = anchor.channel = channel
    context = build(anchor)
    grounded = context["grounded_person"]
    assert grounded["id"] == str(DOLLAR.id)
    assert grounded["basis"] == "reply_target"
    assert "declared_name" not in grounded


# --- declared names --------------------------------------------------------


def test_declared_name_extraction():
    assert declared_name("俺の名前はりんご") == "りんご"
    assert declared_name("俺の名前はりんごです") == "りんご"
    assert declared_name("私の名前は ringo-dev です") == "ringo-dev"
    assert declared_name("はじめまして、僕はpikoです") == "piko"
    assert declared_name("名前はまだ決めてない") is None
    assert declared_name("普通の発言です") is None
    assert declared_name("") is None


def test_reply_to_declaring_message_grounds_author_with_name():
    intro = message(1, PIKO, "俺の名前はりんご", minutes=0)
    anchor = message(2, TAMANE, "こいつのGitHubさがして", minutes=1,
                     reply_to=reference(1, resolved=intro))
    channel = FakeChannel([intro, anchor])
    intro.channel = anchor.channel = channel
    context = build(anchor)
    grounded = context["grounded_person"]
    assert grounded["id"] == str(PIKO.id)
    assert grounded["basis"] == "reply_target"
    assert grounded["declared_name"] == "りんご"
    assert context["reply_target"]["declared_name"] == "りんご"


# --- pending turns: the durable answer -> question route -------------------


@pytest.fixture
def store(tmp_path):
    return ServerLog(tmp_path / "data" / "server_log.db")


def test_trigger_for_answer_round_trip(store):
    init_pending_turns(store)
    register_turn(
        store, message_id=1001, channel_id=10, guild_id=100,
        author_id=401, content="俺誰かわかる？")
    assert trigger_for_answer(store, 2002) is None
    mark_answered(store, 1001, [2002])
    assert trigger_for_answer(store, 2002) == "1001"
    assert trigger_for_answer(store, 9999) is None


# --- resolution: a trusted seed may name the caller ------------------------


def rec(number, author_id, name, text, *, channel="10"):
    return {
        "id": str(number),
        "guild_id": "100",
        "channel_id": channel,
        "_channel": "general",
        "_parent": "development",
        "author": {"id": str(author_id), "username": name, "bot": False},
        "timestamp": f"2026-01-01T00:{number:02d}:00+00:00",
        "content": text,
    }


def test_trusted_seed_resolves_the_caller(store):
    store.upsert(rec(1, 32, "bob", "自分の発言"))
    # Untrusted (transcript-derived) seed naming the caller: ignored,
    # as before — nothing else in the text resolves, so nobody.
    assert resolve_people(
        store, [10], "ギットハブ探して", 32, seed=32) == []
    # Reply-anchor seed naming the caller: honoured.
    assert resolve_people(
        store, [10], "ギットハブ探して", 32, seed=32,
        seed_trusted=True) == ["32"]


# --- pre_retrieve: the incident shape, deterministically -------------------

AKI_FACTS = {
    "full_name": "aki-dev/proj", "owner": "aki-dev", "fork": False,
    "parent": None, "source": None, "description": "",
}
LOUD_FACTS = {
    "full_name": "loud-c/repo", "owner": "loud-c", "fork": False,
    "parent": None, "source": None, "description": "",
}
RINGO_FACTS = {
    "full_name": "ringo-dev/apple", "owner": "ringo-dev", "fork": False,
    "parent": None, "source": None, "description": "",
}


@pytest.fixture
def fake_fetch(monkeypatch):
    table = {
        ("aki-dev", "proj"): ("ok", AKI_FACTS),
        ("loud-c", "repo"): ("ok", LOUD_FACTS),
        ("ringo-dev", "apple"): ("ok", RINGO_FACTS),
    }

    def fetch(owner, repo):
        return table.get(
            (owner.casefold(), repo.casefold()), ("missing", None))

    monkeypatch.setattr(github_meta, "fetch_repo", fetch)
    return table


def seed_github_people(store):
    store.upsert(rec(1, 31, "aki", "https://github.com/aki-dev/proj 作った"))
    store.upsert(rec(2, 31, "aki", "https://github.com/aki-dev/proj 直した"))
    store.upsert(rec(3, 33, "loud", "https://github.com/loud-c/repo 見て"))
    store.upsert(rec(4, 33, "loud", "https://github.com/loud-c/repo また見た"))
    store.upsert(rec(5, 32, "bob", "雑談"))


def test_anchor_seed_delivers_answer_subjects_github_block(
        store, fake_fetch):
    seed_github_people(store)
    log = LogTools(store, [10], caller_id=32)
    result = asyncio.run(pre_retrieve(
        log, "ギットハブ探して", timeout=5,
        seed_person=31, seed_anchor=True))
    diagnostics = result.diagnostics
    assert diagnostics["people"] == ["31"]
    assert diagnostics["seed_anchor"] is True
    assert diagnostics["github_delivered"] is True
    text = str(result)
    assert "この人物の GitHub アカウント: github.com/aki-dev" in text
    assert "github.com/loud-c" not in text


def test_declared_name_account_block_rides_along(store, fake_fetch,
                                                 monkeypatch):
    store.upsert(rec(1, 34, "piko", "俺の名前はringo-dev"))
    store.upsert(rec(2, 34, "piko", "よろしく"))
    store.upsert(rec(3, 32, "bob", "雑談"))
    calls = []

    def api_get(path):
        calls.append(path)
        if path == "/users/ringo-dev":
            return "ok", {"login": "ringo-dev", "public_repos": 1}
        if path.startswith("/users/ringo-dev/repos"):
            return "ok", [{"name": "apple"}]
        return "missing", None

    monkeypatch.setattr(github_meta, "_api_get", api_get)
    log = LogTools(store, [10], caller_id=32)
    result = asyncio.run(pre_retrieve(
        log, "こいつのGitHubさがして", timeout=5,
        seed_person=34, seed_anchor=True, declared_name="ringo-dev"))
    diagnostics = result.diagnostics
    assert diagnostics["people"] == ["34"]
    assert diagnostics["declared_name"] is True
    assert diagnostics["github_delivered"] is True
    text = str(result)
    assert "github.com/ringo-dev" in text
    assert "ringo-dev/apple" in text
    assert calls and calls[0] == "/users/ringo-dev"


def test_declared_name_not_a_login_adds_no_api_call(store, fake_fetch,
                                                    monkeypatch):
    store.upsert(rec(1, 34, "piko", "俺の名前はりんご"))
    store.upsert(rec(2, 32, "bob", "雑談"))
    calls = []
    monkeypatch.setattr(
        github_meta, "_api_get",
        lambda path: calls.append(path) or ("missing", None))
    log = LogTools(store, [10], caller_id=32)
    result = asyncio.run(pre_retrieve(
        log, "こいつのGitHubさがして", timeout=5,
        seed_person=34, seed_anchor=True, declared_name="りんご"))
    assert result.diagnostics["people"] == ["34"]
    assert calls == []


def test_account_block_states_a_missing_account(store, monkeypatch):
    monkeypatch.setattr(
        github_meta, "_api_get", lambda path: ("missing", None))
    block = github_meta.account_block(store, "ghost-dev")
    assert block is not None
    assert "github.com/ghost-dev" not in block
    assert "確認できなかった" in block
    assert github_meta.account_block(store, "りんご") is None
    assert github_meta.plausible_login("ringo-dev") is True
    assert github_meta.plausible_login("りんご") is False
    assert github_meta.plausible_login("-leading") is False


# --- prompt -----------------------------------------------------------------


def test_prompt_carries_the_reply_anchor_rules():
    assert "Reply anchor" in LOG_GUIDANCE
    assert "bot_answer_subject" in LOG_GUIDANCE
    assert "declared_name" in LOG_GUIDANCE
    assert "the flow of nearby talk does not" in LOG_GUIDANCE
    assert "bot_answer_subject" in LABEL
    assert "declared_name" in LABEL
