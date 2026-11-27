"""v1.41.0: unanswered-turn recovery.

Accepted turns are registered as pending in the server log DB and
marked answered on delivery, so a crash (VM replacement, restart)
between acceptance and delivery no longer loses the question: the
startup recovery pass re-answers what is still owed, adopting answers
that were posted just before the crash instead of duplicating them,
and sweeping the archive for calls that arrived while the bot was
down entirely. All fixtures are synthetic.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from nezubot_discrod.blacklist import add_to_blacklist, init_blacklist
from nezubot_discrod.bot import Bot, _run_unanswered_recovery
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.pending_turns import (
    claim_turn,
    claimed_answer_ids,
    expire_turns_before,
    get_turn,
    has_answered_duplicate,
    init_pending_turns,
    list_recoverable,
    mark_answered,
    next_trigger_id,
    register_turn,
    set_turn_state,
    sweep_candidates,
)
from nezubot_discrod.server_log import ServerLog

BOT_ID = 99
CHANNEL_ID = 1
AUTHOR_ID = 456


def settings():
    return Settings(
        "token", "key", "https://example.com", "test",
        SandboxSettings(), guild_id=123, owner_id=456,
    )


def make_store(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="123")
    init_pending_turns(store)
    return store


def register(store, message_id, *, author_id=AUTHOR_ID, content="question",
             channel_id=CHANNEL_ID):
    return register_turn(
        store,
        message_id=message_id,
        channel_id=channel_id,
        guild_id=123,
        author_id=author_id,
        content=content,
    )


# ---------------------------------------------------------------- store


def test_register_and_mark_answered_roundtrip(tmp_path):
    store = make_store(tmp_path)
    assert register(store, 5000) is True
    assert register(store, 5000) is False  # already tracked
    row = get_turn(store, 5000)
    assert row["state"] == "pending"
    assert row["content"] == "question"
    assert mark_answered(store, 5000, [6001, 6002]) is True
    row = get_turn(store, 5000)
    assert row["state"] == "answered"
    assert row["answer_message_ids"] == '["6001", "6002"]'
    # A resolved row never moves again.
    assert mark_answered(store, 5000, [6003]) is False
    assert set_turn_state(store, 5000, "dropped") is False
    assert claim_turn(store, 5000) is False


def test_claim_and_terminal_states(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    assert claim_turn(store, 5000) is True
    assert get_turn(store, 5000)["state"] == "recovering"
    # A claimed (recovering) row can still be marked answered — that
    # is how a row handed to a live queue resolves.
    assert mark_answered(store, 5000, [6001]) is True
    register(store, 5001)
    assert set_turn_state(store, 5001, "superseded") is True
    assert get_turn(store, 5001)["state"] == "superseded"


def test_list_recoverable_order_and_window(tmp_path):
    store = make_store(tmp_path)
    register(store, 5002)
    register(store, 5001)
    register(store, 5003)
    mark_answered(store, 5003, [6000])
    now = datetime.now(UTC)
    rows = list_recoverable(
        store, since_iso=(now - timedelta(minutes=5)).isoformat()
    )
    assert [row["message_id"] for row in rows] == ["5001", "5002"]
    rows = list_recoverable(
        store, since_iso=(now + timedelta(minutes=5)).isoformat()
    )
    assert rows == []


def test_expire_turns_before(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    register(store, 5001)
    mark_answered(store, 5001, [6000])
    future = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    assert expire_turns_before(store, future) == 1
    assert get_turn(store, 5000)["state"] == "expired"
    assert get_turn(store, 5001)["state"] == "answered"


def test_claimed_answer_ids_and_next_trigger(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    register(store, 7000)
    mark_answered(store, 5000, [6001, 6002])
    assert claimed_answer_ids(store, CHANNEL_ID) == {"6001", "6002"}
    assert claimed_answer_ids(store, 999) == set()
    assert next_trigger_id(store, CHANNEL_ID, 5000) == "7000"
    assert next_trigger_id(store, CHANNEL_ID, 7000) is None


def test_has_answered_duplicate(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000, content="same  question")
    register(store, 7000, content="same question")
    register(store, 8000, content="other")
    mark_answered(store, 7000, [9000])
    mark_answered(store, 8000, [9001])
    # A later answered re-ask (whitespace-normalized) covers 5000.
    assert has_answered_duplicate(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=5000,
    )
    # The answered row itself has no later duplicate.
    assert not has_answered_duplicate(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=7000,
    )
    # Different author or content: not a duplicate.
    assert not has_answered_duplicate(
        store, channel_id=CHANNEL_ID, author_id=777,
        content="same question", message_id=5000,
    )
    assert not has_answered_duplicate(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="never answered", message_id=5000,
    )


def archive_record(message_id, *, author_id=AUTHOR_ID, content="<@99> question",
                   channel_id=CHANNEL_ID, reply_to=None, bot=False):
    return {
        "id": str(message_id),
        "guild_id": "123",
        "channel_id": str(channel_id),
        "_channel": "test",
        "_parent": None,
        "author": {
            "id": str(author_id),
            "username": "someone",
            "global_name": None,
            "nick": None,
            "display_name": "Someone",
            "bot": bot,
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "edited_timestamp": None,
        "message_reference": (
            {"message_id": str(reply_to)} if reply_to else None
        ),
        "content": content,
        "embeds": [],
        "attachments": [],
        "sticker_items": [],
    }


def test_sweep_candidates_finds_mentions_and_replies(tmp_path):
    store = make_store(tmp_path)
    store.upsert(archive_record(5000))  # mention
    store.upsert(archive_record(5001, content="just chatting"))
    store.upsert(archive_record(5002, author_id=BOT_ID, content="answer",
                                 bot=True))
    store.upsert(archive_record(5003, content="replying to the bot",
                                reply_to=5002))
    store.upsert(archive_record(5004, content="<@123456> not the bot"))
    since = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    found = sweep_candidates(store, bot_id=BOT_ID, since_iso=since)
    assert [c["message_id"] for c in found] == ["5000", "5003"]


# ------------------------------------------------------------- live path


class FakeChannel:
    """A channel stand-in with send/history/fetch for recovery tests."""

    def __init__(self, channel_id=CHANNEL_ID, *, history=(), messages=()):
        self.id = channel_id
        self.sent = []
        self._history = list(history)
        self._messages = {m.id: m for m in messages}
        self._next_id = 9000
        self.send = AsyncMock(side_effect=self._send)
        self.typing = Mock(side_effect=asyncio.Lock)

    async def _send(self, content=None, **kwargs):
        self._next_id += 1
        message = SimpleNamespace(
            id=self._next_id, content=content,
            author=SimpleNamespace(id=BOT_ID),
        )
        self.sent.append(message)
        return message

    async def fetch_message(self, message_id):
        if message_id in self._messages:
            return self._messages[message_id]
        raise Exception("unknown message")

    def history(self, *, limit=100, after=None, oldest_first=False):
        items = [
            m for m in self._history if after is None or m.id > after.id
        ]
        items.sort(key=lambda m: m.id, reverse=not oldest_first)
        items = items[:limit]

        async def generate():
            for item in items:
                yield item

        return generate()


def trigger_message(channel, text="question", *, author_id=AUTHOR_ID,
                    message_id=5000):
    user = SimpleNamespace(id=BOT_ID)
    return SimpleNamespace(
        author=SimpleNamespace(
            id=author_id, bot=False, name="someone",
            display_name="Someone", nick=None,
        ),
        content=f"<@{BOT_ID}> {text}",
        guild=SimpleNamespace(id=123),
        channel=channel,
        mentions=[user],
        id=message_id,
    )


def bot_message(message_id, content="an answer"):
    return SimpleNamespace(
        id=message_id, content=content,
        author=SimpleNamespace(id=BOT_ID),
    )


def recovery_bot(store, channel, *, started_minutes_ago=-1):
    """Fake bot wired for _run_unanswered_recovery.

    started_minutes_ago=-1 puts the process start in the future, so
    rows registered by the test count as a previous process's; a
    positive value makes them this process's own.
    """
    bot = SimpleNamespace(
        settings=settings(),
        user=SimpleNamespace(id=BOT_ID),
        busy=set(),
        turn_queues={},
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value="reply")),
        get_guild=Mock(return_value=None),
        get_channel=Mock(return_value=channel),
        fetch_channel=AsyncMock(return_value=channel),
        improvements=SimpleNamespace(settings=settings()),
        improvement_tasks=set(),
        server_log=store,
        backfill_task=None,
        _process_started_at=datetime.now(UTC)
        - timedelta(minutes=started_minutes_ago),
    )
    return bot


def run_recovery(bot):
    asyncio.run(_run_unanswered_recovery(bot))


def test_live_turn_registers_and_marks_answered(tmp_path):
    store = make_store(tmp_path)
    channel = FakeChannel()
    bot = recovery_bot(store, channel)
    bot.archive = AsyncMock()
    asyncio.run(Bot.on_message(bot, trigger_message(channel)))
    row = get_turn(store, 5000)
    assert row is not None
    assert row["state"] == "answered"
    assert row["origin"] == "live"
    assert row["answer_message_ids"] == '["9001"]'
    assert len(channel.sent) == 1


# -------------------------------------------------------------- recovery


def test_recovery_answers_pending_turn_exactly_once(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(channel, message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert len(channel.sent) == 1
    row = get_turn(store, 5000)
    assert row["state"] == "answered"
    # A second startup with nothing left unanswered posts nothing.
    channel.sent.clear()
    run_recovery(recovery_bot(store, channel))
    assert channel.sent == []


def test_recovery_adopts_answer_posted_before_crash(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    channel = FakeChannel(history=[bot_message(5001)])
    channel._messages[5000] = trigger_message(channel, message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []  # no duplicate answer
    row = get_turn(store, 5000)
    assert row["state"] == "answered"
    assert row["answer_message_ids"] == '["5001"]'


def test_recovery_respects_claimed_answers_and_boundary(tmp_path):
    # Two pending triggers; the only bot message in history sits
    # after the second trigger, so it can only be the second one's
    # answer: the first is generated, the second adopts it.
    store = make_store(tmp_path)
    register(store, 5000, content="first")
    register(store, 6000, content="second")
    channel = FakeChannel(history=[bot_message(6500)])
    channel._messages[5000] = trigger_message(
        channel, "first", message_id=5000)
    channel._messages[6000] = trigger_message(
        channel, "second", message_id=6000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert len(channel.sent) == 1  # only the first was generated
    assert get_turn(store, 5000)["state"] == "answered"
    adopted = get_turn(store, 6000)
    assert adopted["state"] == "answered"
    assert adopted["answer_message_ids"] == '["6500"]'


def test_recovery_supersedes_duplicate_within_run(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000, content="same question")
    register(store, 5001, content="same question")
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(
        channel, "same question", message_id=5000)
    channel._messages[5001] = trigger_message(
        channel, "same question", message_id=5001)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert len(channel.sent) == 1
    assert get_turn(store, 5000)["state"] == "answered"
    assert get_turn(store, 5001)["state"] == "superseded"


def test_recovery_supersedes_when_later_reask_was_answered(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000, content="same question")
    register(store, 7000, content="same question")
    mark_answered(store, 7000, [9000])
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(
        channel, "same question", message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []
    assert get_turn(store, 5000)["state"] == "superseded"


def test_recovery_expires_old_turns_without_answering(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    with store.connect() as db:
        db.execute(
            "UPDATE pending_turns SET accepted_at = ? WHERE message_id = ?",
            ((datetime.now(UTC) - timedelta(hours=3)).isoformat(), "5000"),
        )
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(channel, message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []
    assert get_turn(store, 5000)["state"] == "expired"


def test_recovery_skips_this_process_rows(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(channel, message_id=5000)
    bot = recovery_bot(store, channel, started_minutes_ago=60)
    run_recovery(bot)
    assert channel.sent == []
    assert get_turn(store, 5000)["state"] == "pending"


def test_recovery_drops_turn_of_newly_blacklisted_author(tmp_path):
    store = make_store(tmp_path)
    init_blacklist(store)
    register(store, 5000, author_id=777)
    add_to_blacklist(store, 777)
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(
        channel, author_id=777, message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []
    bot.assistant.reply.assert_not_awaited()
    assert get_turn(store, 5000)["state"] == "dropped"


def test_recovery_drops_turn_whose_message_is_gone(tmp_path):
    store = make_store(tmp_path)
    register(store, 5000)
    channel = FakeChannel()  # fetch_message will raise
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []
    assert get_turn(store, 5000)["state"] == "dropped"


def test_sweep_answers_message_posted_while_bot_was_down(tmp_path):
    store = make_store(tmp_path)
    store.upsert(archive_record(5000))
    channel = FakeChannel()
    channel._messages[5000] = trigger_message(channel, message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert len(channel.sent) == 1
    row = get_turn(store, 5000)
    assert row is not None
    assert row["origin"] == "sweep"
    assert row["state"] == "answered"


def test_sweep_adopts_already_answered_message(tmp_path):
    store = make_store(tmp_path)
    store.upsert(archive_record(5000))
    channel = FakeChannel(history=[bot_message(5001)])
    channel._messages[5000] = trigger_message(channel, message_id=5000)
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []
    assert get_turn(store, 5000) is None  # nothing needed recording


def test_recovery_with_nothing_pending_posts_nothing(tmp_path):
    store = make_store(tmp_path)
    channel = FakeChannel()
    bot = recovery_bot(store, channel)
    run_recovery(bot)
    assert channel.sent == []
    bot.assistant.reply.assert_not_awaited()
