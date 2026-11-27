"""v1.42.1: the unanswered-turn recovery must not answer a re-ask
whose first asking was already answered.

The 2026-11-27 production incident: two identical calls 16 seconds
apart (the first with an image attached) were posted while the bot
was down. The deploy-startup sweep answered the first; the re-ask's
supersede verdict lived only in that pass's memory, no durable check
looked for an already-answered EARLIER duplicate, and the history
cross-check correctly refuses to adopt an answer another turn
claims — so when a later process picked the re-ask up from a pending
row, it generated a second answer. These tests pin the fix: the
predecessor check, the durable recording of sweep supersede
verdicts, and the regression boundaries (different text, different
author, and a same-text question far outside the proximity window
are all still answered). All fixtures are synthetic.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from nezubot_discrod.bot import Bot, _run_unanswered_recovery
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.message_images import MAX_IMAGE_BYTES
from nezubot_discrod.pending_turns import (
    get_turn,
    has_answered_predecessor,
    init_pending_turns,
    mark_answered,
    register_turn,
)
from nezubot_discrod.server_log import ServerLog

BOT_ID = 99
CHANNEL_ID = 1
AUTHOR_ID = 456
DISCORD_EPOCH_MS = 1420070400000


def snowflake(when, *, seq=0):
    """A Discord-shaped snowflake id for a datetime."""
    ms = int(when.timestamp() * 1000)
    return ((ms - DISCORD_EPOCH_MS) << 22) + seq


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
             channel_id=CHANNEL_ID, origin="live"):
    return register_turn(
        store,
        message_id=message_id,
        channel_id=channel_id,
        guild_id=123,
        author_id=author_id,
        content=content,
        origin=origin,
    )


# ---------------------------------------------------------------- store


def test_has_answered_predecessor(tmp_path):
    now = datetime.now(UTC)
    first = snowflake(now - timedelta(seconds=16))
    reask = snowflake(now)
    store = make_store(tmp_path)
    register(store, first, content="same  question")
    register(store, reask, content="same question")
    # Not yet answered: no predecessor verdict.
    assert not has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=reask,
    )
    mark_answered(store, first, [snowflake(now)])
    # The re-ask now has an answered predecessor (whitespace-normalized).
    assert has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=reask,
    )
    # The answered row itself has no earlier duplicate.
    assert not has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=first,
    )
    # Different author, channel or content: not a predecessor.
    assert not has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=777,
        content="same question", message_id=reask,
    )
    assert not has_answered_predecessor(
        store, channel_id=999, author_id=AUTHOR_ID,
        content="same question", message_id=reask,
    )
    assert not has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="another question", message_id=reask,
    )


def test_has_answered_predecessor_gap_bound(tmp_path):
    now = datetime.now(UTC)
    first = snowflake(now - timedelta(hours=3))
    near = snowflake(now - timedelta(hours=3) + timedelta(seconds=16))
    far = snowflake(now)
    store = make_store(tmp_path)
    register(store, first, content="same question")
    mark_answered(store, first, [snowflake(now - timedelta(hours=3))])
    window_ms = 120 * 60 * 1000
    # 16 seconds apart: one question.
    assert has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=near, max_gap_ms=window_ms,
    )
    # 3 hours apart: outside the proximity window, a new question.
    assert not has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=far, max_gap_ms=window_ms,
    )
    # Unbounded, the far candidate does see the predecessor.
    assert has_answered_predecessor(
        store, channel_id=CHANNEL_ID, author_id=AUTHOR_ID,
        content="same question", message_id=far,
    )


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
                    message_id=5000, attachments=()):
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
        attachments=list(attachments),
        reference=None,
    )


def bot_message(message_id, content="an answer"):
    return SimpleNamespace(
        id=message_id, content=content,
        author=SimpleNamespace(id=BOT_ID),
    )


def archive_record(message_id, *, author_id=AUTHOR_ID,
                   content="<@99> question", channel_id=CHANNEL_ID):
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
            "bot": False,
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "edited_timestamp": None,
        "message_reference": None,
        "content": content,
        "embeds": [],
        "attachments": [],
        "sticker_items": [],
    }


def recovery_bot(store, channel):
    """Fake bot wired for _run_unanswered_recovery; the process start
    is in the future, so rows the test registered count as a previous
    process's."""
    return SimpleNamespace(
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
        _process_started_at=datetime.now(UTC) + timedelta(minutes=1),
    )


def run_recovery(bot):
    asyncio.run(_run_unanswered_recovery(bot))


# -------------------------------------------------------------- recovery


def test_recovery_does_not_reanswer_reask_with_pending_row(tmp_path):
    # The production incident's decisive shape, phase A route: the
    # first asking is answered (its answer sits in history, claimed
    # by its row), and the re-ask holds a pending row in a later
    # process. Before the fix this generated a second answer.
    now = datetime.now(UTC)
    first = snowflake(now - timedelta(seconds=16))
    reask = snowflake(now)
    answer = snowflake(now + timedelta(seconds=30))
    store = make_store(tmp_path)
    register(store, first, content="これなに")
    mark_answered(store, first, [answer])
    register(store, reask, content="これなに")
    channel = FakeChannel(history=[bot_message(answer)])
    channel._messages[first] = trigger_message(
        channel, "これなに", message_id=first)
    channel._messages[reask] = trigger_message(
        channel, "これなに", message_id=reask)
    run_recovery(recovery_bot(store, channel))
    assert channel.sent == []
    assert get_turn(store, reask)["state"] == "superseded"
    assert get_turn(store, first)["state"] == "answered"


def test_recovery_does_not_reanswer_rowless_reask_in_fresh_process(
    tmp_path,
):
    # Phase B route across processes: the first asking was answered
    # by an earlier pass; the re-ask exists only in the archive. The
    # fresh pass must not answer it, and must record the verdict so
    # no later pass reconsiders it.
    now = datetime.now(UTC)
    first = snowflake(now - timedelta(seconds=16))
    reask = snowflake(now)
    answer = snowflake(now + timedelta(seconds=30))
    store = make_store(tmp_path)
    register(store, first, content="これなに", origin="sweep")
    mark_answered(store, first, [answer])
    store.upsert(archive_record(
        reask, content="<@99> これなに"))
    channel = FakeChannel(history=[bot_message(answer)])
    channel._messages[first] = trigger_message(
        channel, "これなに", message_id=first)
    channel._messages[reask] = trigger_message(
        channel, "これなに", message_id=reask)
    run_recovery(recovery_bot(store, channel))
    assert channel.sent == []
    row = get_turn(store, reask)
    assert row is not None
    assert row["state"] == "superseded"
    # A further startup still posts nothing.
    run_recovery(recovery_bot(store, channel))
    assert channel.sent == []


def test_sweep_pair_answered_exactly_once_first_with_attachment(tmp_path):
    # The incident's exact shape in a single pass: two identical
    # calls 16 seconds apart, only the first carrying an image. The
    # attachment must not split the pair's identity — exactly one
    # answer is generated, for the first asking.
    now = datetime.now(UTC)
    first = snowflake(now - timedelta(seconds=16))
    reask = snowflake(now)
    store = make_store(tmp_path)
    store.upsert(archive_record(
        first, content="<@99> これなに\n[file] IMG_0001.png"))
    store.upsert(archive_record(reask, content="<@99> これなに"))
    attachment = SimpleNamespace(
        filename="IMG_0001.png", content_type="image/png",
        size=MAX_IMAGE_BYTES + 1, url="https://example.invalid/img.png",
    )
    channel = FakeChannel()
    channel._messages[first] = trigger_message(
        channel, "これなに", message_id=first, attachments=[attachment])
    channel._messages[reask] = trigger_message(
        channel, "これなに", message_id=reask)
    run_recovery(recovery_bot(store, channel))
    assert len(channel.sent) == 1
    first_row = get_turn(store, first)
    assert first_row["state"] == "answered"
    assert first_row["origin"] == "sweep"
    assert get_turn(store, reask)["state"] == "superseded"


def test_recovery_answers_two_different_questions_from_same_author(
    tmp_path,
):
    # Regression boundary: same author and channel, different text —
    # both are owed an answer.
    now = datetime.now(UTC)
    one = snowflake(now - timedelta(seconds=16))
    two = snowflake(now)
    store = make_store(tmp_path)
    register(store, one, content="first question")
    register(store, two, content="second question")
    channel = FakeChannel()
    channel._messages[one] = trigger_message(
        channel, "first question", message_id=one)
    channel._messages[two] = trigger_message(
        channel, "second question", message_id=two)
    run_recovery(recovery_bot(store, channel))
    assert len(channel.sent) == 2
    assert get_turn(store, one)["state"] == "answered"
    assert get_turn(store, two)["state"] == "answered"


def test_recovery_answers_same_text_from_two_authors(tmp_path):
    # Regression boundary: identical text, different authors — both
    # are owed an answer.
    now = datetime.now(UTC)
    one = snowflake(now - timedelta(seconds=16))
    two = snowflake(now)
    store = make_store(tmp_path)
    register(store, one, author_id=AUTHOR_ID, content="same question")
    register(store, two, author_id=777, content="same question")
    channel = FakeChannel()
    channel._messages[one] = trigger_message(
        channel, "same question", message_id=one)
    channel._messages[two] = trigger_message(
        channel, "same question", author_id=777, message_id=two)
    run_recovery(recovery_bot(store, channel))
    assert len(channel.sent) == 2
    assert get_turn(store, one)["state"] == "answered"
    assert get_turn(store, two)["state"] == "answered"


def test_recovery_answers_repeat_question_outside_proximity_window(
    tmp_path,
):
    # Same text asked again 3 hours later is a new question, not a
    # re-ask: the predecessor inside the window must not suppress it.
    now = datetime.now(UTC)
    first = snowflake(now - timedelta(hours=3))
    repeat = snowflake(now)
    store = make_store(tmp_path)
    register(store, first, content="same question")
    mark_answered(store, first, [snowflake(now - timedelta(hours=3))])
    register(store, repeat, content="same question")
    channel = FakeChannel()
    channel._messages[repeat] = trigger_message(
        channel, "same question", message_id=repeat)
    run_recovery(recovery_bot(store, channel))
    assert len(channel.sent) == 1
    assert get_turn(store, repeat)["state"] == "answered"


def test_live_reask_is_still_answered_live(tmp_path):
    # The live path is untouched by the fix: an online re-ask is a
    # new turn and gets its own answer, exactly as before.
    store = make_store(tmp_path)
    channel = FakeChannel()
    bot = recovery_bot(store, channel)
    bot.archive = AsyncMock()
    now = datetime.now(UTC)
    one = snowflake(now - timedelta(seconds=16))
    two = snowflake(now)
    asyncio.run(Bot.on_message(
        bot, trigger_message(channel, "これなに", message_id=one)))
    asyncio.run(Bot.on_message(
        bot, trigger_message(channel, "これなに", message_id=two)))
    assert len(channel.sent) == 2
    assert get_turn(store, one)["state"] == "answered"
    assert get_turn(store, two)["state"] == "answered"
