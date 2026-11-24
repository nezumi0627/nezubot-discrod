import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from nezubot_discrod.bot import Bot
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.conversation import TRANSCRIPT_BUDGET, build_conversation_context

JST = timezone(timedelta(hours=9))
BASE = datetime(2026, 11, 24, 17, 29, tzinfo=JST)


def user(user_id, name, *, bot=False):
    return SimpleNamespace(id=user_id, name=name, display_name=name, bot=bot)


BOT_USER = user(99, "nezubot-discrod", bot=True)
AZAZEL = user(301, "AzazeliusZ")
KOKURA = user(302, "kokura_server")
CAROL = user(303, "carol")
OWNER_ID = 456


def reference(message_id, resolved=None):
    return SimpleNamespace(message_id=message_id, resolved=resolved)


def message(mid, author, content, *, minutes=0, mentions=(), reply_to=None, channel=None):
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
    """History yields newest first, like discord.py; fetch by id."""

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


def build(anchor):
    return asyncio.run(build_conversation_context(anchor, BOT_USER.id))


def test_reply_chain_contents_authors_and_mentions():
    intro = message(
        1, AZAZEL, "皆さん、こんにちは！以前から開発していたAIコンパニオンの話です",
        mentions=[CAROL],
    )
    anchor = message(
        2, KOKURA, "<@99> この人物の過去のメッセージさがして", minutes=1,
        mentions=[BOT_USER], reply_to=reference(1, resolved=intro),
    )
    channel = FakeChannel([intro, anchor])
    intro.channel = anchor.channel = channel
    context = build(anchor)
    assert context["reply_chain"] == [
        {"author": "AzazeliusZ", "author_id": "301", "content": intro.content}
    ]
    # Carol is named in the replied-to message; the bot itself is excluded.
    assert {entry["id"] for entry in context["mentions"]} == {"303"}
    assert context["grounded_person"] == {
        "id": "301", "display_name": "AzazeliusZ", "basis": "reply_target",
    }


def test_reply_chain_fetches_unresolved_targets_and_walks_hops():
    first = message(1, CAROL, "最初の発言")
    second = message(
        2, AZAZEL, "それへの返信", minutes=1, reply_to=reference(1, resolved=first),
    )
    anchor = message(
        3, KOKURA, "<@99> この人物は？", minutes=2,
        mentions=[BOT_USER], reply_to=reference(2),  # resolved is None: fetch it
    )
    channel = FakeChannel([first, second, anchor])
    for item in (first, second, anchor):
        item.channel = channel
    context = build(anchor)
    assert [entry["author"] for entry in context["reply_chain"]] == ["AzazeliusZ", "carol"]
    assert context["grounded_person"]["id"] == "301"
    assert context["grounded_person"]["basis"] == "reply_target"


def test_screenshot_case_grounds_to_previous_speaker_not_owner():
    """雑談で AzazeliusZ が自己紹介した直後、kokura_server がメンションで
    「この人物の過去のメッセージさがして」と頼んだケースの再現。
    接地先は直前の発言者 AzazeliusZ で、オーナーでも発言者本人でもない。"""
    intro = message(
        1, AZAZEL, "皆さん、こんにちは！以前から開発していたPC向けAIコンパニオンを公開しました",
    )
    anchor = message(
        2, KOKURA, "<@99> この人物の過去のメッセージさがして", minutes=1,
        mentions=[BOT_USER],
    )
    channel = FakeChannel([intro, anchor])
    intro.channel = anchor.channel = channel
    context = build(anchor)
    grounded = context["grounded_person"]
    assert grounded["id"] == "301"
    assert grounded["basis"] == "recent_speaker"
    assert grounded["id"] not in (str(OWNER_ID), str(KOKURA.id))


def test_transcript_is_chronological_with_authors_replies_and_anchor_mark():
    question = message(1, CAROL, "リンクを共有していいですか？")
    answer = message(
        2, AZAZEL, "おけ", minutes=1, reply_to=reference(1, resolved=question),
    )
    anchor = message(3, KOKURA, "<@99> 教えて", minutes=2, mentions=[BOT_USER])
    channel = FakeChannel([question, answer, anchor])
    for item in (question, answer, anchor):
        item.channel = channel
    context = build(anchor)
    lines = context["transcript"].splitlines()
    assert len(lines) == 3
    assert "carol" in lines[0] and "リンクを共有していいですか？" in lines[0]
    assert "AzazeliusZ" in lines[1] and "（↩ carol への返信）" in lines[1]
    assert "kokura_server" in lines[2] and lines[2].endswith("← いま応答する発言")
    assert "grounded_person" not in context


def test_explicit_mention_is_left_to_text_resolution():
    intro = message(1, AZAZEL, "自己紹介です")
    anchor = message(
        2, KOKURA, "<@99> <@303> この人物について教えて", minutes=1,
        mentions=[BOT_USER, CAROL],
    )
    channel = FakeChannel([intro, anchor])
    intro.channel = anchor.channel = channel
    context = build(anchor)
    # An explicit @-mention is resolved from the text by resolve_people;
    # the conversation layer must not override it with the recent speaker.
    assert "grounded_person" not in context
    assert {entry["id"] for entry in context["mentions"]} == {"303"}


def test_transcript_stays_within_budget_and_keeps_the_anchor():
    history = [
        message(index + 1, CAROL if index % 2 else AZAZEL, "長い発言 " + "あ" * 300,
                minutes=index)
        for index in range(25)
    ]
    anchor = message(100, KOKURA, "<@99> この人物は？", minutes=30, mentions=[BOT_USER])
    channel = FakeChannel([*history, anchor])
    for item in (*history, anchor):
        item.channel = channel
    context = build(anchor)
    transcript = context["transcript"]
    assert len(transcript) <= TRANSCRIPT_BUDGET
    assert transcript.splitlines()[-1].endswith("← いま応答する発言")
    # The most recent history message (index 24) is AzazeliusZ's.
    assert context["grounded_person"]["id"] == str(AZAZEL.id)


def test_nothing_beyond_the_message_yields_no_context():
    anchor = message(1, KOKURA, "<@99> remember this", mentions=[BOT_USER])
    anchor.channel = SimpleNamespace(id=1)  # no history, no fetch
    assert build(anchor) is None


def _settings():
    return Settings(
        "token", "key", "https://example.com", "test", SandboxSettings(),
        guild_id=123, owner_id=OWNER_ID,
    )


def test_on_message_routes_conversation_context_to_the_assistant():
    intro = message(1, AZAZEL, "皆さん、こんにちは！AIコンパニオンを公開しました")
    channel = FakeChannel([intro])
    channel.send = AsyncMock(return_value=SimpleNamespace(id=500))
    channel.typing = Mock(side_effect=asyncio.Lock)
    bot = SimpleNamespace(
        settings=_settings(),
        user=BOT_USER,
        busy=set(),
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value="reply")),
        get_guild=Mock(return_value=None),
        improvements=SimpleNamespace(settings=_settings()),
        improvement_tasks=set(),
    )
    anchor = message(
        2, KOKURA, "<@99> この人物の過去のメッセージさがして", minutes=1,
        mentions=[BOT_USER], channel=channel,
    )
    anchor.guild = SimpleNamespace(id=123)
    intro.channel = channel
    asyncio.run(Bot.on_message(bot, anchor))
    conversation = bot.assistant.reply.call_args.kwargs["conversation"]
    assert conversation["grounded_person"]["id"] == "301"
    assert conversation["grounded_person"]["basis"] == "recent_speaker"
    assert "AzazeliusZ" in conversation["transcript"]
