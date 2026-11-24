import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from test_assistant import client_for
from test_assistant import message as completion_message
from test_bot import settings

from nezubot_discrod.ambient import skip_answer
from nezubot_discrod.assistant import Assistant
from nezubot_discrod.bot import Bot
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.prompts import AMBIENT_GUIDANCE


def incoming():
    me = SimpleNamespace(id=99)
    permissions = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
        send_messages=True,
        send_messages_in_threads=True,
        manage_threads=False,
    )
    channel = SimpleNamespace(
        id=10,
        permissions_for=Mock(return_value=permissions),
        send=AsyncMock(),
        typing=Mock(side_effect=asyncio.Lock),
    )
    msg = SimpleNamespace(
        id=42,
        content="Rustを教えてほしい",
        channel=channel,
        guild=SimpleNamespace(id=123, me=me),
        author=SimpleNamespace(id=20, bot=False),
        mentions=[],
        reference=None,
    )
    return msg


@pytest.mark.parametrize(
    "text,skip",
    [
        ("(skip)", True),
        (" （SKIP）。 ", True),
        ("`(skip)`", True),
        ("", True),
        (None, True),
        ("   ", True),
        ("skip this", False),
        ("(skip) but actually", False),
        ("回答です", False),
    ],
)
def test_skip_suppression(text, skip):
    assert skip_answer(text) is skip


def bot_for(msg, answer="reply"):
    return SimpleNamespace(
        settings=settings(),
        user=msg.guild.me,
        server_log=Mock(),
        archive=AsyncMock(),
        busy=set(),
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value=answer)),
    )


@pytest.mark.parametrize("content", [
    "Rustを教えてほしい", "nezubot-discrod、これ何？", "Nezu どう思う？", "誰か分かる人いますか？",
])
def test_unaddressed_guild_message_is_never_answered(content):
    msg = incoming()
    msg.content = content
    msg.guild.me.display_name = "nezubot-discrod"
    bot = bot_for(msg)
    with patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[10])):
        asyncio.run(Bot.on_message(bot, msg))
    bot.archive.assert_awaited_once_with(msg)
    bot.assistant.reply.assert_not_awaited()
    msg.channel.send.assert_not_awaited()


def test_busy_unaddressed_message_stays_silent():
    msg = incoming()
    bot = bot_for(msg)
    bot.busy.add(10)
    asyncio.run(Bot.on_message(bot, msg))
    msg.channel.send.assert_not_awaited()


@pytest.mark.parametrize("trigger", ["mention", "reply"])
def test_mention_or_reply_to_bot_is_answered(trigger):
    msg = incoming()
    if trigger == "mention":
        msg.mentions = [msg.guild.me]
    else:
        msg.reference = SimpleNamespace(
            message_id=1, resolved=SimpleNamespace(author=SimpleNamespace(id=99)),
        )
    bot = bot_for(msg)
    with (
        patch("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[10, 11])),
        patch("nezubot_discrod.bot.audience_channel_ids", return_value=[10]) as audience,
    ):
        asyncio.run(Bot.on_message(bot, msg))
    audience.assert_called_once_with(bot, msg.channel, [10, 11])
    bot.assistant.reply.assert_awaited_once_with(
        10, msg.content, member_id=20, guild_id=123,
        guild_info={
            "name": None,
            "owner": {"id": None, "display_name": None},
            "created_at": None,
            "member_count": None,
        },
        channel_ids=[10, 11],
        retrieval_channel_ids=[10], message_id=42,
        speaker_names={"username": None, "display_name": None, "server_nickname": None},
    )
    msg.channel.send.assert_awaited_once_with("reply\n-# test")


def test_reply_to_another_member_is_not_answered():
    msg = incoming()
    msg.reference = SimpleNamespace(
        message_id=1, resolved=SimpleNamespace(author=SimpleNamespace(id=21)),
    )
    bot = bot_for(msg)
    asyncio.run(Bot.on_message(bot, msg))
    bot.assistant.reply.assert_not_awaited()
    msg.channel.send.assert_not_awaited()


def test_ambient_prompt_no_wiki_and_empty_does_not_persist_history():
    client, sent = client_for(completion_message(""))
    wiki = Mock()
    ai = Assistant(client, "test", SandboxSettings(), wiki=wiki)
    assert asyncio.run(ai.reply(1, "question", member_id=20, ambient=True)) == "(skip)"
    assert AMBIENT_GUIDANCE in sent[0]["messages"][0]["content"]
    wiki.for_member.assert_not_called()
    assert not ai.history
