import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nezubot_discrod.bot import Bot, allowed_source, deliver_answer
from nezubot_discrod.config import SandboxSettings, Settings


def settings():
    return Settings(
        "token", "key", "https://example.com", "test", SandboxSettings(), guild_id=123, owner_id=456
    )


@pytest.mark.parametrize(
    "guild,user,allowed",
    [
        (123, 1, True),
        (124, 456, False),
        (None, 456, True),
        (None, 1, False),
    ],
)
def test_source_gate(guild, user, allowed):
    assert allowed_source(settings(), guild, user) is allowed


def test_messages_are_ignored_during_restart():
    with patch("nezubot_discrod.bot.accepting_work", return_value=False):
        asyncio.run(Bot.on_message(SimpleNamespace(), SimpleNamespace()))


@pytest.mark.parametrize("guild,user", [(124, 456), (None, 1)])
def test_messages_from_disallowed_sources_are_ignored(guild, user):
    bot = SimpleNamespace(settings=settings(), user=SimpleNamespace(id=99), assistant=Mock())
    message = SimpleNamespace(
        author=SimpleNamespace(id=user, bot=False),
        guild=SimpleNamespace(id=guild) if guild else None,
    )
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.assert_not_called()


@pytest.mark.parametrize("guild", [123, None])
def test_message_routes_discord_identity_to_the_assistant(guild):
    user = SimpleNamespace(id=99)
    channel = SimpleNamespace(id=1, send=AsyncMock(), typing=Mock(side_effect=asyncio.Lock))
    bot = SimpleNamespace(
        settings=settings(),
        user=user,
        busy=set(),
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value="reply")),
        get_guild=Mock(return_value=None),
        improvements=SimpleNamespace(settings=settings()),
        improvement_tasks=set(),
    )
    message = SimpleNamespace(
        author=SimpleNamespace(
            id=456, bot=False, name="testuser", display_name="テストユーザー",
            nick="店長" if guild else None,
        ),
        content="<@99> remember this",
        guild=SimpleNamespace(id=guild) if guild else None,
        channel=channel,
        mentions=[user],
        id=1000,
    )
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_awaited_once_with(
        1,
        "remember this",
        member_id=456,
        speaker_names={
            "username": "testuser", "display_name": "テストユーザー",
            "server_nickname": "店長" if guild else None,
        },
        guild_id=guild,
        guild_info=(
            {
                "name": None,
                "owner": {"id": None, "display_name": None},
                "created_at": None,
                "member_count": None,
            }
            if guild
            else None
        ),
        channel_ids=[],
        message_id=1000,
        self_improve=bot.assistant.reply.call_args.kwargs["self_improve"],
    )
    channel.send.assert_awaited_once_with("reply\n-# test")


@pytest.mark.parametrize("busy", [{1}, set(range(16))])
def test_busy_gate_drops_message_silently(busy):
    # The busy gate must never post a "処理中です" style notice to the
    # channel: the trigger is dropped without any visible message.
    user = SimpleNamespace(id=99)
    channel = SimpleNamespace(id=1, send=AsyncMock(), typing=Mock(side_effect=asyncio.Lock))
    bot = SimpleNamespace(
        settings=settings(),
        user=user,
        busy=busy,
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value="reply")),
    )
    message = SimpleNamespace(
        author=SimpleNamespace(
            id=456, bot=False, name="testuser", display_name="テストユーザー", nick=None,
        ),
        content="<@99> hello",
        guild=SimpleNamespace(id=123),
        channel=channel,
        mentions=[user],
        id=1000,
    )
    asyncio.run(Bot.on_message(bot, message))
    channel.send.assert_not_awaited()
    bot.assistant.reply.assert_not_awaited()


def _reply_setup(*, resolved, fetch_result=None):
    """Bot fake plus a channel whose fetch_message resolves replies."""
    user = SimpleNamespace(id=99)
    channel = SimpleNamespace(
        id=1,
        send=AsyncMock(),
        typing=Mock(side_effect=asyncio.Lock),
        fetch_message=AsyncMock(return_value=fetch_result),
    )
    bot = SimpleNamespace(
        settings=settings(),
        user=user,
        busy=set(),
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value="reply")),
        get_guild=Mock(return_value=None),
        improvements=SimpleNamespace(settings=settings()),
        improvement_tasks=set(),
    )
    message = SimpleNamespace(
        author=SimpleNamespace(
            id=456, bot=False, name="testuser", display_name="テストユーザー", nick=None,
        ),
        content="つづきを教えて",
        guild=SimpleNamespace(id=123),
        channel=channel,
        mentions=[],
        reference=SimpleNamespace(message_id=555, resolved=resolved),
        id=1000,
    )
    return bot, channel, message


def test_reply_to_bot_message_triggers_without_mention():
    # A reply to the bot's own message is a call even with no @mention;
    # the same path serves replies inside the bot's teaser threads,
    # whose channels take the identical on_message route.
    bot_user = SimpleNamespace(id=99)
    target = SimpleNamespace(id=555, author=bot_user)
    bot, channel, message = _reply_setup(resolved=target)
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_awaited_once()
    args, kwargs = bot.assistant.reply.call_args
    assert args[:2] == (1, "つづきを教えて")
    assert kwargs["message_id"] == 1000
    channel.send.assert_awaited_once()


def test_reply_to_bot_message_triggers_when_reference_has_no_author():
    # The gateway may attach an authorless DeletedReferencedMessage
    # for a target that still exists; on_message must still trigger
    # via the one-fetch fallback in is_call.
    target = SimpleNamespace(id=555, author=SimpleNamespace(id=99))
    bot, channel, message = _reply_setup(
        resolved=SimpleNamespace(id=555), fetch_result=target,
    )
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_awaited_once()
    channel.fetch_message.assert_awaited_once_with(555)


def test_reply_to_another_member_does_not_trigger():
    target = SimpleNamespace(id=555, author=SimpleNamespace(id=777))
    bot, channel, message = _reply_setup(resolved=target)
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_not_awaited()
    channel.send.assert_not_awaited()


def test_reply_to_bot_with_unfetchable_target_does_not_trigger():
    import discord

    bot, channel, message = _reply_setup(resolved=SimpleNamespace(id=555))
    channel.fetch_message.side_effect = discord.NotFound(Mock(status=404), "gone")
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_not_awaited()
    channel.send.assert_not_awaited()


@pytest.mark.parametrize("author_id", [99, 555])
def test_bot_authored_reply_never_triggers(author_id):
    # Replies written by the bot itself or by another bot account are
    # dropped by the author gate even when they reply to the bot.
    target = SimpleNamespace(id=555, author=SimpleNamespace(id=99))
    bot, channel, message = _reply_setup(resolved=target)
    message.author = SimpleNamespace(
        id=author_id, bot=True, name="bot", display_name="bot", nick=None,
    )
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_not_awaited()
    channel.send.assert_not_awaited()


def _text_channel():
    import discord

    channel = Mock(spec=discord.TextChannel)
    channel.id = 7
    thread = SimpleNamespace(send=AsyncMock())
    teaser = SimpleNamespace(
        create_thread=AsyncMock(return_value=thread), delete=AsyncMock(),
    )
    channel.send = AsyncMock(return_value=teaser)
    return channel, teaser, thread


def test_long_answer_gets_teaser_and_thread():
    channel, teaser, thread = _text_channel()
    answer = "あ" * 300
    asyncio.run(deliver_answer(channel, "質問 です", answer, ["全文1", "全文2"]))
    channel.send.assert_awaited_once()
    teaser_text = channel.send.call_args.args[0]
    assert teaser_text.endswith("続きはコチラから⬇️")
    assert len(teaser_text) < 150
    teaser.create_thread.assert_awaited_once()
    assert teaser.create_thread.call_args.kwargs["name"] == "質問 です"
    assert [call.args[0] for call in thread.send.await_args_list] == ["全文1", "全文2"]


def test_long_answer_falls_back_when_thread_creation_fails():
    channel, teaser, thread = _text_channel()
    teaser.create_thread = AsyncMock(side_effect=RuntimeError("no threads"))
    answer = "あ" * 300
    asyncio.run(deliver_answer(channel, "質問", answer, ["全文"]))
    teaser.delete.assert_awaited_once()
    assert channel.send.await_count == 2
    assert channel.send.await_args_list[-1].args[0] == "全文"
    thread.send.assert_not_awaited()


def test_short_answer_is_sent_directly():
    channel, teaser, thread = _text_channel()
    asyncio.run(deliver_answer(channel, "質問", "短い答え", ["短い答え"]))
    channel.send.assert_awaited_once_with("短い答え")
    teaser.create_thread.assert_not_awaited()


def test_guild_facts_come_from_the_discord_guild_object():
    from datetime import UTC, datetime

    from nezubot_discrod.bot import guild_facts

    guild = SimpleNamespace(
        name="Example Developers",
        owner_id=999,
        owner=SimpleNamespace(display_name="あめっくす"),
        created_at=datetime(2024, 6, 26, 12, 0, tzinfo=UTC),
        member_count=42,
    )
    assert guild_facts(guild) == {
        "name": "Example Developers",
        "owner": {"id": "999", "display_name": "あめっくす"},
        "created_at": "2024-06-26",
        "member_count": 42,
    }
    assert guild_facts(None) is None
    uncached_owner = SimpleNamespace(
        name="G", owner_id=7, owner=None, created_at=None, member_count=None,
    )
    assert guild_facts(uncached_owner)["owner"] == {"id": "7", "display_name": None}


def test_slash_commands_sync_only_to_selected_guild(caplog):
    bot = SimpleNamespace(settings=settings(), tree=Mock(sync=AsyncMock(return_value=[])))
    with (caplog.at_level("INFO", logger="nezubot_discrod.bot"),
          patch("nezubot_discrod.bot.load_features") as load):
        asyncio.run(Bot.setup_hook(bot))
    load.assert_called_once_with(bot.tree, 123)
    guild = bot.tree.sync.call_args.kwargs["guild"]
    assert guild.id == 123
    bot.tree.copy_global_to.assert_called_once_with(guild=guild)
    assert "Synced 0 slash commands to guild 123" in caplog.text


def test_discord_id_config(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "test")
    monkeypatch.setenv("AI_API_KEY", "test")
    monkeypatch.setenv("DISCORD_GUILD_ID", "123")
    monkeypatch.setenv("DISCORD_OWNER_ID", "456")
    assert Settings.from_env().guild_id == 123
    assert Settings.from_env().owner_id == 456
