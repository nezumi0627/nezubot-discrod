import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nezubot_discrod.blacklist import (
    INITIAL_BLACKLIST_IDS,
    add_to_blacklist,
    init_blacklist,
    is_blacklisted,
    list_blacklist,
    remove_from_blacklist,
)
from nezubot_discrod.bot import Bot, show_blacklist, slash_blacklist_check
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.feature_loader import BUILTIN_COMMANDS
from nezubot_discrod.server_log import ServerLog

GUILD_ID = 123
OWNER = 456
BLOCKED = 777
SEED_ID = "123456789012345678"


@pytest.fixture()
def store(tmp_path):
    return ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")


# --- store level -------------------------------------------------------


def test_init_seeds_initial_member_once(store):
    assert SEED_ID in INITIAL_BLACKLIST_IDS
    init_blacklist(store)
    init_blacklist(store)  # idempotent: still exactly one row
    rows = list_blacklist(store)
    assert [row[0] for row in rows] == [SEED_ID]
    assert is_blacklisted(store, int(SEED_ID))


def test_seed_removal_is_not_undone_by_reinit(store):
    init_blacklist(store)
    assert remove_from_blacklist(store, SEED_ID) is True
    init_blacklist(store)  # table exists now: the seed must not return
    assert not is_blacklisted(store, int(SEED_ID))
    assert list_blacklist(store) == []


def test_add_lookup_remove_roundtrip(store):
    assert not is_blacklisted(store, BLOCKED)
    assert add_to_blacklist(store, BLOCKED, added_by=OWNER) is True
    # The very next lookup sees it: add invalidates the cache.
    assert is_blacklisted(store, BLOCKED)
    assert add_to_blacklist(store, BLOCKED, added_by=OWNER) is False
    rows = list_blacklist(store)
    assert len(rows) == 1
    assert rows[0][0] == str(BLOCKED) and rows[0][1] == str(OWNER)
    assert remove_from_blacklist(store, BLOCKED) is True
    assert not is_blacklisted(store, BLOCKED)
    assert remove_from_blacklist(store, BLOCKED) is False


def test_blacklist_is_per_user(store):
    add_to_blacklist(store, BLOCKED)
    assert is_blacklisted(store, BLOCKED)
    assert not is_blacklisted(store, 888)
    assert not is_blacklisted(store, None)


# --- /blacklist handler --------------------------------------------------


def make_interaction(user_id=OWNER, guild_id=GUILD_ID, channel=None):
    return SimpleNamespace(
        guild_id=guild_id,
        user=SimpleNamespace(id=user_id, display_name="だれか"),
        channel=channel,
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def make_channel():
    return SimpleNamespace(id=555, send=AsyncMock())


def make_blacklist_bot(store):
    return SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID, owner_id=OWNER),
        server_log=store,
    )


def target(user_id=BLOCKED):
    return SimpleNamespace(id=user_id, display_name="対象者")


def sent_text(interaction):
    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    assert kwargs["ephemeral"] is True
    return args[0]


def test_show_blacklist_add_then_list_then_remove(store):
    bot = make_blacklist_bot(store)
    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "add", target()))
    assert "追加した" in sent_text(interaction)
    assert is_blacklisted(store, BLOCKED)

    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "list", None))
    text = sent_text(interaction)
    assert str(BLOCKED) in text and "1人" in text

    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "remove", target()))
    assert "外した" in sent_text(interaction)
    assert not is_blacklisted(store, BLOCKED)


def test_show_blacklist_add_announces_embed_in_channel(store):
    bot = make_blacklist_bot(store)
    channel = make_channel()
    interaction = make_interaction(channel=channel)
    asyncio.run(show_blacklist(interaction, bot, "add", target()))
    # The operator still gets the ephemeral confirmation…
    assert "追加した" in sent_text(interaction)
    # …and the add is announced in the command's channel.
    channel.send.assert_awaited_once()
    embed = channel.send.call_args.kwargs["embed"]
    assert "ブラックリスト" in embed.title
    fields = {field.name: field.value for field in embed.fields}
    assert str(BLOCKED) in fields["対象ユーザー"]
    assert "登録しました" in fields["適用内容"]


def test_show_blacklist_remove_announces_embed_in_channel(store):
    bot = make_blacklist_bot(store)
    add_to_blacklist(store, BLOCKED)
    channel = make_channel()
    interaction = make_interaction(channel=channel)
    asyncio.run(show_blacklist(interaction, bot, "remove", target()))
    assert "外した" in sent_text(interaction)
    channel.send.assert_awaited_once()
    embed = channel.send.call_args.kwargs["embed"]
    assert "解除" in embed.title
    fields = {field.name: field.value for field in embed.fields}
    assert str(BLOCKED) in fields["対象ユーザー"]


def test_show_blacklist_no_channel_embed_when_nothing_changed(store):
    bot = make_blacklist_bot(store)
    asyncio.run(show_blacklist(make_interaction(), bot, "add", target()))
    channel = make_channel()
    # Already listed: the gentle ephemeral note only, no announcement.
    interaction = make_interaction(channel=channel)
    asyncio.run(show_blacklist(interaction, bot, "add", target()))
    assert "すでに" in sent_text(interaction)
    channel.send.assert_not_awaited()
    # Removing someone who is not listed: likewise silent.
    interaction = make_interaction(channel=channel)
    asyncio.run(show_blacklist(interaction, bot, "remove", target(888)))
    assert "入っていない" in sent_text(interaction)
    channel.send.assert_not_awaited()


def test_show_blacklist_add_twice_and_remove_missing_are_gentle(store):
    bot = make_blacklist_bot(store)
    asyncio.run(show_blacklist(make_interaction(), bot, "add", target()))
    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "add", target()))
    assert "すでに" in sent_text(interaction)
    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "remove", target(888)))
    assert "入っていない" in sent_text(interaction)


def test_show_blacklist_empty_list(store):
    bot = make_blacklist_bot(store)
    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "list", None))
    assert "空です" in sent_text(interaction)


def test_show_blacklist_add_without_target_asks_for_one(store):
    bot = make_blacklist_bot(store)
    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "add", None))
    assert "指定して" in sent_text(interaction)
    assert list_blacklist(store) == []


def test_show_blacklist_refuses_non_owner_and_writes_nothing(store):
    bot = make_blacklist_bot(store)
    interaction = make_interaction(user_id=999)
    asyncio.run(show_blacklist(interaction, bot, "add", target()))
    assert "オーナー専用" in sent_text(interaction)
    assert list_blacklist(store) == []


def test_show_blacklist_wrong_guild_refuses(store):
    bot = make_blacklist_bot(store)
    interaction = make_interaction(guild_id=999)
    asyncio.run(show_blacklist(interaction, bot, "add", target()))
    assert "指定サーバー" in sent_text(interaction)
    assert list_blacklist(store) == []


def test_show_blacklist_owner_cannot_be_added(store):
    bot = make_blacklist_bot(store)
    interaction = make_interaction()
    asyncio.run(show_blacklist(interaction, bot, "add", target(OWNER)))
    assert "オーナーは" in sent_text(interaction)
    assert not is_blacklisted(store, OWNER)


def test_blacklist_name_is_reserved_for_builtins():
    assert "blacklist" in BUILTIN_COMMANDS


# --- slash gate (tree interaction_check) ---------------------------------


def make_gate_interaction(user_id, command_name="help"):
    return SimpleNamespace(
        command=SimpleNamespace(name=command_name),
        user=SimpleNamespace(id=user_id),
        response=SimpleNamespace(send_message=AsyncMock()),
    )


def test_slash_gate_refuses_blacklisted_member(store):
    add_to_blacklist(store, BLOCKED)
    bot = SimpleNamespace(server_log=store)
    interaction = make_gate_interaction(BLOCKED)
    assert asyncio.run(slash_blacklist_check(bot, interaction)) is False
    interaction.response.send_message.assert_awaited_once()
    _args, kwargs = interaction.response.send_message.call_args
    assert kwargs["ephemeral"] is True


def test_slash_gate_passes_unlisted_member(store):
    bot = SimpleNamespace(server_log=store)
    interaction = make_gate_interaction(888)
    assert asyncio.run(slash_blacklist_check(bot, interaction)) is True
    interaction.response.send_message.assert_not_awaited()


def test_slash_gate_exempts_blacklist_command_itself(store):
    # Even a listed member reaches the /blacklist handler (which is
    # owner-only), so the list can never lock its own management out.
    add_to_blacklist(store, BLOCKED)
    bot = SimpleNamespace(server_log=store)
    interaction = make_gate_interaction(BLOCKED, command_name="blacklist")
    assert asyncio.run(slash_blacklist_check(bot, interaction)) is True
    interaction.response.send_message.assert_not_awaited()


# --- message gate (on_message) --------------------------------------------


def settings():
    return Settings(
        "token", "key", "https://example.com", "test",
        SandboxSettings(), guild_id=123, owner_id=456,
    )


def make_message_bot(store):
    user = SimpleNamespace(id=99)
    channel = SimpleNamespace(
        id=1, send=AsyncMock(), typing=Mock(side_effect=asyncio.Lock)
    )
    bot = SimpleNamespace(
        settings=settings(),
        user=user,
        busy=set(),
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value="reply")),
        server_log=store,
        archive=AsyncMock(),
        get_guild=Mock(return_value=None),
        improvements=SimpleNamespace(settings=settings()),
        improvement_tasks=set(),
    )
    return bot, user, channel


def make_message(author_id, channel, user, content="<@99> hello"):
    return SimpleNamespace(
        author=SimpleNamespace(
            id=author_id, bot=False, name="someone",
            display_name="Someone", nick=None,
        ),
        content=content,
        guild=SimpleNamespace(id=123),
        channel=channel,
        mentions=[user],
        id=1000,
    )


def test_blacklisted_mention_is_dropped_silently(store):
    add_to_blacklist(store, BLOCKED)
    bot, user, channel = make_message_bot(store)
    asyncio.run(Bot.on_message(bot, make_message(BLOCKED, channel, user)))
    bot.assistant.reply.assert_not_awaited()
    channel.send.assert_not_awaited()


def test_blacklisted_prefix_command_is_ignored(store):
    add_to_blacklist(store, BLOCKED)
    bot, user, channel = make_message_bot(store)
    message = make_message(BLOCKED, channel, user, content="n!del")
    with patch(
        "nezubot_discrod.bot.handle_prefix_command", new=AsyncMock()
    ) as handler:
        asyncio.run(Bot.on_message(bot, message))
    handler.assert_not_called()
    bot.assistant.reply.assert_not_awaited()


def test_unlisted_member_still_reaches_the_assistant(store):
    bot, user, channel = make_message_bot(store)
    asyncio.run(Bot.on_message(bot, make_message(OWNER, channel, user)))
    bot.assistant.reply.assert_awaited_once()
    channel.send.assert_awaited_once_with("reply\n-# test")


def test_show_blacklist_list_shows_server_nickname(store):
    add_to_blacklist(store, BLOCKED)
    bot = make_blacklist_bot(store)
    member = SimpleNamespace(id=BLOCKED, display_name="サーバーでの名前")
    guild = SimpleNamespace(get_member=Mock(return_value=member))
    interaction = make_interaction()
    interaction.guild = guild
    asyncio.run(show_blacklist(interaction, bot, "list", None))
    text = sent_text(interaction)
    assert "サーバーでの名前" in text
    assert f"ID: {BLOCKED}" in text


def test_show_blacklist_list_fetches_member_when_not_cached(store):
    add_to_blacklist(store, BLOCKED)
    bot = make_blacklist_bot(store)
    member = SimpleNamespace(id=BLOCKED, display_name="取得した名前")
    guild = SimpleNamespace(
        get_member=Mock(return_value=None),
        fetch_member=AsyncMock(return_value=member),
    )
    interaction = make_interaction()
    interaction.guild = guild
    asyncio.run(show_blacklist(interaction, bot, "list", None))
    text = sent_text(interaction)
    assert "取得した名前" in text
    guild.fetch_member.assert_awaited_once_with(BLOCKED)
