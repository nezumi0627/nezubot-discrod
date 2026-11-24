"""The n! prefix commands, routed before normal conversation handling.

A message starting with n! (any case, no mention needed) is a command,
not conversation. Only n!del remains: replying to one of the bot's own
messages with n!del retracts it. The informational commands moved to
ephemeral slash commands (/help, /status, /limit in bot.py); the old
n!help / n!status / n!limit spellings are still recognised here so a
message starting with one is swallowed silently instead of falling
into the conversation flow — no DM, no public post, no reaction.
Anything else that merely contains n! is left to the normal
mention/reply flow.
"""

import logging
import re

import discord

logger = logging.getLogger(__name__)

_PREFIX = re.compile(r"^n!(help|del|status|limit)\b", re.IGNORECASE)
ANSWER_AUTHOR_LIMIT = 1000


def parse_prefix_command(content):
    """The n! command name at the start of content, or None."""
    if not content:
        return None
    match = _PREFIX.match(content.strip())
    return match.group(1).lower() if match else None


def record_answer_authors(bot, sent_messages, author_id):
    """Remember who asked for each answer the bot just sent.

    n!del uses this to let the asker retract the answer they triggered.
    The map is in-memory only and bounded; after a restart the trigger
    is unknown and n!del falls back to owner-only.
    """
    store = getattr(bot, "answer_authors", None)
    if store is None:
        return
    for sent in sent_messages or ():
        message_id = getattr(sent, "id", None)
        if message_id is not None:
            store[message_id] = author_id
    while len(store) > ANSWER_AUTHOR_LIMIT:
        store.popitem(last=False)


async def handle_prefix_command(bot, message):
    """Run the n! command in message. True when the message was a command."""
    command = parse_prefix_command(getattr(message, "content", "") or "")
    if command is None:
        return False
    if command == "del":
        await _delete_referenced(bot, message)
    else:
        # Retired informational commands (help / status / limit): they
        # live on as ephemeral slash commands now. Swallow the message
        # silently — never DM, never post, never react.
        logger.info(
            "n!%s was retired in favour of slash commands; ignoring message %s",
            command,
            getattr(message, "id", None),
        )
    return True


async def _delete_referenced(bot, message):
    reference = getattr(message, "reference", None)
    if reference is None or not getattr(reference, "message_id", None):
        await message.channel.send(
            "n!del は、消したい nezubot-discrodの発言に「リプライ」して送ってね。"
        )
        return
    target = getattr(reference, "resolved", None)
    if target is None:
        try:
            target = await message.channel.fetch_message(reference.message_id)
        except discord.HTTPException:
            logger.warning(
                "n!del: referenced message %s could not be fetched",
                reference.message_id,
                exc_info=True,
            )
            await message.channel.send("その発言は見つからなかった（もう消えてるかも）。")
            return
    if getattr(getattr(target, "author", None), "id", None) != bot.user.id:
        await message.channel.send("それは nezubot-discrodの発言じゃないから消せないよ。")
        return
    requester = message.author.id
    trigger = getattr(bot, "answer_authors", {}).get(getattr(target, "id", None))
    guild_owner = getattr(message.guild, "owner_id", None) if message.guild else None
    allowed_ids = {
        candidate
        for candidate in (trigger, guild_owner, bot.settings.owner_id)
        if candidate is not None
    }
    if requester not in allowed_ids:
        await message.channel.send(
            "その発言を消せるのは、質問した本人かサーバーオーナーだけだよ。"
        )
        return
    try:
        await target.delete()
    except discord.HTTPException:
        logger.warning("n!del: failed to delete message %s", target.id, exc_info=True)
        await message.channel.send("削除に失敗した…（権限がないか、もう消えてるかも）。")
        return
    logger.info("n!del: message %s deleted at the request of %s", target.id, requester)
    try:
        await message.add_reaction("✅")
    except discord.HTTPException:
        logger.warning("n!del: ✅ reaction failed on message %s", message.id, exc_info=True)
