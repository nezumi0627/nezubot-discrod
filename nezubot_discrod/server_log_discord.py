import asyncio
import logging

import aiohttp
import discord

from nezubot_discrod.server_log import message_record

logger = logging.getLogger(__name__)


def history_visible(channel, member):
    try:
        permissions = channel.permissions_for(member)
    except discord.ClientException:
        return False
    return permissions.view_channel and permissions.read_message_history


async def allowed_channel_ids(client, requester):
    guild = client.get_guild(client.settings.guild_id)
    if guild is None:
        return []
    if isinstance(requester, discord.Member) and requester.guild.id == guild.id:
        member = requester
    else:
        try:
            member = await guild.fetch_member(requester.id)
        except discord.HTTPException:
            return []
    channels = {channel.id: channel for channel in [*guild.channels, *guild.threads]}
    channels.update(client.log_threads)
    allowed = []
    for channel in channels.values():
        if not hasattr(channel, "history") or not history_visible(channel, member):
            continue
        if isinstance(channel, discord.Thread) and channel.is_private():
            permissions = channel.permissions_for(member)
            if not permissions.manage_threads:
                try:
                    await channel.fetch_member(member.id)
                except discord.HTTPException:
                    continue
        allowed.append(channel.id)
    return allowed


def audience_channel_ids(client, reply_channel, channel_ids):
    def age_safe(channel):
        try:
            return channel.is_nsfw() is False or reply_channel.is_nsfw() is True
        except (AttributeError, TypeError, discord.ClientException):
            return False

    guild = getattr(reply_channel, "guild", None)
    threads = getattr(client, "log_threads", {})
    if guild is None:
        get_channel = getattr(client, "get_channel", lambda value: None)
        return [value for value in channel_ids if age_safe(
            get_channel(int(value)) or threads.get(int(value))
        )]

    def public(channel):
        if channel is None or getattr(channel, "type", None) == discord.ChannelType.private_thread:
            return False
        try:
            if not history_visible(channel, guild.default_role):
                return False
            mappings = [getattr(channel, "overwrites", None)]
            if channel.type in (discord.ChannelType.public_thread, discord.ChannelType.news_thread):
                parent = channel.parent
                if parent is None or not history_visible(parent, guild.default_role):
                    return False
                mappings = [parent.overwrites, getattr(channel, "overwrites", {})]
            return all(
                isinstance(overwrites, dict) and all(
                    isinstance(overwrite, discord.PermissionOverwrite)
                    and overwrite.view_channel is not False
                    and overwrite.read_message_history is not False
                    for overwrite in overwrites.values()
                ) for overwrites in mappings
            )
        except (AttributeError, TypeError, discord.ClientException):
            return False

    if not public(reply_channel):
        return [value for value in channel_ids if str(value) == str(reply_channel.id)]
    allowed = []
    for value in channel_ids:
        channel = guild.get_channel_or_thread(int(value)) or threads.get(int(value))
        if public(channel) and age_safe(channel):
            allowed.append(value)
    return allowed


async def is_call(message, user):
    if user in message.mentions:
        return True
    reference = getattr(message, "reference", None)
    if reference and reference.message_id:
        target = getattr(reference, "resolved", None)
        author_id = getattr(getattr(target, "author", None), "id", None)
        if author_id is None:
            # The gateway gave no author for the reply target: the
            # reference is unresolved, or it resolved to a
            # DeletedReferencedMessage, which carries no author even
            # when the target message still exists (Discord sends
            # referenced_message=null for targets it cannot attach,
            # not only for actual deletions). Resolve it with one
            # fetch before deciding; any failure stays a non-call, so
            # the trigger fails closed.
            try:
                target = await message.channel.fetch_message(reference.message_id)
            except discord.HTTPException:
                target = None
            author_id = getattr(getattr(target, "author", None), "id", None)
        if author_id == user.id:
            logger.info(
                "Reply-to-bot call in channel %s",
                getattr(getattr(message, "channel", None), "id", None),
            )
            return True
    return False


BACKFILL_FLUSH_SIZE = 1000


async def backfill_channel(store, channel):
    cursor = await asyncio.to_thread(store.get_cursor, channel.id)
    batch = []

    async def flush():
        nonlocal cursor
        await asyncio.to_thread(store.upsert_many, batch, cursor_channel_id=channel.id)
        cursor = batch[-1]["id"]
        logger.info(
            "Server log backfill channel %s: %d messages, cursor %s",
            channel.id,
            len(batch),
            batch[-1]["id"],
        )
        batch.clear()

    delay = 1
    while True:
        try:
            async for message in channel.history(
                limit=None,
                oldest_first=True,
                after=discord.Object(id=int(cursor)) if cursor else None,
            ):
                delay = 1
                batch.append(message_record(message))
                if len(batch) >= BACKFILL_FLUSH_SIZE:
                    await flush()
        except (discord.HTTPException, aiohttp.ClientError, OSError) as error:
            if batch:
                await flush()
            if isinstance(error, discord.HTTPException):
                if error.status in {403, 404}:
                    logger.info(
                        "Server log backfill channel %s skipped: HTTP %s", channel.id, error.status
                    )
                    return
                if error.status != 429 and not 500 <= error.status < 600:
                    raise
            logger.info(
                "Server log backfill channel %s retrying in %ss after %s",
                channel.id, delay, getattr(error, "status", type(error).__name__),
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        else:
            if batch:
                await flush()
            return


async def backfill_guild(client):
    guild = client.get_guild(client.settings.guild_id)
    if guild is None or guild.me is None:
        return
    seen = set()

    async def visit(channel):
        if channel.id in seen:
            return
        seen.add(channel.id)
        if isinstance(channel, discord.Thread):
            client.log_threads[channel.id] = channel
        if hasattr(channel, "history") and history_visible(channel, guild.me):
            await backfill_channel(client.server_log, channel)

    for channel in [
        *guild.text_channels, *guild.voice_channels, *guild.stage_channels, *guild.threads,
    ]:
        await visit(channel)
    for parent in [*guild.text_channels, *guild.forums]:
        if not history_visible(parent, guild.me):
            continue
        for thread in parent.threads:
            await visit(thread)
        options = [{}]
        if isinstance(parent, discord.TextChannel):
            options.append(
                {"private": True, "joined": not parent.permissions_for(guild.me).manage_threads}
            )
        for kwargs in options:
            try:
                async for thread in parent.archived_threads(limit=None, **kwargs):
                    await visit(thread)
            except discord.HTTPException as error:
                logger.info("Server log archived threads %s: HTTP %s", parent.id, error.status)
    logger.info("Server log backfill finished: %d channels/threads", len(seen))
