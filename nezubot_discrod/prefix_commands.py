"""The n! prefix commands, routed before normal conversation handling.

A message starting with n! (any case, no mention needed) is a command,
not conversation: n!help shows usage, n!del retracts one of the bot's
own messages, n!status shows the learning stats. Anything else that
merely contains n! is left to the normal mention/reply flow.
"""

import asyncio
import logging
import re
import time

import discord

from nezubot_discrod.status import build_status_embed, collect_status
from nezubot_discrod.updates import latest_version
from nezubot_discrod.usage_limits import build_limit_embed, parse_rate_limits

logger = logging.getLogger(__name__)

_PREFIX = re.compile(r"^n!(help|del|status|limit)\b", re.IGNORECASE)
STATUS_CACHE_SECONDS = 60.0
LIMIT_CACHE_SECONDS = 45.0
LIMIT_FETCH_TIMEOUT = 45.0
ANSWER_AUTHOR_LIMIT = 1000

HELP_TEXT = (
    "**nezubot-discrodの使い方**\n"
    "\n"
    "**n! コマンド（メンション不要・メッセージの先頭に付ける）**\n"
    "- `n!help` — この使い方を表示\n"
    "- `n!del` — nezubot-discrodの発言に**リプライ**して送ると、その発言を取り消す"
    "（質問した本人かサーバーオーナーだけ）\n"
    "- `n!status` — 学習状況（学習したメッセージ数・人数・チャンネル・学習率など）を表示\n"
    "- `n!limit` — ChatGPT/Codex の利用枠（5時間枠・週間枠の残り %）を表示\n"
    "\n"
    "**話しかけ方**\n"
    "- @nezubot-discrod へのメンションか、nezubot-discrodの発言へのリプライで質問すると、"
    "サーバーの過去ログを根拠に答えるよ\n"
    "- 長い回答は、チャンネルには要約だけ出して全文をスレッドに分けるよ\n"
    "- 会話履歴のリセットは /reset"
)


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
    if command == "help":
        await message.channel.send(HELP_TEXT)
    elif command == "del":
        await _delete_referenced(bot, message)
    elif command == "status":
        await _send_status(bot, message)
    elif command == "limit":
        await _send_limit(bot, message)
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


async def _cached_status(bot):
    cache = getattr(bot, "_status_cache", None)
    now = time.monotonic()
    if cache is not None and now - cache[0] < STATUS_CACHE_SECONDS:
        return cache[1]
    server_log = bot.server_log
    stats = await asyncio.to_thread(
        collect_status,
        server_log.path,
        teacher_model=server_log.model,
        static_model=server_log.static_model,
        server_wiki_path=server_log.path.with_name("server_wiki.db"),
    )
    bot._status_cache = (now, stats)
    return stats


async def _send_status(bot, message):
    try:
        stats = await _cached_status(bot)
    except Exception:
        logger.exception("n!status: stat collection failed")
        await message.channel.send("ステータスの集計に失敗した。時間をおいて再試行してね。")
        return
    started_at = getattr(bot, "started_at", None)
    uptime = time.monotonic() - started_at if started_at is not None else None
    embed = build_status_embed(
        stats,
        model=bot.settings.model,
        version=latest_version(),
        uptime_seconds=uptime,
    )
    await message.channel.send(embed=embed)


async def _cached_limits(bot):
    cache = getattr(bot, "_limit_cache", None)
    now = time.monotonic()
    if cache is not None and now - cache[0] < LIMIT_CACHE_SECONDS:
        return cache[1]
    fetch = getattr(bot.ai_client, "rate_limits", None)
    if fetch is None:
        raise RuntimeError("the AI client cannot read rate limits")
    raw = await asyncio.wait_for(fetch(), timeout=LIMIT_FETCH_TIMEOUT)
    usage = parse_rate_limits(raw)
    if usage is None:
        raise ValueError("rate limit response had no usable windows")
    bot._limit_cache = (now, usage)
    return usage


async def _send_limit(bot, message):
    try:
        usage = await _cached_limits(bot)
    except Exception:
        logger.warning("n!limit: usage fetch failed", exc_info=True)
        await message.channel.send("利用枠を取得できませんでした。時間をおいて再試行してね。")
        return
    await message.channel.send(embed=build_limit_embed(usage))
