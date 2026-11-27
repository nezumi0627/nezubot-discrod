import asyncio
import importlib
import logging
import os
import sqlite3
import time
from collections import OrderedDict, deque
from datetime import UTC, datetime, timedelta
from typing import Literal, NamedTuple
from urllib.parse import unquote, urlsplit

import aiohttp
import discord
from discord import app_commands
from openai import AsyncOpenAI

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.blacklist import (
    add_to_blacklist,
    init_blacklist,
    is_blacklisted,
    list_blacklist,
    remove_from_blacklist,
)
from nezubot_discrod.codex import CodexClient
from nezubot_discrod.config import Settings
from nezubot_discrod.conversation import build_conversation_context
from nezubot_discrod.devin_answer import DevinAnswerer
from nezubot_discrod.feature_loader import load_features
from nezubot_discrod.improvement_ui import handle_improve, improvement_callback
from nezubot_discrod.member_wiki import MemberWiki
from nezubot_discrod.message_images import collect_turn_images
from nezubot_discrod.messages import (
    TEASER_TRIGGER_CHARS,
    Reply,
    make_teaser,
    split_message,
    split_reply,
    strip_small_text,
)
from nezubot_discrod.model_overrides import (
    MODEL_CHOICES,
    RESET_CHOICE,
    clear_model_override,
    is_devin_model,
    resolve_model,
    set_model_override,
)
from nezubot_discrod.moderation import (
    REFUSAL_TEXT,
    build_blacklist_change_embed,
    build_escalation_embed,
    check_message,
    extract_policy_marker,
    format_mute_refusal,
    get_mute_info,
    init_moderation,
    is_muted,
    record_violation,
)
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
    normalize_content,
    register_turn,
    set_turn_state,
    sweep_candidates,
)
from nezubot_discrod.persona import PERSONA
from nezubot_discrod.prefix_commands import (
    handle_prefix_command,
    parse_prefix_command,
    record_answer_authors,
)
from nezubot_discrod.prompts import (
    CODEX_TOOL_INSTRUCTIONS,
    CODEX_WIKI_TOOL_INSTRUCTIONS,
    LOG_GUIDANCE,
    SELF_MOD_PROMPT,
    SERVER_WIKI_BUILDER_SYSTEM,
    SYSTEM,
    WEB_GUIDANCE,
    WIKI_GUIDANCE,
)
from nezubot_discrod.ranking import build_rank_embed, collect_ranking
from nezubot_discrod.reading import build_bookshelf_embed, collect_bookshelf
from nezubot_discrod.sandbox import accepting_work, run_command_enabled
from nezubot_discrod.selfmod import ImprovementService
from nezubot_discrod.server_log import ServerLog, format_hits, message_record
from nezubot_discrod.server_log_discord import (
    allowed_channel_ids,
    audience_channel_ids,
    backfill_guild,
    is_call,
)
from nezubot_discrod.server_wiki import ServerWiki, format_pages
from nezubot_discrod.status import build_status_embed, collect_status
from nezubot_discrod.understanding import (
    build_me_embed,
    collect_understanding,
    record_me_use,
)
from nezubot_discrod.updates import latest_version
from nezubot_discrod.usage_limits import build_limit_embed, parse_rate_limits
from nezubot_discrod.weather_tools import WeatherTools
from nezubot_discrod.web_tools import WebTools

logger = logging.getLogger(__name__)


def proxy_from_env():
    """Discord HTTP/gateway proxy from the standard *_PROXY environment.

    Hosts whose egress requires an HTTP proxy (no direct 443) set
    HTTPS_PROXY; discord.py does not read it itself. Credentials embedded
    in the URL are split out into proxy_auth and never logged.
    """
    raw = ""
    for name in ("DISCORD_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        raw = os.environ.get(name, "")
        if raw:
            break
    if not raw:
        return {}
    parts = urlsplit(raw)
    if not parts.hostname:
        return {}
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    proxy = f"{parts.scheme or 'http'}://{host}"
    if parts.port:
        proxy += f":{parts.port}"
    options = {"proxy": proxy}
    if parts.username:
        options["proxy_auth"] = aiohttp.BasicAuth(
            unquote(parts.username), unquote(parts.password or "")
        )
    return options


def allowed_source(settings: Settings, guild_id: int | None, user_id: int) -> bool:
    return guild_id == settings.guild_id if guild_id is not None else user_id == settings.owner_id


def guild_facts(guild):
    """Discord-verified server facts for the answer context.

    The owner, creation date and member count come from Discord itself, so
    questions about them never have to be inferred from log messages.
    """
    if guild is None:
        return None
    owner = getattr(guild, "owner", None)
    owner_id = getattr(guild, "owner_id", None)
    created_at = getattr(guild, "created_at", None)
    return {
        "name": getattr(guild, "name", None),
        "owner": {
            "id": str(owner_id) if owner_id else None,
            "display_name": getattr(owner, "display_name", None),
        },
        "created_at": created_at.date().isoformat() if created_at is not None else None,
        "member_count": getattr(guild, "member_count", None),
    }


async def show_prompt(interaction, settings):
    if interaction.guild_id != settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    content = (
        f"会話用 system prompt:\n{SYSTEM}\n\n{WIKI_GUIDANCE}\n\n{LOG_GUIDANCE}"
        f"\n\nWeb 有効時:\n{WEB_GUIDANCE}"
        f"\n\nサーバー Wiki 構築用 system prompt:\n{SERVER_WIKI_BUILDER_SYSTEM}"
        f"\n\n自己改善用 system prompt:\n{SELF_MOD_PROMPT}"
    )
    if getattr(settings, "provider", "openai") == "codex":
        content += (
            f"\n\nCodex developer instructions (Wiki / Log 有効):\n{CODEX_WIKI_TOOL_INSTRUCTIONS}"
            f"\n\nCodex developer instructions (自己改善):\n{CODEX_TOOL_INSTRUCTIONS}"
        )
    parts = split_message(content)
    await interaction.response.send_message(parts[0], ephemeral=True)
    for part in parts[1:]:
        await interaction.followup.send(part, ephemeral=True)


STATUS_CACHE_SECONDS = 600.0
LIMIT_CACHE_SECONDS = 45.0
LIMIT_FETCH_TIMEOUT = 45.0

HELP_TEXT = (
    "**nezubot-discrodの使い方**\n"
    "\n"
    "**スラッシュコマンド（結果は使った本人だけに見える）**\n"
    "- `/help` — この使い方を表示\n"
    "- `/status` — 学習状況（学習したメッセージ数・人数・チャンネル・学習率など）を表示\n"
    "- `/limit` — ChatGPT/Codex の利用枠（5時間枠・週間枠の残り %）を表示\n"
    "- `/me` — nezubot-discrodが自分をどれくらい理解しているか（理解度 % と内訳）を表示。"
    "使うと裏で自分の理解を深める処理が走る\n"
    "- `/model` — 自分への回答に使うモデルを切り替える（自分だけに効く。"
    "デフォルトに戻すこともできる）\n"
    "- `/rank` — サーバーのメッセージ数ランキング（トップ5と1位の詳細）を表示\n"
    "- `/reset` — このチャンネルの会話履歴をリセット\n"
    "- `/blacklist` — ブラックリストの追加・削除・一覧（オーナー専用。"
    "登録された人はコマンドも回答も使えなくなる）\n"
    "\n"
    "**n! コマンド（メンション不要・メッセージの先頭に付ける）**\n"
    "- `n!del` — nezubot-discrodの発言に**リプライ**して送ると、その発言を取り消す"
    "（質問した本人かサーバーオーナーだけ）\n"
    "\n"
    "**話しかけ方**\n"
    "- @nezubot-discrod へのメンションか、nezubot-discrodの発言へのリプライで質問すると、"
    "サーバーの過去ログを根拠に答えるよ\n"
    "- 長い回答は、チャンネルには要約だけ出して全文をスレッドに分けるよ"
)


async def show_help(interaction, bot):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    parts = split_message(HELP_TEXT)
    await interaction.response.send_message(parts[0], ephemeral=True)
    for part in parts[1:]:
        await interaction.followup.send(part, ephemeral=True)


async def _fetch_status(bot):
    server_log = bot.server_log
    return await asyncio.to_thread(
        collect_status,
        server_log.path,
        teacher_model=server_log.model,
        static_model=server_log.static_model,
        server_wiki_path=server_log.path.with_name("server_wiki.db"),
    )


async def _refresh_status(bot):
    try:
        stats = await _fetch_status(bot)
        bot._status_cache = (time.monotonic(), stats)
    except Exception:
        logger.exception("/status: background stat refresh failed")
    finally:
        bot._status_refreshing = False


async def cached_status(bot):
    """Status stats with stale-while-revalidate caching.

    collect_status reads the worker-maintained snapshot row, so a fetch
    is normally milliseconds; the cache plus background refresh keep
    even the fallback full aggregation off the /status path — a stale
    value is served instantly while one background task recomputes.
    """
    cache = getattr(bot, "_status_cache", None)
    now = time.monotonic()
    if cache is not None:
        if now - cache[0] < STATUS_CACHE_SECONDS:
            return cache[1]
        if not getattr(bot, "_status_refreshing", False):
            bot._status_refreshing = True
            asyncio.ensure_future(_refresh_status(bot))
        return cache[1]
    stats = await _fetch_status(bot)
    bot._status_cache = (time.monotonic(), stats)
    return stats


async def show_status(interaction, bot):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    try:
        stats = await cached_status(bot)
    except Exception:
        logger.exception("/status: stat collection failed")
        await interaction.followup.send(
            "ステータスの集計に失敗した。時間をおいて再試行してね。", ephemeral=True
        )
        return
    started_at = getattr(bot, "started_at", None)
    uptime = time.monotonic() - started_at if started_at is not None else None
    embed = build_status_embed(
        stats,
        model=bot.settings.model,
        version=latest_version(),
        uptime_seconds=uptime,
    )
    await interaction.followup.send(embed=embed, ephemeral=True)


async def cached_limits(bot):
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


async def show_limit(interaction, bot):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    try:
        usage = await cached_limits(bot)
    except Exception:
        logger.warning("/limit: usage fetch failed", exc_info=True)
        await interaction.followup.send(
            "利用枠を取得できませんでした。時間をおいて再試行してね。", ephemeral=True
        )
        return
    await interaction.followup.send(embed=build_limit_embed(usage), ephemeral=True)


async def show_me(interaction, bot):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    server_log = bot.server_log
    user_id = str(interaction.user.id)
    try:
        _use_count, queued = await asyncio.to_thread(
            record_me_use, server_log, user_id
        )
        report = await asyncio.to_thread(
            collect_understanding,
            server_log,
            user_id,
            server_wiki_path=server_log.path.with_name("server_wiki.db"),
        )
    except Exception:
        logger.exception("/me: understanding collection failed")
        await interaction.followup.send(
            "理解度の集計に失敗した。時間をおいて再試行してね。", ephemeral=True
        )
        return
    display_name = getattr(interaction.user, "display_name", None) or str(
        interaction.user
    )
    embed = build_me_embed(report, display_name=display_name, queued=queued)
    await interaction.followup.send(embed=embed, ephemeral=True)


async def show_rank(interaction, bot):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    try:
        ranking = await asyncio.to_thread(collect_ranking, bot.server_log.path)
    except Exception:
        logger.exception("/rank: ranking collection failed")
        await interaction.followup.send(
            "ランキングの集計に失敗した。時間をおいて再試行してね。", ephemeral=True
        )
        return
    await interaction.followup.send(embed=build_rank_embed(ranking), ephemeral=True)


async def show_bookshelf(interaction, bot):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    try:
        books = await asyncio.to_thread(
            collect_bookshelf, bot.server_log.path.with_name("reading.db")
        )
    except Exception:
        logger.exception("/bookshelf: book collection failed")
        await interaction.followup.send(
            "本棚の取得に失敗した。時間をおいて再試行してね。", ephemeral=True
        )
        return
    await interaction.followup.send(
        embed=build_bookshelf_embed(books), ephemeral=True
    )


async def show_model(interaction, bot, choice):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    user_id = str(interaction.user.id)
    default = bot.settings.model
    server_log = bot.server_log
    try:
        current = await asyncio.to_thread(
            resolve_model, server_log, user_id, default
        )
        if choice == RESET_CHOICE:
            removed = await asyncio.to_thread(
                clear_model_override, server_log, user_id
            )
            if removed:
                text = (
                    "回答モデルをデフォルトに戻したよ。\n"
                    f"いまの有効モデル: {current} → デフォルト（{default}）\n"
                    "次に話しかけたときからデフォルトで答えるね。"
                )
            else:
                text = (
                    f"もともとデフォルト（{default}）で答えているよ。"
                    "個別のモデル指定は付いていないよ。"
                )
        elif choice not in MODEL_CHOICES:
            text = "そのモデルは選べないよ。/model の選択肢から選んでね。"
        else:
            catalog_slugs = getattr(bot.ai_client, "catalog_slugs", None)
            slugs = catalog_slugs() if callable(catalog_slugs) else None
            # swe-2-1 is a Devin backend, not a Codex catalog slug, so
            # the live-catalog check never applies to it.
            if slugs is not None and choice not in slugs and not is_devin_model(choice):
                text = (
                    f"{choice} は現在のモデルカタログに無いみたい。"
                    "別のモデルを選んでね。"
                )
            else:
                await asyncio.to_thread(
                    set_model_override, server_log, user_id, choice
                )
                text = (
                    "あなたの回答モデルを切り替えたよ。\n"
                    f"いまの有効モデル: {current} → {choice}\n"
                    f"次に話しかけたときから {choice} で答えるね"
                    "（他の人のモデルは変わらないよ）。"
                )
                if is_devin_model(choice):
                    text += (
                        "\nswe-2-1 はツール（サーバーログ検索・Web検索"
                        "など）を使わず、あらかじめ用意された文脈だけで"
                        "回答するよ。"
                    )
    except (OSError, sqlite3.Error):
        logger.exception("/model: override update failed")
        text = "モデルの切り替えに失敗した。時間をおいて再試行してね。"
    await interaction.response.send_message(text, ephemeral=True)


BLACKLIST_REFUSAL = (
    "あなたはブラックリストに登録されているため、このコマンドは利用できません。"
)

MUTE_REFUSAL = (
    "あなたは現在、ポリシー違反による利用制限中のため、このコマンドは利用できません。"
)


async def slash_blacklist_check(bot, interaction):
    """Global slash gate, wired as the tree's interaction_check.

    A blacklisted member's slash commands — built-ins and loaded
    features alike — are refused with one short ephemeral note and
    never reach their handler. A member on the moderation no-response
    list (POLICY.md: 3 points) is refused the same way while the mute
    runs, with a note that says why and how much longer the limit
    lasts. /blacklist itself is exempt so the list stays manageable
    (its handler is owner-only anyway), and a failed lookup fails
    open like the other per-user lookups: a DB hiccup must not lock
    every member out of every command.
    """
    command = getattr(interaction, "command", None)
    if command is not None and getattr(command, "name", None) == "blacklist":
        return True
    user = getattr(interaction, "user", None)
    store = getattr(bot, "server_log", None)
    if user is None or store is None:
        return True
    try:
        blocked = await asyncio.to_thread(is_blacklisted, store, user.id)
        mute_info = None
        if not blocked:
            mute_info = await asyncio.to_thread(get_mute_info, store, user.id)
    except Exception:
        logger.exception("Blacklist/mute check failed; allowing the command")
        return True
    if not blocked and mute_info is None:
        return True
    logger.info(
        "Refused slash command from %s user %s",
        "blacklisted" if blocked else "muted",
        user.id,
    )
    if blocked:
        text = BLACKLIST_REFUSAL
    elif mute_info is not None:
        text = format_mute_refusal(mute_info)
    else:
        text = MUTE_REFUSAL
    try:
        await interaction.response.send_message(text, ephemeral=True)
    except Exception:
        pass
    return False


async def post_channel_embed(channel, embed):
    """Post a moderation notice embed to a channel. Never raises: a
    failed send (no permission, channel gone, network) is logged and
    the bot's own processing carries on."""
    if channel is None:
        return
    try:
        await channel.send(embed=embed)
    except Exception:
        logger.exception("Moderation notice: channel post failed")


async def apply_policy_violation(
    bot, author, category, *, source, excerpt, channel=None
):
    """Record one policy violation (POLICY.md) and, only when it
    escalates to a mute or a policy blacklist entry, announce it with
    an embed in the channel where the violation happened. The owner
    is never DM'd, and the public embed never carries the violating
    text. A plain point addition notifies nobody and shows nothing
    anywhere. Returns the outcome, or None when recording failed — a
    DB hiccup must not break the turn."""
    store = getattr(bot, "server_log", None)
    if store is None:
        return None
    try:
        outcome = await asyncio.to_thread(
            record_violation,
            store,
            author.id,
            category,
            excerpt=excerpt,
            source=source,
            owner_id=bot.settings.owner_id,
        )
    except Exception:
        logger.exception("Moderation: violation recording failed")
        return None
    if outcome is not None and outcome.notice is not None:
        name = getattr(author, "display_name", None) or getattr(
            author, "name", None
        )
        embed = build_escalation_embed(outcome.notice, user_name=name)
        await post_channel_embed(channel, embed)
    return outcome


async def _server_display_name(interaction, bot, user_id):
    """Best-effort name of a user as shown in this server.

    A guild member's display_name is the server nickname when one is
    set, so the member lookup comes first (cache, then a fetch); the
    bot-wide user cache is only a fallback for people who left.
    """
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    guild = getattr(interaction, "guild", None)
    if guild is not None:
        member = None
        get_member = getattr(guild, "get_member", None)
        if callable(get_member):
            try:
                member = get_member(uid)
            except Exception:
                member = None
        if member is None:
            fetch_member = getattr(guild, "fetch_member", None)
            if callable(fetch_member):
                try:
                    member = await fetch_member(uid)
                except Exception:
                    member = None
        if member is not None:
            return getattr(member, "display_name", None) or str(member)
    get_user = getattr(bot, "get_user", None)
    if callable(get_user):
        try:
            cached = get_user(uid)
        except (TypeError, ValueError):
            cached = None
        if cached is not None:
            return getattr(cached, "display_name", None) or str(cached)
    return None


async def show_blacklist(interaction, bot, action, target):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    if interaction.user.id != bot.settings.owner_id:
        await interaction.response.send_message("このコマンドはオーナー専用です。", ephemeral=True)
        return
    store = bot.server_log
    notice_embed = None
    try:
        if action == "list":
            rows = await asyncio.to_thread(list_blacklist, store)
            if not rows:
                text = "ブラックリストは空です。"
            else:
                lines = [f"**ブラックリスト（{len(rows)}人）**"]
                for user_id, _added_by, created_at, _note in rows:
                    name = await _server_display_name(interaction, bot, user_id)
                    label = f"{name}（ID: {user_id}）" if name else f"ID: {user_id}"
                    day = (created_at or "")[:10]
                    lines.append(f"- {label}" + (f" — {day} 登録" if day else ""))
                text = "\n".join(lines)
        elif action in ("add", "remove"):
            if target is None:
                text = "対象ユーザーを指定してね。"
            else:
                name = getattr(target, "display_name", None) or str(target)
                if action == "add":
                    if target.id == bot.settings.owner_id:
                        text = "オーナーはブラックリストに追加できません。"
                    else:
                        added = await asyncio.to_thread(
                            add_to_blacklist,
                            store,
                            target.id,
                            added_by=interaction.user.id,
                        )
                        if added:
                            notice_embed = build_blacklist_change_embed(
                                target.id, added=True, user_name=name
                            )
                        text = (
                            f"{name} をブラックリストに追加したよ。"
                            "この人のコマンドと回答は今後すべて止まるよ。"
                            if added
                            else f"{name} はすでにブラックリストに入っているよ。"
                        )
                else:
                    removed = await asyncio.to_thread(
                        remove_from_blacklist, store, target.id
                    )
                    if removed:
                        notice_embed = build_blacklist_change_embed(
                            target.id, added=False, user_name=name
                        )
                    text = (
                        f"{name} をブラックリストから外したよ。"
                        if removed
                        else f"{name} はブラックリストに入っていないよ。"
                    )
        else:
            text = "操作は 追加 / 削除 / 一覧 から選んでね。"
    except (OSError, sqlite3.Error):
        logger.exception("/blacklist: update failed")
        text = "ブラックリストの更新に失敗した。時間をおいて再試行してね。"
    parts = split_message(text)
    await interaction.response.send_message(parts[0], ephemeral=True)
    for part in parts[1:]:
        await interaction.followup.send(part, ephemeral=True)
    if notice_embed is not None:
        # A successful add/remove is also announced in the channel
        # the command ran in (the ephemeral reply above stays the
        # operator's own confirmation).
        await post_channel_embed(
            getattr(interaction, "channel", None), notice_embed
        )


async def deliver_answer(channel, question, answer, parts):
    """Long answers get a ~100-char teaser in the channel and their full
    text in a thread on the teaser; anything else is sent as before.

    Returns the messages that were sent, so the caller can remember who
    triggered each answer (n!del retracts by that attribution).
    """
    body = strip_small_text(str(answer))
    if len(body) <= TEASER_TRIGGER_CHARS or not isinstance(channel, discord.TextChannel):
        return [await channel.send(part) for part in parts]
    teaser = await channel.send(make_teaser(body))
    try:
        name = " ".join(question.split())[:40] or f"{PERSONA.bot_name}の回答"
        thread = await teaser.create_thread(name=name, auto_archive_duration=1440)
    except Exception:
        logger.warning("thread creation failed (channel %s)", channel.id)
        try:
            await teaser.delete()
        except Exception:
            pass
        return [await channel.send(part) for part in parts]
    sent = [teaser]
    for part in parts:
        sent.append(await thread.send(part))
    return sent


# Per-channel turn queue (v1.27.0): while a channel's answer turn is
# running, further triggers wait in a FIFO queue and are answered in
# arrival order, instead of being dropped silently (the v1.9.1 busy
# drop). The cap counts messages WAITING behind the running turn, per
# channel; it is a spam guard that normal use never reaches. Override
# with BOT_CHANNEL_QUEUE_LIMIT (0 disables queueing).
DEFAULT_TURN_QUEUE_LIMIT = 8

# How many channels may hold a turn slot at once (unchanged from the
# pre-queue busy gate); triggers for a further channel are dropped.
MAX_BUSY_CHANNELS = 16


def _turn_queue_limit():
    raw = os.environ.get("BOT_CHANNEL_QUEUE_LIMIT", "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TURN_QUEUE_LIMIT
    return value if value >= 0 else DEFAULT_TURN_QUEUE_LIMIT


class _QueuedTurn(NamedTuple):
    message: object
    content: str


class Bot(discord.Client):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True
        options = proxy_from_env()
        ca_file = os.environ.get("DISCORD_PROXY_CA", "")
        if ca_file:
            # A local relay CA (see host-tools/ws-relay.py) terminates TLS
            # for the Discord hosts; trust only that CA for this client.
            import ssl

            options["connector"] = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(cafile=ca_file)
            )
        super().__init__(
            intents=intents, allowed_mentions=discord.AllowedMentions.none(), **options
        )
        self.settings = settings
        self.started_at = time.monotonic()
        self.answer_authors = OrderedDict()
        self._status_cache = None
        self._status_refreshing = False
        self._limit_cache = None
        self.server_log = ServerLog(
            checkout=settings.self_checkout or None,
            guild_id=settings.guild_id,
        )
        try:
            init_blacklist(self.server_log)
        except (OSError, sqlite3.Error):
            # The gates fail open on a missing table; a DB hiccup at
            # startup must not take the whole bot down with it.
            logger.exception("Blacklist initialisation failed")
        try:
            init_moderation(self.server_log)
        except (OSError, sqlite3.Error):
            logger.exception("Moderation initialisation failed")
        try:
            init_pending_turns(self.server_log)
        except (OSError, sqlite3.Error):
            # Recovery is a backstop; a DB hiccup at startup must not
            # take the whole bot down with it.
            logger.exception("Pending-turn initialisation failed")
        self.log_threads = {}
        self.backfill_task = None
        self.warmup_task = None
        self.recovery_task = None
        self._recovery_started = False
        self._process_started_at = datetime.now(UTC)
        if settings.provider == "codex":
            self.ai_client = CodexClient(settings, run_command_enabled=run_command_enabled())
        else:
            self.ai_client = AsyncOpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=60,
                max_retries=1,
            )
        web_tools = WebTools.from_env()
        # One-shot Devin backend for members whose /model choice is
        # swe-2-1; lazy — it only talks to the bridge when such a
        # member actually speaks.
        self.devin_answerer = DevinAnswerer()
        self.assistant = Assistant(
            self.ai_client,
            settings.model,
            settings.sandbox,
            wiki=MemberWiki(checkout=settings.self_checkout or None),
            wiki_completion=self.ai_client.create if settings.provider == "codex" else None,
            server_log=self.server_log,
            web=web_tools,
            weather=WeatherTools(web_tools) if web_tools is not None else None,
            run_command_enabled=run_command_enabled(),
            devin_responder=self.devin_answerer.answer,
        )
        self.tree = app_commands.CommandTree(self)

        async def _slash_gate(interaction):
            return await slash_blacklist_check(self, interaction)

        # One gate for every slash command on this tree, built-ins
        # and loaded features alike: blacklisted members are refused
        # centrally instead of per handler.
        self.tree.interaction_check = _slash_gate
        self.busy = set()
        self.turn_queues = {}
        self.slots = asyncio.Semaphore(4)
        self.improvements = ImprovementService(settings)
        self.improvement_tasks = set()

        @self.tree.command(name="search", description="閲覧できるサーバーログを検索")
        async def search(interaction: discord.Interaction, query: str):
            await search_logs(interaction, self, query)

        @self.tree.command(name="wiki", description="閲覧可能なサーバー Wiki を自分だけに表示")
        async def server_wiki(
            interaction: discord.Interaction, kind: Literal["person", "event"], query: str
        ):
            await search_server_wiki(interaction, self, kind, query)

        @self.tree.command(name="prompt", description="固定の system prompt を表示")
        async def prompt(interaction: discord.Interaction):
            await show_prompt(interaction, self.settings)

        @self.tree.command(name="improve", description="オーナー専用: 自己改善 PR を作成")
        async def improve(interaction: discord.Interaction, request: str):
            await handle_improve(interaction, request, self.improvements)

        @self.tree.command(name="reset", description="このチャンネルの会話履歴をリセット")
        async def reset(interaction: discord.Interaction):
            if interaction.channel_id in self.busy:
                await interaction.response.send_message(
                    "応答完了後に再試行してください。", ephemeral=True
                )
                return
            self.assistant.reset(interaction.channel_id)
            await interaction.response.send_message("会話履歴をリセットしました。", ephemeral=True)

        @self.tree.command(name="help", description="使い方を表示（自分だけ表示）")
        async def help_command(interaction: discord.Interaction):
            await show_help(interaction, self)

        @self.tree.command(name="status", description="学習状況を表示（自分だけ表示）")
        async def status_command(interaction: discord.Interaction):
            await show_status(interaction, self)

        @self.tree.command(name="limit", description="ChatGPT/Codex の利用枠を表示（自分だけ表示）")
        async def limit_command(interaction: discord.Interaction):
            await show_limit(interaction, self)

        @self.tree.command(name="me", description="自分への理解度を表示（自分だけ表示）")
        async def me_command(interaction: discord.Interaction):
            await show_me(interaction, self)

        @self.tree.command(name="rank", description="メッセージ数ランキングを表示（自分だけ表示）")
        async def rank_command(interaction: discord.Interaction):
            await show_rank(interaction, self)

        @self.tree.command(name="bookshelf", description="読書の時間の本棚を表示（自分だけ表示）")
        async def bookshelf_command(interaction: discord.Interaction):
            await show_bookshelf(interaction, self)

        @self.tree.command(name="model", description="自分の回答モデルを切り替え（自分だけ表示）")
        @app_commands.describe(model="切り替えるモデル（自分の回答だけに効く）")
        @app_commands.choices(model=[
            app_commands.Choice(name="デフォルトに戻す", value=RESET_CHOICE),
            *(
                app_commands.Choice(name=slug, value=slug)
                for slug in MODEL_CHOICES
            ),
        ])
        async def model_command(interaction: discord.Interaction, model: str):
            await show_model(interaction, self, model)

        @self.tree.command(
            name="blacklist", description="オーナー専用: ブラックリスト管理（自分だけ表示）"
        )
        @app_commands.describe(
            action="追加 / 削除 / 一覧",
            user="対象ユーザー（追加・削除のとき指定）",
        )
        @app_commands.choices(action=[
            app_commands.Choice(name="追加", value="add"),
            app_commands.Choice(name="削除", value="remove"),
            app_commands.Choice(name="一覧", value="list"),
        ])
        async def blacklist_command(
            interaction: discord.Interaction,
            action: str,
            user: discord.User = None,
        ):
            await show_blacklist(interaction, self, action, user)

    async def setup_hook(self):
        if getattr(self, "server_log", None) is not None:
            try:
                await asyncio.to_thread(importlib.import_module, "fastembed")
            except ImportError:
                logger.info("FastEmbed unavailable; server log uses lexical retrieval")
            else:
                if os.environ.get("SERVER_LOG_EMBEDDER", "").strip().lower() == "off":
                    logger.info("Server log embedder off; a separate process owns embedding")
                else:
                    self.server_log.start_embedder()
            warmup = getattr(self, "warmup_task", None)
            if hasattr(self, "_warmup_loop") and (warmup is None or warmup.done()):
                self.warmup_task = asyncio.create_task(self._warmup_loop())
        guild = discord.Object(id=self.settings.guild_id)
        self.tree.copy_global_to(guild=guild)
        load_features(self.tree, self.settings.guild_id)
        commands = await self.tree.sync(guild=guild)
        logger.info("Synced %d slash commands to guild %s", len(commands), guild.id)

    async def _warmup_loop(self):
        """Keep the dense snapshot fresh and the archive's hot index
        pages resident, so a question after a quiet period does not pay
        a snapshot rebuild or a cold FTS scan inside its deadline."""
        while True:
            await asyncio.sleep(240)
            try:
                await asyncio.to_thread(self._warmup_once)
            except Exception:
                logger.warning("Server log warmup failed", exc_info=True)

    def _warmup_once(self):
        dense = getattr(self.server_log, "_dense", None)
        if dense is not None:
            dense.request_refresh()
        with self.server_log.connect() as db:
            db.execute(
                "SELECT rowid FROM message_fts WHERE message_fts MATCH ? LIMIT 1",
                ('"ねず"',),
            ).fetchone()
            # Grounding's appellation graph: loading it cold inside a
            # turn's 400 ms grounding budget is what let the progress
            # handler interrupt resolution down to nobody. Keeping it
            # loaded here, off the turn path, means resolution always
            # sees the warm index (the store-level cache serves turns).
            try:
                from nezubot_discrod.server_log_context import _appellation_index

                _appellation_index(self.server_log, db)
            except Exception:
                pass

    async def close(self):
        if self.warmup_task is not None:
            self.warmup_task.cancel()
            await asyncio.gather(self.warmup_task, return_exceptions=True)
        if self.backfill_task is not None:
            self.backfill_task.cancel()
            await asyncio.gather(self.backfill_task, return_exceptions=True)
        recovery = getattr(self, "recovery_task", None)
        if recovery is not None:
            recovery.cancel()
            await asyncio.gather(recovery, return_exceptions=True)
        await asyncio.to_thread(self.server_log.stop_embedder, timeout=5)
        await self.ai_client.close()
        await super().close()

    async def on_ready(self):
        logger.info("Connected as %s; configured guild %s", self.user, self.settings.guild_id)
        if self.backfill_task is None or self.backfill_task.done():
            self.backfill_task = asyncio.create_task(self.backfill())
        recover = getattr(self, "_recover_unanswered", None)
        if not getattr(self, "_recovery_started", False) and callable(recover):
            # Once per process (on_ready can refire on a reconnect):
            # answer whatever a previous process left unanswered.
            self._recovery_started = True
            self.recovery_task = asyncio.create_task(recover())

    async def _recover_unanswered(self):
        try:
            await _run_unanswered_recovery(self)
        except Exception:
            logger.exception("Unanswered-turn recovery failed")

    async def backfill(self):
        try:
            await backfill_guild(self)
        except Exception:
            logger.exception("Server log backfill failed; live archiving continues")

    async def archive(self, message):
        if message.guild is not None and message.guild.id == self.settings.guild_id:
            if isinstance(message.channel, discord.Thread):
                self.log_threads[message.channel.id] = message.channel
            try:
                await asyncio.to_thread(self.server_log.upsert, message_record(message))
            except (OSError, sqlite3.Error, ValueError):
                logger.exception("Server log archive failed")

    async def on_raw_message_delete(self, payload):
        if payload.guild_id == self.settings.guild_id:
            await asyncio.to_thread(self.server_log.mark_deleted, payload.message_id)

    async def on_raw_thread_delete(self, payload):
        if payload.guild_id == self.settings.guild_id:
            self.log_threads.pop(payload.thread_id, None)

    async def on_raw_bulk_message_delete(self, payload):
        if payload.guild_id == self.settings.guild_id:
            for message_id in payload.message_ids:
                await asyncio.to_thread(self.server_log.mark_deleted, message_id)

    async def on_raw_message_edit(self, payload):
        if payload.guild_id != self.settings.guild_id:
            return
        try:
            channel = self.get_channel(payload.channel_id) or await self.fetch_channel(
                payload.channel_id
            )
            message = await channel.fetch_message(payload.message_id)
        except discord.HTTPException:
            return
        await self.archive(message)

    async def on_message(self, message: discord.Message):
        if getattr(self, "server_log", None) is not None:
            await self.archive(message)
            # Keep the prefetch warm in channels with a live
            # conversation state; a no-op everywhere else.
            assistant = getattr(self, "assistant", None)
            note = getattr(assistant, "note_channel_activity", None)
            if callable(note):
                note(message.channel.id)
        await _handle_trigger(self, message)


async def _handle_trigger(bot, message, *, origin="live"):
    """Run one message through the trigger gates and the turn
    machinery: blacklist/mute gate, prefix commands, the call
    check, moderation layer 1, then the per-channel turn slot and
    FIFO queue.

    This is the part of on_message after archiving. The unanswered-
    turn recovery pass re-runs recovered messages through this same
    function, so a recovered turn is gated and answered exactly
    like a live one; ``origin`` only labels the pending-turn row
    the acceptance points below register.
    """
    if not accepting_work():
        return
    if message.author.bot or bot.user is None:
        return
    if not allowed_source(
        bot.settings,
        message.guild.id if message.guild else None,
        message.author.id,
    ):
        return
    # Blacklist / no-response gate: a listed member's prefix
    # commands and mention/reply triggers are dropped silently
    # (INFO log only), before either can reach a handler or the
    # assistant. The no-response list is the moderation system's
    # 24h mute (POLICY.md) and is enforced exactly like the
    # blacklist. Their messages are still archived above like
    # everyone else's — the lists gate responses, not the record.
    # Both answer paths (Codex and Devin) sit below this point,
    # so one check covers them together, and a listed member is
    # never re-judged by the moderation layers below.
    store = getattr(bot, "server_log", None)
    if store is not None:
        try:
            blocked = await asyncio.to_thread(
                is_blacklisted, store, message.author.id
            )
            muted = False
            if not blocked:
                muted = await asyncio.to_thread(
                    is_muted, store, message.author.id
                )
        except Exception:
            logger.exception(
                "Blacklist/mute check failed; allowing the message"
            )
            blocked = False
            muted = False
        if blocked or muted:
            logger.info(
                "Dropping message from %s user %s",
                "blacklisted" if blocked else "muted",
                message.author.id,
            )
            return
    # n! prefix commands fire without a mention, ahead of the call
    # check and the turn queue, so n!del stays reachable
    # while an answer is being generated.
    if parse_prefix_command(message.content or "") is not None:
        await handle_prefix_command(bot, message)
        return
    if message.guild is not None and not await is_call(message, bot.user):
        return
    channel = message.channel
    content = (
        message.content.replace(f"<@{bot.user.id}>", "")
        .replace(f"<@!{bot.user.id}>", "")
        .strip()
    )
    if not content:
        await channel.send("メッセージを入力してください。履歴の削除は /reset です。")
        return
    # Moderation layer 1 (POLICY.md): a calling message whose
    # text is an unambiguous violation never reaches the model.
    # It gets one fixed refusal and the violation is recorded
    # silently — points are never shown to anyone, and only an
    # escalation (mute / blacklist) is announced, with an embed
    # in this channel. The owner is exempt from moderation
    # entirely.
    if store is not None and message.author.id != bot.settings.owner_id:
        match = check_message(content)
        if match is not None:
            logger.info(
                "Policy layer-1 hit (category=%s) from user %s",
                match.category,
                message.author.id,
            )
            await apply_policy_violation(
                bot,
                message.author,
                match.category,
                source="pattern",
                excerpt=content,
                channel=channel,
            )
            await channel.send(REFUSAL_TEXT)
            return
    # Turn slot + per-channel FIFO queue (v1.27.0). Every gate
    # above has passed, so a message that reaches this point is
    # entitled to an answer: while the channel's turn is running,
    # further triggers wait in the queue and are answered in
    # arrival order instead of being dropped silently (the
    # v1.9.1 busy drop). Two drops remain, both INFO-logged and
    # neither reachable in normal use: the per-channel waiting
    # cap (spam guard) and the global busy-channel cap. No
    # "busy, retry later" notice is posted either way — a visible
    # notice is noise the owner asked to remove (2026-11-24).
    if channel.id in bot.busy:
        queue = _turn_queue(bot, channel.id, create=True)
        if len(queue) >= _turn_queue_limit():
            logger.info(
                "Dropping message in channel %s: turn queue full "
                "(%d waiting)",
                channel.id, len(queue),
            )
            return
        queue.append(_QueuedTurn(message, content))
        logger.info(
            "Queued message in busy channel %s (waiting=%d)",
            channel.id, len(queue),
        )
        await _register_pending_turn(bot, message, content, origin=origin)
        return
    if len(bot.busy) >= MAX_BUSY_CHANNELS:
        logger.info(
            "Dropping message in channel %s: too many busy "
            "channels (busy=%d)",
            channel.id, len(bot.busy),
        )
        return
    bot.busy.add(channel.id)
    try:
        await _register_pending_turn(bot, message, content, origin=origin)
        job = _QueuedTurn(message, content)
        from_queue = False
        while job is not None:
            if from_queue:
                waiting = _turn_queue(bot, channel.id)
                logger.info(
                    "Processing queued message in channel %s "
                    "(waiting=%d)",
                    channel.id, len(waiting) if waiting else 0,
                )
            try:
                await _run_turn(bot, job)
            except Exception:
                # One failed turn must not strand the messages
                # queued behind it; discord.py would have logged
                # an escape from on_message the same way.
                logger.exception(
                    "AI turn failed (channel %s)", channel.id
                )
            queue = _turn_queue(bot, channel.id)
            job = queue.popleft() if queue else None
            from_queue = job is not None
    finally:
        bot.busy.discard(channel.id)
        queue = _turn_queue(bot, channel.id)
        if queue is not None and not queue:
            queues = getattr(bot, "turn_queues", None)
            if queues is not None:
                queues.pop(channel.id, None)
def _turn_queue(bot, channel_id, *, create=False):
    """The channel's FIFO of waiting turns (None when absent).

    Lazily created so lightweight stand-ins for the bot in tests
    do not need the attribute pre-set.
    """
    queues = getattr(bot, "turn_queues", None)
    if queues is None:
        queues = bot.turn_queues = {}
    if create:
        return queues.setdefault(channel_id, deque())
    return queues.get(channel_id)


async def _run_turn(bot, job):
    """Run one answer turn for a message that passed every gate.

    Runs for the message that took the channel's turn slot and,
    in arrival order, for each message that waited in the
    channel's queue. The wait happens before this is called, so
    queueing never counts against the turn's own timeout. Both
    answer paths (Codex and Devin) run inside Assistant.reply,
    so queued turns take whichever path the member's model
    choice selects, exactly like an unqueued turn.
    """
    message = job.message
    content = job.content
    channel = message.channel
    conversation = await build_conversation_context(message, bot.user.id)
    async with bot.slots:
        try:
            channel_ids = await allowed_channel_ids(bot, message.author)
            retrieval = {}
            if channel_ids:
                retrieval["retrieval_channel_ids"] = audience_channel_ids(
                    bot, channel, channel_ids,
                )
            async with channel.typing():
                improvement = {}
                if message.author.id == bot.settings.owner_id:
                    improvement = {"self_improve": improvement_callback(
                        bot.improvements, message, bot.improvement_tasks, bot.user.id
                    )}
                # Image attachments (v1.42.0): downloaded under caps
                # by message_images; a collection failure degrades
                # the turn to text-only, never breaks it.
                try:
                    images = await collect_turn_images(message)
                except Exception:
                    logger.exception(
                        "Turn image collection failed (channel %s)",
                        channel.id,
                    )
                    images = None
                answer = await bot.assistant.reply(
                    channel.id,
                    content,
                    member_id=message.author.id,
                    speaker_names={
                        "username": getattr(message.author, "name", None),
                        "display_name": getattr(message.author, "display_name", None),
                        "server_nickname": getattr(message.author, "nick", None),
                    },
                    guild_id=message.guild.id if message.guild else None,
                    guild_info=guild_facts(message.guild),
                    channel_ids=channel_ids,
                    **retrieval,
                    message_id=message.id,
                    **({"images": images} if images else {}),
                    **({"conversation": conversation} if conversation else {}),
                    **improvement,
                )
        except Exception:
            logger.exception("AI request failed (channel %s)", channel.id)
            answer = "AI への接続に失敗しました。時間をおいて再試行してください。"
        # Moderation layer 2 (POLICY.md): when the model
        # refused a policy-violating request it ends its
        # answer with a hidden [[policy:<category>]] marker.
        # Strip the marker here — on the delivered text, so
        # the Codex and Devin paths are covered alike — and
        # record the violation silently. The strip always
        # runs so a marker never leaks into a message;
        # recording exempts the owner inside record_violation.
        cleaned, flagged = extract_policy_marker(answer)
        if flagged is not None:
            answer = Reply(
                cleaned,
                usage=getattr(answer, "usage", None),
                sources=getattr(answer, "sources", ()),
            )
            await apply_policy_violation(
                bot,
                message.author,
                flagged,
                source="model",
                excerpt=content,
                channel=channel,
            )
        footer_model = bot.settings.model
        if getattr(bot, "server_log", None) is not None:
            try:
                footer_model = await asyncio.to_thread(
                    resolve_model,
                    bot.server_log,
                    message.author.id,
                    bot.settings.model,
                )
            except Exception:
                # The footer is cosmetic; never lose the
                # answer over a failed override lookup.
                pass
        parts = split_reply(answer, footer_model)
        sent = await deliver_answer(channel, content, answer, parts)
        record_answer_authors(bot, sent, message.author.id)
        await _mark_turn_answered(bot, message, sent)


async def _register_pending_turn(bot, message, content, *, origin):
    """Record an accepted turn as unanswered (v1.41.0), so a crash
    before its answer is delivered leaves a recoverable trace.

    Best-effort: a DB hiccup must never break the live turn. The row
    is a recovery aid, not the only line of defence — the startup
    history cross-check also adopts answers whose marking was lost.
    """
    store = getattr(bot, "server_log", None)
    if store is None:
        return
    try:
        await asyncio.to_thread(
            register_turn,
            store,
            message_id=message.id,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            author_id=message.author.id,
            content=content,
            origin=origin,
        )
    except Exception:
        logger.exception(
            "Pending-turn registration failed (message %s)", message.id
        )


async def _mark_turn_answered(bot, message, sent):
    """Mark the turn's pending row answered, with the sent messages'
    ids (v1.41.0). Best-effort like registration: when the marking is
    lost to a crash in the window after posting, the startup
    cross-check adopts the posted answer instead of answering twice.
    """
    store = getattr(bot, "server_log", None)
    if store is None:
        return
    try:
        await asyncio.to_thread(
            mark_answered, store, message.id, [item.id for item in sent]
        )
    except Exception:
        logger.exception(
            "Pending-turn answered marking failed (message %s)", message.id
        )


# Unanswered-turn recovery (v1.41.0): see pending_turns.py for the
# durable record. The recovery window bounds how old a question may
# be and still earn a late answer; past it the conversation has moved
# on and a late answer would be noise. Override with
# BOT_UNANSWERED_RECOVERY_MAX_AGE_MINUTES.
DEFAULT_RECOVERY_MAX_AGE_MINUTES = 120
RECOVERY_ANSWER_LIMIT = 20
RECOVERY_SCAN_LIMIT = 100
_RECOVERY_HISTORY_LIMIT = 60


def _recovery_max_age_minutes():
    raw = os.environ.get("BOT_UNANSWERED_RECOVERY_MAX_AGE_MINUTES", "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_RECOVERY_MAX_AGE_MINUTES
    return value if value > 0 else DEFAULT_RECOVERY_MAX_AGE_MINUTES


def _turn_content(bot, message):
    """The mention-stripped text a turn for this message would use —
    the same computation _handle_trigger performs."""
    return (
        (message.content or "")
        .replace(f"<@{bot.user.id}>", "")
        .replace(f"<@!{bot.user.id}>", "")
        .strip()
    )


async def _resolve_recovery_channel(bot, channel_id):
    get_channel = getattr(bot, "get_channel", None)
    channel = None
    if callable(get_channel):
        try:
            channel = get_channel(int(channel_id))
        except Exception:
            channel = None
    if channel is None:
        fetch_channel = getattr(bot, "fetch_channel", None)
        if callable(fetch_channel):
            try:
                channel = await fetch_channel(int(channel_id))
            except Exception:
                logger.warning(
                    "Unanswered recovery: channel %s unfetchable",
                    channel_id,
                    exc_info=True,
                )
                return None
    return channel


async def _fetch_recovery_message(channel, message_id):
    try:
        return await channel.fetch_message(int(message_id))
    except Exception:
        # Deleted, or the channel is no longer readable: the question
        # is gone either way, so there is nothing to answer.
        logger.warning(
            "Unanswered recovery: message %s in channel %s unfetchable",
            message_id,
            getattr(channel, "id", "?"),
            exc_info=True,
        )
        return None


async def _find_unclaimed_answer(bot, store, channel, trigger_id):
    """Bot messages already posted after the trigger that no answered
    turn claims: proof the answer went out just before a crash, in the
    window between posting and marking. Returns their ids (empty when
    the trigger really is unanswered)."""
    try:
        boundary = await asyncio.to_thread(
            next_trigger_id, store, channel.id, trigger_id
        )
        claimed = await asyncio.to_thread(
            claimed_answer_ids, store, channel.id
        )
    except Exception:
        logger.exception(
            "Unanswered recovery: cross-check lookup failed (channel %s)",
            channel.id,
        )
        return ()
    boundary_id = int(boundary) if boundary is not None else None
    found = []
    try:
        history = channel.history(
            limit=_RECOVERY_HISTORY_LIMIT,
            after=discord.Object(id=int(trigger_id)),
            oldest_first=True,
        )
        async for item in history:
            if boundary_id is not None and item.id >= boundary_id:
                break
            author = getattr(item, "author", None)
            if author is None or author.id != bot.user.id:
                continue
            if str(item.id) in claimed:
                continue
            if not (getattr(item, "content", "") or "").strip():
                # Embed-only posts (moderation notices and the like)
                # are not answers.
                continue
            found.append(str(item.id))
    except Exception:
        logger.exception(
            "Unanswered recovery: history fetch failed (channel %s)",
            channel.id,
        )
        return ()
    return tuple(found)


async def _recover_message(bot, store, message, answered_keys, *, origin):
    """Decide one recovered candidate's fate and, when it is owed an
    answer, run it through the ordinary trigger path.

    Returns (outcome, adopted_answer_ids); outcome is one of
    "answered", "adopted" (an answer was already posted; the caller
    records it), "superseded" (a duplicate of an answered question),
    "deferred" (the live machinery owns the channel right now) or
    "refused" (a gate stopped it, exactly as a live trigger would be
    stopped).
    """
    channel = message.channel
    if not accepting_work():
        return "deferred", ()
    if channel.id in getattr(bot, "busy", set()):
        return "deferred", ()
    queue = _turn_queue(bot, channel.id)
    if queue:
        return "deferred", ()
    adopted = await _find_unclaimed_answer(bot, store, channel, message.id)
    content = _turn_content(bot, message)
    key = (str(channel.id), str(message.author.id), normalize_content(content))
    if adopted:
        answered_keys.add(key)
        return "adopted", adopted
    if key in answered_keys:
        return "superseded", ()
    try:
        duplicate = await asyncio.to_thread(
            has_answered_duplicate,
            store,
            channel_id=channel.id,
            author_id=message.author.id,
            content=content,
            message_id=message.id,
        )
    except Exception:
        logger.exception(
            "Unanswered recovery: duplicate check failed (message %s)",
            message.id,
        )
        duplicate = False
    if duplicate:
        return "superseded", ()
    await _handle_trigger(bot, message, origin=origin)
    try:
        row = await asyncio.to_thread(get_turn, store, message.id)
    except Exception:
        row = None
    if row is not None and row["state"] == "answered":
        answered_keys.add(key)
        return "answered", ()
    queue = _turn_queue(bot, channel.id)
    if queue and any(job.message.id == message.id for job in queue):
        # Handed to the channel's live turn loop; it marks the row
        # when it delivers, like any queued turn.
        return "deferred", ()
    if channel.id in getattr(bot, "busy", set()):
        return "deferred", ()
    return "refused", ()


async def _recover_pending_row(bot, store, row, answered_keys):
    """Recover one registered-but-unanswered turn (phase A)."""
    message_id = row["message_id"]
    if not await asyncio.to_thread(claim_turn, store, message_id):
        return "skipped"
    channel = await _resolve_recovery_channel(bot, row["channel_id"])
    if channel is None:
        await asyncio.to_thread(set_turn_state, store, message_id, "dropped")
        return "dropped"
    message = await _fetch_recovery_message(channel, message_id)
    if message is None:
        await asyncio.to_thread(set_turn_state, store, message_id, "dropped")
        return "dropped"
    outcome, adopted = await _recover_message(
        bot, store, message, answered_keys, origin="live"
    )
    if outcome == "adopted":
        await asyncio.to_thread(mark_answered, store, message_id, list(adopted))
        logger.info(
            "Unanswered recovery: message %s was already answered "
            "(answer posted before the crash); adopted",
            message_id,
        )
        return "adopted"
    if outcome == "superseded":
        await asyncio.to_thread(
            set_turn_state, store, message_id, "superseded"
        )
        logger.info(
            "Unanswered recovery: message %s superseded by an answered "
            "re-ask; not answering again",
            message_id,
        )
        return "superseded"
    if outcome == "refused":
        await asyncio.to_thread(set_turn_state, store, message_id, "dropped")
        logger.info(
            "Unanswered recovery: message %s refused by the gates at "
            "recovery time; dropped",
            message_id,
        )
        return "dropped"
    if outcome == "answered":
        logger.info("Unanswered recovery: answered message %s", message_id)
    return outcome


async def _recover_sweep_candidate(bot, store, candidate, answered_keys):
    """Recover one trigger-shaped archive message that no pending row
    covers (phase B): it arrived while no bot process was running, so
    it was never registered — or it was gated out live, which the
    re-run through the trigger path simply reproduces."""
    channel = await _resolve_recovery_channel(bot, candidate["channel_id"])
    if channel is None:
        return "dropped"
    message = await _fetch_recovery_message(channel, candidate["message_id"])
    if message is None:
        return "dropped"
    outcome, _adopted = await _recover_message(
        bot, store, message, answered_keys, origin="sweep"
    )
    if outcome == "answered":
        logger.info(
            "Unanswered recovery: answered message %s (posted while "
            "the bot was down)",
            candidate["message_id"],
        )
    return outcome


async def _run_unanswered_recovery(bot):
    """Startup pass: answer what a previous process (or an outage)
    left unanswered. Runs once per process, from on_ready.

    Phase A works from the pending-turn table; phase B sweeps the
    archive for calls that arrived while the bot was down entirely.
    Both share the same decision path, so duplicates are impossible
    beyond the cross-checks documented in pending_turns.py.
    """
    store = getattr(bot, "server_log", None)
    if store is None or bot.user is None:
        return
    cutoff = datetime.now(UTC) - timedelta(
        minutes=_recovery_max_age_minutes()
    )
    cutoff_iso = cutoff.isoformat()
    try:
        expired = await asyncio.to_thread(
            expire_turns_before, store, cutoff_iso
        )
    except Exception:
        logger.exception("Unanswered recovery: expiry pass failed")
        expired = 0
    if expired:
        logger.info(
            "Unanswered recovery: expired %d turn(s) older than the "
            "recovery window",
            expired,
        )
    started_at = getattr(bot, "_process_started_at", None)
    answered_keys = set()
    answered = 0
    try:
        rows = await asyncio.to_thread(
            list_recoverable,
            store,
            since_iso=cutoff_iso,
            limit=RECOVERY_SCAN_LIMIT,
        )
    except Exception:
        logger.exception("Unanswered recovery: pending scan failed")
        rows = []
    for row in rows:
        if answered >= RECOVERY_ANSWER_LIMIT:
            break
        if started_at is not None:
            try:
                accepted = datetime.fromisoformat(row["accepted_at"])
            except (TypeError, ValueError):
                accepted = None
            if accepted is not None and accepted >= started_at:
                # Registered by THIS process: the live machinery owns
                # it. Recovery only handles what a previous process
                # left behind.
                continue
        outcome = await _recover_pending_row(bot, store, row, answered_keys)
        if outcome == "answered":
            answered += 1
    # Phase B reads the archive, so it waits for the startup backfill
    # to finish first (bounded: a failed or stuck backfill must not
    # block recovery forever — the sweep then works with whatever the
    # archive already holds).
    backfill = getattr(bot, "backfill_task", None)
    if backfill is not None and not backfill.done():
        try:
            await asyncio.wait_for(asyncio.shield(backfill), timeout=600)
        except Exception:
            logger.warning(
                "Unanswered recovery: backfill wait ended abnormally; "
                "sweeping what is archived",
                exc_info=True,
            )
    try:
        candidates = await asyncio.to_thread(
            sweep_candidates,
            store,
            bot_id=bot.user.id,
            since_iso=cutoff_iso,
            limit=RECOVERY_SCAN_LIMIT,
        )
    except Exception:
        logger.exception("Unanswered recovery: archive sweep failed")
        candidates = []
    for candidate in candidates:
        if answered >= RECOVERY_ANSWER_LIMIT:
            break
        try:
            existing = await asyncio.to_thread(
                get_turn, store, candidate["message_id"]
            )
        except Exception:
            existing = None
        if existing is not None:
            # The pending table owns this message (phase A or live).
            continue
        outcome = await _recover_sweep_candidate(
            bot, store, candidate, answered_keys
        )
        if outcome == "answered":
            answered += 1
    logger.info(
        "Unanswered recovery finished: answered=%d expired=%d",
        answered,
        expired,
    )


async def search_logs(interaction, bot, query):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    try:
        channels = await allowed_channel_ids(bot, interaction.user)
        hits = await asyncio.to_thread(
            bot.server_log.search, query[:2000], channel_ids=channels, limit=8
        )
        text = format_hits(hits)
    except (OSError, sqlite3.Error):
        logger.exception("Slash server log search failed")
        text = "ログ検索に失敗しました。時間をおいて再試行してください。"
    for part in split_message(text):
        await interaction.followup.send(
            part, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
        )


async def search_server_wiki(interaction, bot, kind, query):
    if interaction.guild_id != bot.settings.guild_id:
        await interaction.response.send_message("指定サーバーでのみ利用できます。", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    try:
        channels = await allowed_channel_ids(bot, interaction.user)
        pages = await asyncio.to_thread(
            ServerWiki(bot.server_log).lookup, kind, query.strip()[:100], channels
        )
        text = format_pages(pages)
    except (OSError, sqlite3.Error):
        logger.exception("Slash Server Wiki lookup failed")
        text = "Server Wiki は現在利用できません。"
    for part in split_message(text):
        await interaction.followup.send(
            part, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
        )


async def _serve(settings):
    async with Bot(settings) as bot:
        await bot.start(settings.discord_token)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings.from_env()
    try:
        asyncio.run(_serve(settings))
    except KeyboardInterrupt:
        pass
