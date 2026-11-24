import asyncio
import importlib
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from typing import Literal
from urllib.parse import unquote, urlsplit

import aiohttp
import discord
from discord import app_commands
from openai import AsyncOpenAI

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import CodexClient
from nezubot_discrod.config import Settings
from nezubot_discrod.conversation import build_conversation_context
from nezubot_discrod.feature_loader import load_features
from nezubot_discrod.improvement_ui import handle_improve, improvement_callback
from nezubot_discrod.member_wiki import MemberWiki
from nezubot_discrod.messages import (
    TEASER_TRIGGER_CHARS,
    make_teaser,
    split_message,
    split_reply,
    strip_small_text,
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
from nezubot_discrod.updates import latest_version
from nezubot_discrod.usage_limits import build_limit_embed, parse_rate_limits
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
    "- `/reset` — このチャンネルの会話履歴をリセット\n"
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
        self.log_threads = {}
        self.backfill_task = None
        self.warmup_task = None
        if settings.provider == "codex":
            self.ai_client = CodexClient(settings, run_command_enabled=run_command_enabled())
        else:
            self.ai_client = AsyncOpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=60,
                max_retries=1,
            )
        self.assistant = Assistant(
            self.ai_client,
            settings.model,
            settings.sandbox,
            wiki=MemberWiki(checkout=settings.self_checkout or None),
            wiki_completion=self.ai_client.create if settings.provider == "codex" else None,
            server_log=self.server_log,
            web=WebTools.from_env(),
            run_command_enabled=run_command_enabled(),
        )
        self.tree = app_commands.CommandTree(self)
        self.busy = set()
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

    async def close(self):
        if self.warmup_task is not None:
            self.warmup_task.cancel()
            await asyncio.gather(self.warmup_task, return_exceptions=True)
        if self.backfill_task is not None:
            self.backfill_task.cancel()
            await asyncio.gather(self.backfill_task, return_exceptions=True)
        await asyncio.to_thread(self.server_log.stop_embedder, timeout=5)
        await self.ai_client.close()
        await super().close()

    async def on_ready(self):
        logger.info("Connected as %s; configured guild %s", self.user, self.settings.guild_id)
        if self.backfill_task is None or self.backfill_task.done():
            self.backfill_task = asyncio.create_task(self.backfill())

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
        if not accepting_work():
            return
        if message.author.bot or self.user is None:
            return
        if not allowed_source(
            self.settings,
            message.guild.id if message.guild else None,
            message.author.id,
        ):
            return
        # n! prefix commands fire without a mention, ahead of the call
        # check and the busy gate, so n!del stays reachable
        # while an answer is being generated.
        if parse_prefix_command(message.content or "") is not None:
            await handle_prefix_command(self, message)
            return
        if message.guild is not None and not await is_call(message, self.user):
            return
        channel = message.channel
        if channel.id in self.busy or len(self.busy) >= 16:
            # Busy gate: drop the trigger silently. A "busy, retry later"
            # notice posted to the channel is noise the owner asked to
            # remove (2026-11-24); the drop stays visible in the log only.
            logger.info(
                "Dropping message in busy channel %s (busy=%d)",
                channel.id, len(self.busy),
            )
            return
        content = (
            message.content.replace(f"<@{self.user.id}>", "")
            .replace(f"<@!{self.user.id}>", "")
            .strip()
        )
        if not content:
            await channel.send("メッセージを入力してください。履歴の削除は /reset です。")
            return
        self.busy.add(channel.id)
        try:
            conversation = await build_conversation_context(message, self.user.id)
            async with self.slots:
                try:
                    channel_ids = await allowed_channel_ids(self, message.author)
                    retrieval = {}
                    if channel_ids:
                        retrieval["retrieval_channel_ids"] = audience_channel_ids(
                            self, channel, channel_ids,
                        )
                    async with channel.typing():
                        improvement = {}
                        if message.author.id == self.settings.owner_id:
                            improvement = {"self_improve": improvement_callback(
                                self.improvements, message, self.improvement_tasks, self.user.id
                            )}
                        answer = await self.assistant.reply(
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
                            **({"conversation": conversation} if conversation else {}),
                            **improvement,
                        )
                except Exception:
                    logger.warning("AI request failed (channel %s)", channel.id)
                    answer = "AI への接続に失敗しました。時間をおいて再試行してください。"
                parts = split_reply(answer, self.settings.model)
                sent = await deliver_answer(channel, content, answer, parts)
                record_answer_authors(self, sent, message.author.id)
        finally:
            self.busy.discard(channel.id)


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
