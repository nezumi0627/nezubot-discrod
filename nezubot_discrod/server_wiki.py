import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Event
from typing import NamedTuple

from dotenv import load_dotenv
from openai import OpenAI

from nezubot_discrod.config import ModelSettings, Settings
from nezubot_discrod.messages import token_usage
from nezubot_discrod.prompts import SERVER_WIKI_BUILDER_SYSTEM
from nezubot_discrod.server_log import ServerLog, _check_private_file

logger = logging.getLogger(__name__)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS server_wiki_batch (
    guild TEXT NOT NULL, channel TEXT NOT NULL, start INTEGER NOT NULL, stop INTEGER NOT NULL,
    digest TEXT NOT NULL, PRIMARY KEY (guild, channel, start)
);
CREATE TABLE IF NOT EXISTS server_wiki_entry (
    guild TEXT NOT NULL, channel TEXT NOT NULL, start INTEGER NOT NULL, kind TEXT NOT NULL,
    key TEXT NOT NULL, title TEXT NOT NULL, summary TEXT NOT NULL, sources TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS server_wiki_lookup ON server_wiki_entry(guild, kind, key,
    channel, start);
CREATE TABLE IF NOT EXISTS server_wiki_skipped (
    guild TEXT NOT NULL, channel TEXT NOT NULL, start INTEGER NOT NULL,
    message_id TEXT NOT NULL, reason TEXT NOT NULL,
    PRIMARY KEY (guild, channel, start, message_id)
);
CREATE TABLE IF NOT EXISTS server_wiki_sample (
    guild TEXT NOT NULL, channel TEXT NOT NULL, start INTEGER NOT NULL, message_ids TEXT NOT NULL,
    PRIMARY KEY (guild, channel, start)
);
"""
_ROW_SQL = """
SELECT m.rowid, m.id, m.channel_id, m.author_id, m.author_name, m.is_bot, m.created_at, m.edited_at,
       m.reply_to, m.text, (m.deleted OR d.id IS NOT NULL) AS deleted
FROM messages m LEFT JOIN deleted_messages d ON d.id = m.id
WHERE m.guild_id = ? AND m.rowid > ? AND m.rowid <= ?
"""
_BATCH_KEY = "WHERE guild=? AND channel=? AND start=?"
_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_EMOJI = re.compile(r"<a?:\w+:\d+>")
_COMMAND = re.compile(r"^(?:[/!$][a-zA-Z][\w-]*\b|</[^>]+:\d+>)")
_PROFILE_ROWS = """
WITH counts AS (
    SELECT author_id, channel_id, count(*) AS total FROM messages m
    WHERE guild_id=? AND is_bot=0 AND deleted=0
        AND NOT EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id=m.id)
    GROUP BY author_id, channel_id
), best AS (
    SELECT *, row_number() OVER (
        PARTITION BY author_id ORDER BY total DESC, channel_id
    ) AS priority FROM counts
), ranked AS (
    SELECT m.rowid, b.total, row_number() OVER (
        PARTITION BY m.author_id ORDER BY m.created_at DESC, m.id DESC
    ) AS position
    FROM messages m JOIN best b USING(author_id, channel_id)
    WHERE m.guild_id=? AND m.deleted=0 AND m.is_bot=0 AND b.priority=1
        AND NOT EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id=m.id)
)
SELECT m.rowid, m.id, m.channel_id, m.author_id, m.author_name, m.is_bot, m.created_at, m.edited_at,
       m.reply_to, m.text, m.deleted
FROM ranked r JOIN messages m ON m.rowid=r.rowid WHERE r.position <= 48
ORDER BY r.total DESC, m.author_id, r.position
"""


class BuildResult(NamedTuple):
    done: int
    unchanged: int
    input_tokens: int
    output_tokens: int
    splits: int
    skipped: int
    filtered: int


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _message(row):
    return {
        "id": row["id"], "author_id": row["author_id"], "author": row["author_name"],
        "bot": bool(row["is_bot"]), "date": row["created_at"], "edited": row["edited_at"],
        "reply_to": row["reply_to"], "text": row["text"],
    }


def _messages(rows):
    return [_message(r) for r in sorted(rows, key=lambda r: (r["created_at"], r["id"]))
            if not r["deleted"]]


def _digest(messages):
    return hashlib.sha256(_json(messages).encode()).hexdigest()


def _first_object(content):
    if not isinstance(content, str):
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            data, _ = decoder.raw_decode(content, match.start())
        except (ValueError, RecursionError):
            continue
        return data
    return None


def _summaries(content, messages):
    data = _first_object(content)
    if data is None:
        return None
    by_id = {m["id"]: m for m in messages}
    entries = []
    empty = True
    for kind, group, limit, field in [
        ("person", "people", 16, "id"), ("event", "events", 8, "title")
    ]:
        items = data.get(group)
        empty = empty and items == []
        if not isinstance(items, list):
            continue
        accepted = 0
        for item in items:
            if not isinstance(item, dict) or set(item) != {field, "summary", "sources"}:
                continue
            label, summary, sources = item[field], item["summary"], item["sources"]
            if (
                not isinstance(label, str) or not label.strip() or len(label) > 100
                or not isinstance(summary, str) or not summary.strip() or len(summary) > 240
                or not isinstance(sources, list) or not 1 <= len(sources) <= 3
                or any(not isinstance(s, str) or s not in by_id for s in sources)
            ):
                continue
            try:
                label.encode()
                summary.encode()
            except UnicodeEncodeError:
                continue
            sources = list(dict.fromkeys(sources))
            if kind == "person":
                if not any(by_id[s]["author_id"] == label and not by_id[s]["bot"] for s in sources):
                    continue
                key, title = label, ""
            else:
                key, title = min(sources, key=lambda s: (by_id[s]["date"], s)), label.strip()
            entries.append((kind, key, title, summary.strip(), _json(sources)))
            accepted += 1
            if accepted == limit:
                break
    return entries if entries or empty else None


def _aliases(messages):
    aliases = {
        (m["author_id"], m["author"]): m["id"] for m in messages if not m["bot"]
    }
    return [
        ("person", author_id, name, "", _json([message_id]))
        for (author_id, name), message_id in aliases.items()
    ]


def _compact_text(message):
    if message["bot"]:
        return ""
    text = message["text"].strip()
    if not text or _COMMAND.match(text):
        return ""
    text = _URL.sub("↗", text)
    if not any(char.isalnum() for char in _EMOJI.sub("", text)):
        return ""
    text = " ".join(text.split())
    return text if len(text) <= 600 else text[:450] + " … " + text[-147:]


def _input_weight(text):
    return len(_json(text).encode()) + 128 if text else 0


def _model_payload(messages, *, profiles):
    authors, rows = {}, []
    for message in messages:
        author = (message["author_id"], message["author"])
        index = authors.setdefault(author, len(authors))
        rows.append([
            message["id"], index, message["date"][:16], message["reply_to"],
            message["compact_text"],
        ])
    return _json({
        "task": (
            "Build one supported person observation per author when possible; return events: []."
            if profiles else "Extract notable server events and durable person observations."
        ),
        "legend": "authors: [Discord ID, name]. messages: [source ID, 0-based author index, "
                  "date (UTC), reply-to source ID, text]. Cite source IDs and use Discord IDs.",
        "authors": list(authors),
        "messages": rows,
    })


@contextmanager
def _readonly(path):
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        yield db


class ServerWiki:
    def __init__(self, archive, path=None):
        self.archive = archive
        self.path = Path(path) if path is not None else archive.path.with_name("server_wiki.db")

    @contextmanager
    def writer(self):
        path = ServerLog(self.path, checkout=self.archive.checkout)._guarded_path()
        if path == self.archive.path.resolve() or (
            path.exists() and path.samefile(self.archive.path)
        ):
            raise ValueError("Wiki output must not be the archive")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            _check_private_file(path.lstat())
        else:
            os.close(fd)
        with closing(sqlite3.connect(path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            db.executescript(_SCHEMA)
            yield db

    def batches(self, archive, batch_size, byte_budget, boundaries):
        guild = self.archive.guild_id
        edges = boundaries.keys() | {(channel, stop) for (channel, _), stop in boundaries.items()}
        ceiling = archive.execute("SELECT COALESCE(max(rowid), 0) FROM messages").fetchone()[0]
        buffers, cursor = {}, 0
        while rows := archive.execute(
            _ROW_SQL + " ORDER BY m.rowid LIMIT 500", (guild, cursor, ceiling)
        ).fetchall():
            for row in rows:
                channel = row["channel_id"]
                start, size, group = buffers.get(channel, (0, 0, []))
                text = _compact_text(_message(row)) if not row["deleted"] else ""
                weight = _input_weight(text)
                if group and (channel, start) not in boundaries and (
                    len(group) >= batch_size or size + weight > byte_budget
                ):
                    stop = group[-1]["rowid"]
                    yield (guild, channel, start, stop), _messages(group)
                    start, size, group = stop, 0, []
                group.append(row)
                if (channel, row["rowid"]) in edges:
                    yield (guild, channel, start, row["rowid"]), _messages(group)
                    buffers[channel] = row["rowid"], 0, []
                else:
                    buffers[channel] = start, size + weight, group
            cursor = rows[-1]["rowid"]
        for channel, (start, _, group) in buffers.items():
            if group:
                yield (guild, channel, start, group[-1]["rowid"]), _messages(group)

    def profile_batches(self, archive, byte_budget):
        guild = self.archive.guild_id
        by_author, fallback = defaultdict(list), {}
        for row in archive.execute(_PROFILE_ROWS, (guild, guild)):
            group = by_author[row["author_id"]]
            if len(group) < 8 and _compact_text(_message(row)):
                group.append(row)
            fallback.setdefault(row["author_id"], row)
        channels = defaultdict(list)
        for author, group in by_author.items():
            if not group:
                group = [fallback[author]]
            channels[group[0]["channel_id"]].append(group)
        for channel, authors in sorted(channels.items(), key=lambda item: -len(item[1])):
            rows = []
            size = count = 0
            for author in authors:
                weight = sum(_input_weight(_compact_text(_message(row))) for row in author)
                if rows and (count == 12 or size + weight > byte_budget):
                    yield self._profile_batch(guild, channel, rows)
                    rows, size, count = [], 0, 0
                rows.extend(author)
                size += weight
                count += 1
            if rows:
                yield self._profile_batch(guild, channel, rows)

    @staticmethod
    def _profile_batch(guild, channel, rows):
        return (
            guild, channel, -min(row["rowid"] for row in rows), max(row["rowid"] for row in rows)
        ), _messages(rows)

    def build(
        self, summarize, *, model, batch_size=400, byte_budget=64_000, workers=4,
        retry_skipped=False, people_first=False, people_only=False,
    ):
        done = unchanged = input_tokens = output_tokens = splits = 0
        filtered = 0
        with (
            _readonly(self.archive.path) as archive,
            self.writer() as db,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            seen = set()
            pending = {}

            def save(future):
                nonlocal done, input_tokens, output_tokens, splits
                key, digest, aliases, retrying, sample_ids = pending.pop(future)
                entries, failed, subdivided, inputs, outputs = future.result()
                with db:
                    if not retrying:
                        db.execute(f"DELETE FROM server_wiki_entry {_BATCH_KEY}", key[:3])
                        db.execute(
                            "INSERT OR REPLACE INTO server_wiki_batch VALUES (?,?,?,?,?)",
                            (*key, digest),
                        )
                        if sample_ids:
                            db.execute(
                                "INSERT OR REPLACE INTO server_wiki_sample VALUES (?,?,?,?)",
                                (*key[:3], _json(sample_ids)),
                            )
                    db.execute(f"DELETE FROM server_wiki_skipped {_BATCH_KEY}", key[:3])
                    db.executemany(
                        "INSERT INTO server_wiki_entry VALUES (?,?,?,?,?,?,?,?)",
                        [(*key[:3], *entry) for entry in entries + aliases],
                    )
                    db.executemany(
                        "INSERT INTO server_wiki_skipped VALUES (?,?,?,?,?)",
                        [(*key[:3], message_id, "invalid model output") for message_id in failed],
                    )
                done += 1
                splits += subdivided
                input_tokens += inputs
                output_tokens += outputs
                if done % 25 == 0:
                    logger.info(
                        "Checkpoint: %d built, %d unchanged, %d splits", done, unchanged, splits,
                    )

            boundaries = {
                (row["channel"], row["start"]): row["stop"] for row in db.execute(
                    "SELECT channel, start, stop FROM server_wiki_batch WHERE guild=? AND start>=0",
                    (self.archive.guild_id,),
                )
            }

            def drain():
                while pending:
                    for ready in wait(pending, return_when=FIRST_COMPLETED).done:
                        save(ready)

            def batches():
                if people_first or people_only:
                    logger.info("Server Wiki: building member profiles")
                    yield from self.profile_batches(archive, byte_budget)
                    drain()
                if not people_only:
                    logger.info("Server Wiki: processing chronological archive")
                    yield from self.batches(archive, batch_size, byte_budget, boundaries)

            for key, messages in batches():
                seen.add(key[:3])
                profiles = key[2] < 0
                digest = _digest(messages)
                version = _digest(
                    [SERVER_WIKI_BUILDER_SYSTEM, f"profiles:{model}" if profiles else model, digest]
                ) + ":" + digest
                compact = [
                    {**message, "compact_text": _compact_text(message)} for message in messages
                ]
                eligible = [message for message in compact if message["compact_text"]]
                if not profiles or people_only:
                    filtered += len(messages) - len(eligible)
                retrying = db.execute(
                    "SELECT 1 FROM server_wiki_batch WHERE guild=? AND channel=? AND start=? "
                    "AND stop=? AND digest=?", (*key, version),
                ).fetchone() is not None
                aliases = []
                if retrying:
                    failed = {
                        row[0] for row in db.execute(
                            f"SELECT message_id FROM server_wiki_skipped {_BATCH_KEY}", key[:3]
                        )
                    } if retry_skipped else set()
                    if not failed:
                        unchanged += 1
                        continue
                    # The old builder could have checkpointed now-filtered messages as failures.
                    eligible = [message for message in eligible if message["id"] in failed]
                else:
                    aliases = _aliases(messages)
                future = pool.submit(_extract, summarize, eligible, profiles=profiles)
                sample_ids = [message["id"] for message in messages] if profiles else []
                pending[future] = (
                    key, version, aliases, retrying, sample_ids,
                )
                if len(pending) >= workers:
                    for ready in wait(pending, return_when=FIRST_COMPLETED).done:
                        save(ready)
            drain()
            for row in db.execute(
                "SELECT guild, channel, start FROM server_wiki_batch WHERE guild=?",
                (self.archive.guild_id,),
            ).fetchall():
                key = tuple(row)
                if (key in seen or (people_only and key[2] >= 0)
                        or (not people_first and not people_only and key[2] < 0)):
                    continue
                with db:
                    db.execute(f"DELETE FROM server_wiki_entry {_BATCH_KEY}", key)
                    db.execute(f"DELETE FROM server_wiki_skipped {_BATCH_KEY}", key)
                    db.execute(f"DELETE FROM server_wiki_sample {_BATCH_KEY}", key)
                    db.execute(f"DELETE FROM server_wiki_batch {_BATCH_KEY}", key)
            skipped = db.execute(
                "SELECT count(*) FROM server_wiki_skipped WHERE guild=?", (self.archive.guild_id,)
            ).fetchone()[0]
        return BuildResult(done, unchanged, input_tokens, output_tokens, splits, skipped, filtered)

    def lookup(self, kind, query, channel_ids):
        channels = sorted({str(channel) for channel in channel_ids})
        if not channels or not query.strip() or not self.path.exists():
            return []
        guild = self.archive.guild_id
        with _readonly(self.archive.path) as archive, _readonly(self.path) as db:
            archive.execute("BEGIN")
            if kind == "person":
                value = query.strip()
                if mention := re.fullmatch(r"<@!?([0-9]+)>", value):
                    value = mention[1]
                if value.isascii() and value.isdecimal():
                    ids = (value,)
                elif resolve := getattr(self.archive, "_author_ids", None):
                    ids = resolve(archive, value, channels)
                else:
                    return []
                if not ids:
                    return []
                match = f"e.key IN ({','.join('?' for _ in ids)})"
                values = list(ids)
            elif kind == "event":
                match, values = "(e.key = ? OR instr(lower(e.title), lower(?)) > 0)", [query, query]
            else:
                raise ValueError("Invalid Server Wiki page kind")
            candidates = db.execute(
                "SELECT e.*, b.stop, b.digest FROM server_wiki_entry e JOIN server_wiki_batch b "
                "USING(guild, channel, start) WHERE e.guild = ? AND e.kind = ? "
                f"AND e.channel IN ({','.join('?' for _ in channels)}) AND {match} "
                "ORDER BY (e.summary <> '') DESC, b.stop DESC LIMIT 200",
                [guild, kind, *channels, *values],
            )
            return self._pages(archive, db, candidates)

    def _pages(self, archive, db, candidates):
        batches, pages, seen = {}, {}, set()
        for entry in candidates:
            key, title, summary = entry["key"], entry["title"], entry["summary"]
            item = (key, title, summary)
            target_name = "entries" if summary else "aliases"
            if item in seen or (key not in pages and len(pages) >= 5):
                continue
            if key in pages and len(pages[key][target_name]) >= 8:
                continue
            channel, start, stop = entry["channel"], entry["start"], entry["stop"]
            batch = channel, start, stop
            if batch not in batches:
                sql = _ROW_SQL.replace("FROM messages m", "FROM messages m NOT INDEXED")
                sql += " AND m.channel_id = ?"
                args = [entry["guild"], max(0, start), stop, channel]
                if start < 0:
                    sample = db.execute(
                        f"SELECT message_ids FROM server_wiki_sample {_BATCH_KEY}",
                        (entry["guild"], channel, start),
                    ).fetchone()
                    if not sample:
                        continue
                    ids = json.loads(sample[0])
                    sql += (" AND m.rowid IN (SELECT rowid FROM messages WHERE id IN ("
                            + ",".join("?" for _ in ids) + "))")
                    args.extend(ids)
                rows = archive.execute(sql, args)
                messages = _messages(rows)
                batches[batch] = (
                    {m["id"]: m for m in messages}
                    if _digest(messages) == entry["digest"].split(":")[-1] else {}
                )
            messages = batches[batch]
            if not messages:
                continue
            seen.add(item)
            page = pages.setdefault(key, {
                "id": key, "kind": entry["kind"], "entries": [], "aliases": [],
            })
            target = page[target_name]
            sources = [
                {"date": messages[source]["date"],
                 "url": f"https://discord.com/channels/{entry['guild']}/{channel}/{source}"}
                for source in json.loads(entry["sources"])
            ]
            target.append({"title": title, "summary": summary, "sources": sources})
        return list(pages.values())


def _extract(summarize, messages, *, profiles=False):
    if not messages:
        return [], [], 0, 0, 0
    content, usage = summarize(
        SERVER_WIKI_BUILDER_SYSTEM, _model_payload(messages, profiles=profiles)
    )
    inputs, outputs = usage or (0, 0)
    entries = _summaries(content, messages)
    if entries is not None:
        if profiles:
            entries = [entry for entry in entries if entry[0] == "person"]
        return entries, [], 0, inputs, outputs
    if len(messages) == 1:
        return [], [messages[0]["id"]], 0, inputs, outputs

    middle = len(messages) // 2
    left, left_skipped, left_splits, left_in, left_out = _extract(
        summarize, messages[:middle], profiles=profiles
    )
    right, right_skipped, right_splits, right_in, right_out = _extract(
        summarize, messages[middle:], profiles=profiles
    )
    return (
        left + right, left_skipped + right_skipped, 1 + left_splits + right_splits,
        inputs + left_in + right_in, outputs + left_out + right_out,
    )


def format_pages(pages):
    lines = []
    for page in pages:
        label = "人物" if page["kind"] == "person" else "出来事"
        lines.append(f"**{label} {page['id']}** — アーカイブに基づく記録（未検証）")
        for item in [*page["entries"], *page["aliases"]]:
            description = " — ".join(value for value in (item["title"], item["summary"]) if value)
            lines.append("- " + " ".join(description.split()))
            lines.extend(f"  {source['date'][:10]} {source['url']}" for source in item["sources"])
    return "\n".join(lines) or "閲覧できる Server Wiki の記録が見つかりませんでした。"


async def _build_codex(wiki, settings, model, *, workers, timeout, **options):
    from nezubot_discrod.codex import CodexClient, CodexRateLimited

    loop = asyncio.get_running_loop()
    clients = [CodexClient(settings) for _ in range(workers)]
    queue = asyncio.Queue()
    for client in clients:
        queue.put_nowait(client)
    stopping = Event()
    active = set()

    async def complete(prompt, batch):
        task = asyncio.current_task()
        active.add(task)
        client = None
        try:
            if stopping.is_set():
                raise asyncio.CancelledError
            client = await queue.get()
            for attempt in range(4):
                try:
                    async with asyncio.timeout(timeout):
                        response = await client.create(
                            model=model, messages=_request(prompt, batch), server_wiki=True,
                        )
                    return _completion(response)
                except (CodexRateLimited, TimeoutError) as error:
                    if attempt == 3:
                        raise
                    pause = (10 if isinstance(error, CodexRateLimited) else 2) * 2**attempt
                    logger.warning("Codex %s; retrying Server Wiki request", type(error).__name__)
                    await asyncio.sleep(pause + random.uniform(0, 2))
        finally:
            if client is not None:
                queue.put_nowait(client)
            active.discard(task)

    def summarize(prompt, batch):
        return asyncio.run_coroutine_threadsafe(complete(prompt, batch), loop).result()

    try:
        if workers > 1:
            async with asyncio.timeout(timeout):
                auth = await clients[0].warm_server_wiki_auth()
            for client in clients[1:]:
                client.prepare_runtime(auth_file=auth)
        build = asyncio.create_task(asyncio.to_thread(
            wiki.build, summarize, model=f"codex:{model}", workers=workers, **options,
        ))
        try:
            return await asyncio.shield(build)
        except asyncio.CancelledError:
            stopping.set()
            for task in active:
                task.cancel()
            await asyncio.gather(build, return_exceptions=True)
            raise
    finally:
        await asyncio.gather(*(client.close() for client in clients))


def _request(prompt, batch):
    return [{"role": "system", "content": prompt}, {"role": "user", "content": batch}]


def _completion(response):
    content = response.choices[0].message.content if response.choices else None
    return content, token_usage(response.usage)


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the permission-scoped server archive wiki")
    parser.add_argument("--db", type=Path, help="Input server_log.db; opened read-only")
    parser.add_argument("--guild-id", default=os.getenv("DISCORD_GUILD_ID", str(Settings.guild_id)))
    parser.add_argument("--model", help="Override AI_MODEL")
    parser.add_argument("--provider", choices=["openai"], help="Override the bot provider")
    parser.add_argument("--batch-size", type=int, default=400, help="Max rows per fresh batch")
    parser.add_argument("--batch-bytes", type=int, default=64_000, help="Fresh batch input budget")
    parser.add_argument("--workers", type=int, help="Concurrency (default: Codex 8, API 4)")
    parser.add_argument("--call-timeout", type=int, default=120, help="Request deadline in seconds")
    parser.add_argument("--retry-skipped", action="store_true", help="Retry skipped messages")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--people-only", action="store_true", help="Build member profiles and stop")
    phases.add_argument("--events-only", action="store_true", help="Skip the member-profile pass")
    args = parser.parse_args(argv)
    if not args.guild_id.isascii() or not args.guild_id.isdecimal():
        parser.error("--guild-id must be a numeric Discord ID")
    if not 1 <= args.batch_size <= 1000 or (
        args.workers is not None and not 1 <= args.workers <= 16
    ):
        parser.error("Use --batch-size 1–1000 and --workers 1–16")
    if not 1_000 <= args.batch_bytes <= 128_000:
        parser.error("Use --batch-bytes 1000–128000")
    if not 1 <= args.call_timeout <= 600:
        parser.error("Use --call-timeout 1–600 seconds")
    settings = None
    if args.provider is None:
        try:
            settings = ModelSettings.from_env()
        except ValueError as error:
            parser.error(str(error))
        model = args.model or settings.model
        api_key, base_url = settings.api_key, settings.base_url
        if settings.provider == "openai" and not api_key:
            parser.error("AI_API_KEY is required for the OpenAI provider")
    else:
        model = args.model or os.getenv("AI_MODEL")
        api_key, base_url = os.getenv("AI_API_KEY"), os.getenv("AI_BASE_URL")
        if not api_key or not base_url or not model:
            parser.error("The OpenAI override needs AI_API_KEY, AI_BASE_URL and AI_MODEL/--model")
    logging.basicConfig(level=logging.INFO)
    archive = ServerLog(
        args.db, guild_id=args.guild_id, checkout=os.getenv("SELF_CHECKOUT") or None
    )
    wiki = ServerWiki(archive)
    options = {
        "batch_size": args.batch_size, "byte_budget": args.batch_bytes,
        "retry_skipped": args.retry_skipped, "people_first": not args.events_only,
        "people_only": args.people_only,
    }
    if settings and settings.provider == "codex":
        result = asyncio.run(_build_codex(
            wiki, settings, model, workers=args.workers or 8, timeout=args.call_timeout,
            **options,
        ))
    else:
        with OpenAI(
            api_key=api_key, base_url=base_url, timeout=args.call_timeout, max_retries=2
        ) as client:
            def summarize(prompt, batch):
                return _completion(client.chat.completions.create(
                    model=model, messages=_request(prompt, batch),
                ))

            result = wiki.build(
                summarize, model=f"{base_url}:{model}",
                workers=args.workers or 4, **options,
            )
    print(
        f"Server Wiki: {result.done} batches built, {result.unchanged} unchanged; "
        f"reported tokens: {result.input_tokens} in, {result.output_tokens} out; "
        f"{result.splits} splits, {result.skipped} skipped messages, {result.filtered} filtered"
    )


if __name__ == "__main__":
    main()
