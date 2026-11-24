import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import sys
import unicodedata
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter

from dotenv import load_dotenv

from nezubot_discrod.config import Settings, data_dir
from nezubot_discrod.policy import guard_host_write
from nezubot_discrod.server_log_people import (
    PEOPLE_DDL,
    drop_message_people,
    sync_message_people,
)
from nezubot_discrod.server_log_timing import stage

logger = logging.getLogger(__name__)
_CREATE_LOCK = Lock()
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

# Per-turn lexical sharing: pre_retrieve runs a lexical-fallback search and
# (inside each dense-path search) the identical standalone-bm25 FTS query.
# When a memo is installed in this context, searches with the same
# expression/scope/pool reuse the first computation instead of rescanning
# the posting lists. Inactive (None) everywhere else.
LEXICAL_MEMO = ContextVar("server_log_lexical_memo", default=None)


def new_lexical_memo():
    return {"lock": Lock(), "entries": {}}

# Per-connection SQLite tuning for the write-heavy archive (audit F14).
# synchronous=NORMAL is the standard WAL trade-off: an application crash loses
# nothing, an OS/power crash may lose the last transactions of a rebuildable
# archive. cache_size is 64 MiB (negative = KiB), mmap_size 256 MiB.
_CONNECTION_PRAGMAS = (
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 30000",
    "PRAGMA cache_size = -65536",
    "PRAGMA mmap_size = 268435456",
    "PRAGMA wal_autocheckpoint = 1000",
    "PRAGMA temp_store = MEMORY",
)


def _apply_connection_pragmas(db):
    for pragma in _CONNECTION_PRAGMAS:
        db.execute(pragma)


class DenseSnapshotUnavailable(RuntimeError):
    pass


def bigrams(text):
    text = unicodedata.normalize("NFKC", text).lower()
    out = []
    for token in re.findall(r"[\u3040-\u30ff\u3400-\u9fff]+|[a-z0-9_]+", text):
        if re.fullmatch(r"[\u3040-\u30ff\u3400-\u9fff]+", token):
            out.extend(
                [token] if len(token) == 1 else [token[i : i + 2] for i in range(len(token) - 1)]
            )
        else:
            out.append(token)
    return out


def record_text(record):
    parts = [record.get("content") or ""]
    for embed in record.get("embeds", []) or []:
        parts.extend([embed.get("title") or "", embed.get("description") or ""])
    for attachment in record.get("attachments", []) or []:
        parts.append("[file] " + attachment.get("filename", ""))
    for sticker in record.get("sticker_items", []) or []:
        parts.append("[sticker] " + sticker.get("name", ""))
    return "\n".join(part for part in parts if part).strip()


def message_record(message):
    channel = message.channel
    parent = getattr(channel, "parent", None) or getattr(channel, "category", None)
    author = message.author
    return {
        "id": str(message.id),
        "guild_id": str(message.guild.id) if message.guild else "@me",
        "channel_id": str(channel.id),
        "_channel": getattr(channel, "name", "DM"),
        "_parent": getattr(parent, "name", None),
        "author": {
            "id": str(author.id),
            "username": author.name,
            "global_name": getattr(author, "global_name", None),
            "nick": getattr(author, "nick", None),
            "display_name": getattr(author, "display_name", None),
            "bot": author.bot,
        },
        "timestamp": message.created_at.isoformat(),
        "edited_timestamp": message.edited_at.isoformat() if message.edited_at else None,
        "message_reference": {"message_id": str(message.reference.message_id)}
        if message.reference and message.reference.message_id
        else None,
        "content": message.content,
        "embeds": [embed.to_dict() for embed in message.embeds],
        "attachments": [{"filename": item.filename} for item in message.attachments],
        "sticker_items": [{"name": item.name} for item in message.stickers],
    }


def _date(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def embedding_text(row):
    return f"{row['author_name']}: {row['text']}"


def text_hash(row):
    return hashlib.sha256(embedding_text(row).encode()).hexdigest()


@dataclass(frozen=True)
class LogHit:
    id: str
    channel_id: str
    channel_name: str
    author: str
    date: str
    text: str
    url: str
    author_id: str = ""


def _hit(row):
    return LogHit(
        row["id"],
        row["channel_id"],
        row["channel_name"],
        row["author_nickname"] or row["author_name"],
        row["created_at"],
        row["text"],
        f"https://discord.com/channels/{row['guild_id']}/{row['channel_id']}/{row['id']}",
        author_id=row["author_id"],
    )


def format_hits(hits):
    lines = []
    for i, hit in enumerate(hits, 1):
        text = " ".join(hit.text.split())
        if len(text) > 300:
            text = text[:297] + "..."
        channel = " ".join(hit.channel_name.split())
        author = " ".join(hit.author.split())
        lines.append(
            f"{i}. {hit.date} #{channel} {author} (id: {hit.author_id}): {text} {hit.url}"
        )
    return "\n".join(lines) if lines else "No messages found."


def rrf(lists, limit=50):
    scores = {}
    for items in lists:
        for rank, item in enumerate(items, 1):
            scores[item] = scores.get(item, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]


def _name(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split()).removeprefix("@")


def _fusion(query, lexical, dense, limit, *, partial=False, dense_coverage=None,
            extra_dense=None):
    if partial and (dense_coverage is None or dense_coverage < 0.9):
        ids = [row["id"] for row in lexical]
        lexical_ids = set(ids)
        only = [item for item in dense if item not in lexical_ids]
        result = ids[:5]
        for i in range(max(len(ids) - 5, len(only))):
            if i < len(only):
                result.append(only[i])
            if i + 5 < len(ids):
                result.append(ids[i + 5])
        return result[:limit]
    lists = [[row["id"] for row in lexical], dense]
    if extra_dense:
        lists.append(list(extra_dense))
    tail = rrf(lists, limit)
    if not dense:
        return tail
    first = []
    if lexical:
        text = unicodedata.normalize("NFKC", embedding_text(lexical[0])).casefold()
        normalized = unicodedata.normalize("NFKC", query).casefold()
        tokens = set(bigrams(query))
        coverage = len(tokens & set(bigrams(text))) / max(1, len(tokens))
        anchors = re.findall(r"[a-z][a-z0-9_+.#-]{2,}", normalized)
        anchors += re.findall(r'[「『"]([^」』"]{2,50})[」』"]', normalized)
        # Japanese queries have no word boundaries, so the ASCII/quoted
        # anchors above never fire for them. Particle-split noun segments of
        # 3+ characters (e.g. 領事館 in "…の手続きで領事館に行った話") are the
        # equivalent strong exact-term signal.
        anchors += [
            segment
            for run in re.findall(r"[぀-ヿ㐀-鿿]+", normalized)
            for segment in re.split(r"[はがをにのではへとからまでも]", run)
            if len(segment) >= 3
        ]
        anchored = normalized in text or any(anchor in text for anchor in anchors)
        if coverage >= 0.25 and anchored:
            first.append(lexical[0]["id"])
    if dense[0] not in first:
        first.append(dense[0])
    return (first + [item for item in tail if item not in first])[:limit]


def _check_private_file(metadata):
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise PermissionError(
            "Server log SQLite files must be private regular files, no hard links"
        )


class ServerLog:
    """Private archive, lexical-only until enable_dense or start_embedder is called."""

    def __init__(
        self, path=None, *, checkout=None, guild_id=None, model=None, embedder=None,
        max_length=None, hybrid_model=None, hybrid_embedder=None, hybrid_onnx_dir=None
    ):
        self.path = (
            Path(path)
            if path is not None
            else (data_dir() / "server_log.db")
        )
        self.checkout = checkout if checkout is not None else Path(__file__).resolve().parents[1]
        self.guild_id = str(
            guild_id if guild_id is not None else os.getenv("DISCORD_GUILD_ID", Settings.guild_id)
        )
        self.model = model or os.environ.get("SERVER_LOG_EMBEDDING_MODEL", DEFAULT_MODEL)
        # Optional static-distillation fallback: vectors stored under this
        # model name fill snapshot gaps the primary model has not covered.
        self.static_model = os.environ.get("SERVER_LOG_STATIC_MODEL") or None
        # Optional local ONNX directory for models fastembed cannot load.
        self.embedding_onnx_dir = os.environ.get("SERVER_LOG_EMBEDDING_ONNX_DIR") or None
        # Optional hybrid retrieval: a second dense path (a fine-tuned
        # model) whose stored vectors are fused with the primary path.
        # Active only when both the model tag and a query-encoder source
        # (ONNX directory or an injected embedder) are configured; the
        # DenseIndex enforces that and falls back to two-path fusion.
        self.hybrid_model = (
            hybrid_model or os.environ.get("SERVER_LOG_HYBRID_MODEL") or None
        )
        self.hybrid_onnx_dir = (
            hybrid_onnx_dir or os.environ.get("SERVER_LOG_HYBRID_ONNX_DIR") or None
        )
        self._hybrid_embedder = hybrid_embedder
        self._hybrid_dense = None
        # Soft deadline for the lexical (FTS) stage of a search, in
        # milliseconds. The FTS stage runs as a sequence of small token
        # chunks under a SQLite progress handler; once the budget is spent
        # the rows gathered so far are returned instead of letting one
        # stage eat the caller's whole deadline (under load a single
        # full-expression scan could take seconds and starve the dense
        # stage entirely). 0 disables chunking (one unbounded query).
        self.lexical_deadline_ms = int(
            os.getenv("SERVER_LOG_LEXICAL_DEADLINE_MS", "450")
        )
        self.max_length = int(
            max_length if max_length is not None else os.getenv("SERVER_LOG_MAX_LENGTH", "128")
        )
        if self.max_length < 1:
            raise ValueError("SERVER_LOG_MAX_LENGTH must be positive")
        self._dense = None
        self._dense_state = "disabled"
        self._dense_init_lock = Lock()
        self._lexical_reason = None
        self._embedder = embedder
        self._worker = None
        self._worker_lock = Lock()
        self._embedding_lock = Lock()
        self._stop = Event()
        self._schema_lock = Lock()
        self._schema_ready = False

    def _guarded_path(self):
        root = self.path.parent
        guarded_root = guard_host_write(root, checkout=self.checkout, allowed_root=root)
        if guarded_root != root.parent.resolve() / root.name:
            raise PermissionError("Server log data directory must not be a symlink")
        guarded_root.mkdir(mode=0o700, exist_ok=True)
        metadata = guarded_root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise PermissionError("Server log directory must be private and owned by the bot")
        for suffix in ("", "-journal", "-wal", "-shm"):
            name = self.path.name + suffix
            path = guard_host_write(root / name, checkout=self.checkout, allowed_root=root)
            if path != guarded_root / name:
                raise PermissionError("Server log SQLite files must not be symlinks")
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            _check_private_file(metadata)
        return guarded_root / self.path.name

    @contextmanager
    def connect(self):
        with self._schema_lock:
            if not self._schema_ready:
                with self._initialize_connection():
                    pass
                self._schema_ready = True
        path = self._guarded_path()
        with closing(sqlite3.connect(path, timeout=30)) as db, db:
            db.row_factory = sqlite3.Row
            _apply_connection_pragmas(db)
            yield db

    @contextmanager
    def _initialize_connection(self):
        path = self._guarded_path()
        with _CREATE_LOCK:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                _check_private_file(path.lstat())
            else:
                os.close(fd)
        with closing(sqlite3.connect(path, timeout=30)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            _apply_connection_pragmas(db)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, guild_id TEXT NOT NULL, channel_id TEXT NOT NULL,
                    channel_name TEXT NOT NULL, parent_name TEXT, author_id TEXT NOT NULL,
                    author_name TEXT NOT NULL, is_bot INTEGER NOT NULL,
                    created_at TEXT NOT NULL, edited_at TEXT, reply_to TEXT,
                    text TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS messages_channel_date
                    ON messages(channel_id, created_at, id);
                CREATE TABLE IF NOT EXISTS backfill_cursors (
                    channel_id TEXT PRIMARY KEY, message_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deleted_messages (id TEXT PRIMARY KEY);
                CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                    tokens, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS message_vectors (
                    message_id TEXT NOT NULL, model TEXT NOT NULL, text_hash TEXT NOT NULL,
                    dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(message_id, model)
                );
                CREATE TABLE IF NOT EXISTS log_state (id INTEGER PRIMARY KEY, generation INTEGER);
                INSERT OR IGNORE INTO log_state VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS dense_state (id INTEGER PRIMARY KEY, sequence INTEGER);
                INSERT OR IGNORE INTO dense_state VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS dense_changes (
                    message_id TEXT PRIMARY KEY, version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS dense_changes_version ON dense_changes(version);
                CREATE TABLE IF NOT EXISTS log_migrations (name TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS author_aliases (
                    guild_id TEXT NOT NULL, channel_id TEXT NOT NULL, author_id TEXT NOT NULL,
                    name TEXT NOT NULL, normalized TEXT NOT NULL,
                    PRIMARY KEY(guild_id, channel_id, author_id, normalized)
                );
                CREATE INDEX IF NOT EXISTS author_alias_lookup
                    ON author_aliases(normalized, channel_id, author_id);
                CREATE INDEX IF NOT EXISTS messages_author_channel_date
                    ON messages(author_id, channel_id, created_at, id);
            """)
            db.executescript(PEOPLE_DDL)
            if not db.execute(
                "SELECT 1 FROM log_migrations WHERE name = 'speaker_identity'"
            ).fetchone():
                db.execute("BEGIN IMMEDIATE")
                columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
                for column in ("author_username", "author_global_name", "author_nickname"):
                    if column not in columns:
                        db.execute(
                            f"ALTER TABLE messages ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                        )
                for row in db.execute(
                    "SELECT DISTINCT guild_id, channel_id, author_id, author_name FROM messages"
                ).fetchall():
                    self._aliases(db, row, [row["author_id"], row["author_name"]])
                db.execute("INSERT OR IGNORE INTO log_migrations VALUES ('speaker_identity')")
                db.commit()
            for table, column in [
                ("messages", "id"),
                ("message_vectors", "message_id"),
                ("deleted_messages", "id"),
            ]:
                for action, sources in [
                    ("INSERT", ["NEW"]), ("DELETE", ["OLD"]), ("UPDATE", ["OLD", "NEW"])
                ]:
                    statements = "".join(
                        f"""
                        UPDATE dense_state SET sequence = sequence + 1 WHERE id = 1;
                        INSERT INTO dense_changes VALUES (
                            {source}.{column}, (SELECT sequence FROM dense_state WHERE id = 1)
                        ) ON CONFLICT(message_id) DO UPDATE SET version = excluded.version;
                        """
                        for source in sources
                    )
                    db.execute(f"""
                        CREATE TRIGGER IF NOT EXISTS dense_{table}_{action}
                        AFTER {action} ON {table} BEGIN {statements} END
                    """)
            yield db

    @staticmethod
    def _aliases(db, row, names):
        for name in names:
            if name and _name(name):
                db.execute(
                    "INSERT OR IGNORE INTO author_aliases VALUES (?, ?, ?, ?, ?)",
                    (row["guild_id"], row["channel_id"], row["author_id"], str(name), _name(name)),
                )

    def _upsert(self, db, record):
        author = record["author"]
        guild_id = record.get("guild_id") or self.guild_id
        if not guild_id:
            raise ValueError("Missing guild_id; supply it in the record or ServerLog(guild_id=...)")
        row = {
            "id": str(record["id"]),
            "guild_id": str(guild_id),
            "channel_id": str(record["channel_id"]),
            "channel_name": record.get("_channel") or record.get("channel_name") or "unknown",
            "parent_name": record.get("_parent"),
            "author_id": str(author["id"]),
            "author_name": (record.get("member") or {}).get("nick") or author.get("nick")
            or author.get("global_name") or author.get("display_name") or author["username"],
            "author_username": author["username"],
            "author_global_name": author.get("global_name") or "",
            "author_nickname": (record.get("member") or {}).get("nick") or author.get("nick") or "",
            "is_bot": int(bool(author.get("bot"))),
            "created_at": _date(record["timestamp"]),
            "edited_at": _date(record["edited_timestamp"])
            if record.get("edited_timestamp")
            else None,
            "reply_to": (record.get("message_reference") or {}).get("message_id"),
            "text": record_text(record),
            "deleted": int(bool(record.get("deleted", False))),
        }
        if db.execute("SELECT 1 FROM deleted_messages WHERE id = ?", (row["id"],)).fetchone():
            row["deleted"] = 1
        if row["deleted"]:
            db.execute("INSERT OR IGNORE INTO deleted_messages VALUES (?)", (row["id"],))
        previous = db.execute("SELECT rowid, * FROM messages WHERE id = ?", (row["id"],)).fetchone()
        # No-op update (audit F5): an identical row needs no alias re-inserts
        # (the same names were inserted with the original write), no FTS or
        # vector work, and no journal churn. Return before any of it.
        if previous is not None and all(previous[key] == value for key, value in row.items()):
            return
        self._aliases(db, row, [row[key] for key in (
            "author_id", "author_name", "author_username", "author_global_name", "author_nickname"
        )])
        if (
            previous is not None
            and previous["edited_at"]
            and not row["deleted"]
            and (row["edited_at"] or row["created_at"]) < previous["edited_at"]
        ):
            return
        columns = ", ".join(row)
        parameters = ", ".join("?" for _ in row)
        updates = ", ".join(f"{key} = excluded.{key}" for key in row if key != "id")
        db.execute(
            f"INSERT INTO messages ({columns}) VALUES ({parameters}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            tuple(row.values()),
        )
        rowid = db.execute("SELECT rowid FROM messages WHERE id = ?", (row["id"],)).fetchone()[0]
        # Message rows are never deleted from the archive, so a brand-new
        # rowid cannot have a stale FTS entry to remove (audit F5).
        if previous is not None:
            db.execute("DELETE FROM message_fts WHERE rowid = ?", (rowid,))
        if not row["deleted"]:
            db.execute(
                "INSERT INTO message_fts(rowid, tokens) VALUES (?, ?)",
                (
                    rowid,
                    " ".join(bigrams(embedding_text(row))),
                ),
            )
        # embedding_text is f"{author_name}: {text}", so comparing those two
        # fields directly is equivalent to comparing text_hash(previous) with
        # text_hash(row) without computing two SHA-256 hashes (audit F5).
        if (
            previous is None
            or previous["author_name"] != row["author_name"]
            or previous["text"] != row["text"]
            or row["is_bot"]
            or row["deleted"]
        ):
            db.execute("DELETE FROM message_vectors WHERE message_id = ?", (row["id"],))
        # No log_state.generation update here (audit F5): nothing in
        # production reads it; the writers in mark_deleted, the dense
        # embedder, and vector import are unchanged.
        # Person graph (Phase A): keep this message's author/mention/reply
        # rows in step with the stored row. The reply target may not be
        # archived yet during a backfill; the batch build reconciles those.
        reply_author = None
        if row["reply_to"]:
            target = db.execute(
                "SELECT author_id FROM messages WHERE id = ?", (row["reply_to"],)
            ).fetchone()
            reply_author = target[0] if target else None
        sync_message_people(db, row, reply_author)

    def upsert(self, record):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._upsert(db, record)

    def upsert_many(self, records, *, batch_size=1000, cursor_channel_id=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        count = 0
        last_id = None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for record in records:
                self._upsert(db, record)
                last_id = record["id"]
                count += 1
                if count % batch_size == 0:
                    db.commit()
                    db.execute("BEGIN IMMEDIATE")
            if cursor_channel_id is not None and last_id is not None:
                # Advance the backfill cursor in the same transaction as the
                # final chunk, so a crash cannot persist one without the other.
                self._set_cursor(db, cursor_channel_id, last_id)
        return count

    @staticmethod
    def _set_cursor(db, channel_id, message_id):
        db.execute(
            """
            INSERT INTO backfill_cursors VALUES (?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET message_id = excluded.message_id
            """,
            (str(channel_id), str(message_id)),
        )

    def backfill_authors(self, records):
        count = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for record in records:
                row = db.execute(
                    "SELECT * FROM messages WHERE id = ?", (str(record["id"]),)
                ).fetchone()
                author = record.get("author") or {}
                if row is None or row["author_id"] != str(author.get("id")):
                    continue
                values = {
                    "author_username": author.get("username") or "",
                    "author_global_name": author.get("global_name") or "",
                    "author_nickname": (record.get("member") or {}).get("nick")
                    or author.get("nick") or "",
                }
                self._aliases(db, row, [*values.values(), author.get("display_name")])
                missing = {key: value for key, value in values.items() if value and not row[key]}
                if missing:
                    db.execute(
                        "UPDATE messages SET " + ", ".join(f"{key} = ?" for key in missing)
                        + " WHERE id = ?", [*missing.values(), row["id"]],
                    )
                count += 1
        return count

    def mark_deleted(self, message_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO deleted_messages VALUES (?)", (str(message_id),))
            row = db.execute(
                "SELECT rowid FROM messages WHERE id = ?", (str(message_id),)
            ).fetchone()
            if row is None:
                return
            db.execute("UPDATE messages SET deleted = 1 WHERE id = ?", (str(message_id),))
            db.execute("DELETE FROM message_fts WHERE rowid = ?", (row[0],))
            db.execute("DELETE FROM message_vectors WHERE message_id = ?", (str(message_id),))
            drop_message_people(db, str(message_id))
            db.execute("UPDATE log_state SET generation = generation + 1 WHERE id = 1")

    def _scope(self, channel_ids, *, author=None, channel=None, after=None, before=None,
               exclude_message_id=None):
        ids = sorted({str(item) for item in channel_ids})
        if not ids:
            return "0", []
        where = [f"m.channel_id IN ({','.join('?' for _ in ids)})", "m.deleted = 0",
                 "NOT EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id)"]
        args = list(ids)
        if exclude_message_id is not None:
            where.append("m.id != ?")
            args.append(str(exclude_message_id))
        for field, value in [("author", author), ("channel", channel)]:
            if field == "author" and isinstance(value, tuple):
                if not value:
                    where.append("0")
                else:
                    where.append(f"m.author_id IN ({','.join('?' for _ in value)})")
                    args.extend(value)
            elif value is not None:
                left, right = (
                    ("author_id", "author_name")
                    if field == "author"
                    else ("channel_id", "channel_name")
                )
                where.append(f"(m.{left} = ? OR m.{right} = ?)")
                args.extend(
                    [str(value), str(value).removeprefix("#") if field == "channel" else str(value)]
                )
        for operator, value in [(">", after), ("<", before)]:
            if value is not None:
                where.append(f"m.created_at {operator} ?")
                args.append(_date(value))
        return " AND ".join(where), args

    @staticmethod
    def _author_ids(db, author, channel_ids, *, explicit=False):
        value = str(author).strip()
        mention = re.fullmatch(r"<@!?(\d+)>", value)
        if mention:
            value = mention[1]
        if explicit and value.isdecimal():
            return (value,)
        return tuple(row[0] for row in db.execute(
            "SELECT DISTINCT author_id FROM author_aliases WHERE normalized = ? "
            f"AND channel_id IN ({','.join('?' for _ in channel_ids)}) ORDER BY author_id",
            [_name(value), *channel_ids],
        ))

    def _speaker_query(self, db, query, author, channel_ids):
        if author is not None:
            return query, self._author_ids(db, author, channel_ids, explicit=True)
        candidate = re.sub(r"^from:\s*", "", query.strip(), flags=re.IGNORECASE)
        patterns = [
            r"^(.+?)(?:が|の)(?:前に|以前)?"
            r"(?:言ってた|言っていた|話してた|話していた|発言した|書いてた|書いていた|言った|投稿した)"
            r"\s*[「『]?(.*?)[」』]?\s*$",
        ]
        for pattern in patterns:
            if match := re.match(pattern, candidate, re.IGNORECASE):
                ids = self._author_ids(db, match[1], channel_ids)
                if not ids:
                    ids = self._author_ids(db, re.sub(r"(?:さん|氏)$", "", match[1]), channel_ids)
                if ids:
                    return match[2].strip(), ids
        if mention := re.match(r"<@!?\d+>", candidate):
            remaining = re.sub(r"^(?:さん|氏)(?=$|\s)", "", candidate[mention.end():]).strip()
            return remaining, self._author_ids(
                db, mention[0], channel_ids, explicit=True
            )
        boundaries = [len(candidate), *reversed([
            m.start() for m in re.finditer(r"\s+", candidate)
        ])]
        for boundary in boundaries:
            name = candidate[:boundary]
            ids = self._author_ids(db, name, channel_ids)
            if not ids:
                ids = self._author_ids(db, re.sub(r"(?:さん|氏)$", "", name), channel_ids)
            if ids:
                return candidate[boundary:].strip(), ids
        return query, None

    @staticmethod
    def _lexical_query(db, expression, where, args, pool):
        # Rank inside a standalone FTS subquery so SQLite can use the
        # bm25 top-k optimization (ORDER BY bm25 + LIMIT on the bare
        # FTS table prunes posting-list walks). Joining messages and
        # applying the scope inside the FTS query defeats it and the
        # OR of common bigrams scans hundreds of thousands of rows.
        # The subquery over-fetches so scope filtering still fills
        # the pool in narrow scopes.
        return db.execute(
            f"""
            SELECT m.*, f.fts_rank FROM (
                SELECT rowid, bm25(message_fts) AS fts_rank FROM message_fts
                WHERE message_fts MATCH ? ORDER BY bm25(message_fts) LIMIT ?
            ) f JOIN messages m ON m.rowid = f.rowid
            WHERE {where}
            ORDER BY f.fts_rank, m.id LIMIT ?
        """,
            [expression, max(200, pool * 4), *args, pool],
        ).fetchall()

    # Token chunks per FTS subquery when the soft deadline is active:
    # each chunk is one bm25 top-k scan, so a deadline abort loses only
    # the chunks that never ran, not the whole stage.
    _LEXICAL_CHUNK_TOKENS = 6

    def _lexical_compute(self, db, tokens, where, args, pool, deadline_ms=None):
        def expression(chunk):
            return " OR ".join(
                '"' + token.replace('"', '""') + '"' for token in chunk
            )

        budget = (
            self.lexical_deadline_ms if deadline_ms is None else deadline_ms
        ) / 1000.0
        if budget <= 0 or len(tokens) <= self._LEXICAL_CHUNK_TOKENS:
            return self._lexical_query(db, expression(tokens), where, args, pool)
        deadline = perf_counter() + budget
        db.set_progress_handler(lambda: 1 if perf_counter() > deadline else 0, 2000)
        merged = {}
        chunks = [
            tokens[start:start + self._LEXICAL_CHUNK_TOKENS]
            for start in range(0, len(tokens), self._LEXICAL_CHUNK_TOKENS)
        ]
        try:
            for index, chunk in enumerate(chunks):
                try:
                    rows = self._lexical_query(
                        db, expression(chunk), where, args, pool
                    )
                except sqlite3.OperationalError as error:
                    if "interrupted" not in str(error).lower():
                        raise
                    # Soft deadline spent: keep the chunks that finished.
                    # Partial lexical rows beat discarding the whole stage
                    # (and with it, usually, the caller's dense stage too).
                    logger.info(
                        "Lexical stage soft deadline after %d/%d chunks; "
                        "returning %d partial rows",
                        index, len(chunks), len(merged),
                    )
                    break
                for row in rows:
                    current = merged.get(row["id"])
                    if current is None or row["fts_rank"] < current["fts_rank"]:
                        merged[row["id"]] = row
                if perf_counter() > deadline:
                    break
        finally:
            db.set_progress_handler(None, 0)
        return sorted(
            merged.values(), key=lambda row: (row["fts_rank"], row["id"])
        )[:pool]

    def _lexical_rows(self, db, tokens, where, args, pool, deadline_ms=None):
        memo = LEXICAL_MEMO.get()
        if memo is None:
            return self._lexical_compute(db, tokens, where, args, pool, deadline_ms)
        expression = " OR ".join(
            '"' + token.replace('"', '""') + '"' for token in tokens
        )
        key = (expression, where, tuple(args), pool)
        with memo["lock"]:
            entry = memo["entries"].get(key)
            if entry is None:
                entry = memo["entries"][key] = {"event": Event(), "rows": None}
                owner = True
            else:
                owner = False
        if not owner:
            # Another search in this turn is computing (or has computed)
            # the identical query; its rows are materialized and read-only
            # downstream, so they can be shared across connections.
            entry["event"].wait()
            if entry["rows"] is not None:
                return entry["rows"]
            return self._lexical_compute(db, tokens, where, args, pool, deadline_ms)
        try:
            entry["rows"] = self._lexical_compute(
                db, tokens, where, args, pool, deadline_ms
            )
            return entry["rows"]
        finally:
            entry["event"].set()

    def search(
        self, query, *, channel_ids, author=None, channel=None, after=None, before=None, limit=10,
        require_dense=False, infer_author=True, exclude_message_id=None, lexical_only=False,
        snapshot_only=False, allow_hybrid=True, route_budgets=None,
    ):
        # route_budgets optionally caps each retrieval route separately
        # ({"lexical_ms", "dense_ms", "aux_ms"}): a slow route may no
        # longer eat the caller's whole deadline, because its result is
        # discarded (dense/aux) or truncated (lexical chunks) once its
        # own budget is spent. None keeps the previous single-pool
        # behaviour apart from the lexical stage's own soft deadline.
        if lexical_only and require_dense:
            raise ValueError("lexical_only and require_dense are mutually exclusive")
        if limit < 1:
            return []
        channel_ids = tuple(sorted({str(c) for c in channel_ids}))
        if not channel_ids or (not query.strip() and author is None):
            return []
        pool = max(50, limit)
        with self.connect() as db:
            db.execute("BEGIN")
            if infer_author or author is not None:
                query, author = self._speaker_query(db, query, author, channel_ids)
            where, args = self._scope(
                channel_ids, author=author, channel=channel, after=after, before=before,
                exclude_message_id=exclude_message_id,
            )
            if require_dense and not db.execute(
                f"SELECT 1 FROM messages m WHERE {where} LIMIT 1", args
            ).fetchone():
                return []
            if not query:
                return [_hit(row) for row in db.execute(
                    f"SELECT m.* FROM messages m WHERE {where} "
                    "ORDER BY m.created_at DESC, m.id DESC LIMIT ?", [*args, limit]
                )]
            tokens = sorted(set(bigrams(query)))
            if not tokens:
                return []
            dense = None
            reason = self._dense_state
            attempted = False
            budgets = route_budgets or {}
            dense_started = None
            if snapshot_only and require_dense and self._dense is not None and not lexical_only:
                # A missing snapshot is the common failure for this call
                # shape; probe it before paying for the FTS scan whose
                # rows the DenseSnapshotUnavailable raise would discard.
                attempted = True
                reason = "no usable scoped vectors"
                dense_started = perf_counter()
                try:
                    dense = self._dense.search_snapshot(
                        db, query, channel_ids, where, args, pool, author_ids=author,
                        channel=channel, after=after, before=before,
                    )
                except Exception:
                    reason = "dense search failed"
                    logger.warning(
                        "Dense server log search unavailable; using lexical", exc_info=True
                    )
                else:
                    if dense is None:
                        raise DenseSnapshotUnavailable(
                            "No published dense snapshot candidates"
                        )
            with stage("lexical"):
                lexical = self._lexical_rows(
                    db, tokens, where, args, pool,
                    deadline_ms=budgets.get("lexical_ms"),
                )
            if self._dense is not None and not lexical_only and not attempted:
                reason = "no usable scoped vectors"
                dense_started = perf_counter()
                try:
                    search = self._dense.search_snapshot if snapshot_only else self._dense.search
                    dense = search(
                        db, query, channel_ids, where, args, pool, author_ids=author,
                        channel=channel, after=after, before=before,
                    )
                except Exception:
                    reason = "dense search failed"
                    logger.warning(
                        "Dense server log search unavailable; using lexical", exc_info=True
                    )
            dense_ms = budgets.get("dense_ms")
            dense_discarded = False
            if (
                dense is not None and dense_ms is not None
                and dense_started is not None
                and (perf_counter() - dense_started) * 1000 > dense_ms
            ):
                # The dense route overspent its own budget (the query
                # embedding under load is the usual cause). Its result
                # arrives too late to leave the caller room for fusion,
                # so the search degrades exactly as if dense were
                # unavailable instead of dragging the whole retrieval
                # past its deadline. Only callers that pass budgets
                # (pre_retrieve) ever take this branch.
                logger.info(
                    "Dense route exceeded its budget (%.0fms > %sms); discarding",
                    (perf_counter() - dense_started) * 1000, dense_ms,
                )
                dense = None
                dense_discarded = True
                reason = "dense budget spent"
            if snapshot_only and dense is None:
                # A budget-discarded dense result is not an unpublished
                # snapshot: the snapshot exists and the lexical stage
                # already ran, so degrade to the lexical hits instead
                # of reporting the whole search as unavailable. The
                # caller can then tell a thin result from a missing
                # snapshot and retry dense with a larger budget.
                if require_dense and not dense_discarded:
                    raise DenseSnapshotUnavailable("No published dense snapshot candidates")
                if not dense_discarded:
                    return []
            if (require_dense and not dense_discarded
                    and (not dense or getattr(dense, "coverage", None) == 0)):
                raise RuntimeError(f"Dense server log search unavailable: {reason}")
            if dense is None:
                if not lexical_only and self._lexical_reason != reason:
                    logger.warning("Server log search mode=lexical reason=%s", reason)
                    self._lexical_reason = reason
                return [_hit(row) for row in lexical[:limit]]
            self._lexical_reason = None
            logger.info("Server log search mode=dense partial=%s coverage=%s",
                        getattr(dense, "partial", False), getattr(dense, "coverage", None))
            extra_dense = None
            # The hybrid (aux) path only earns its place when person
            # grounding came up empty: measured on real questions, its
            # gains were all in grounding-miss cases, while on scoped
            # questions it dilutes the author-focused ranking. Callers
            # that know the grounding outcome pass allow_hybrid=False;
            # an author resolved here (explicit or inferred) counts as
            # scoped too.
            if allow_hybrid and author is None and self.hybrid_model and self._dense is not None:
                aux_ms = budgets.get("aux_ms")
                aux_started = perf_counter()
                try:
                    aux_search = (
                        self._dense.search_aux_snapshot if snapshot_only
                        else self._dense.search_aux
                    )
                    extra_dense = aux_search(
                        db, query, channel_ids, where, args, pool, author_ids=author,
                        channel=channel, after=after, before=before,
                    )
                except Exception:
                    logger.warning(
                        "Hybrid dense search unavailable; using two-path fusion",
                        exc_info=True,
                    )
                    extra_dense = None
                if (
                    extra_dense is not None and aux_ms is not None
                    and (perf_counter() - aux_started) * 1000 > aux_ms
                ):
                    logger.info(
                        "Hybrid route exceeded its budget (%.0fms > %sms); discarding",
                        (perf_counter() - aux_started) * 1000, aux_ms,
                    )
                    extra_dense = None
            ids = _fusion(query, lexical, dense, pool,
                          partial=getattr(dense, "partial", False),
                          dense_coverage=getattr(dense, "coverage", None),
                          extra_dense=extra_dense)[:limit]
            if not ids:
                return []
            # Reapply scope on the final materialization as well as both rankers.
            rows = db.execute(
                f"SELECT m.* FROM messages m WHERE {where} "
                f"AND m.id IN ({','.join('?' for _ in ids)})",
                [*args, *ids],
            ).fetchall()
            by_id = {row["id"]: _hit(row) for row in rows}
            return [by_id[item] for item in ids if item in by_id]

    def collect(self, *, channel_ids, query="", author=None, limit=24, offset=0,
                exclude_message_id=None):
        if not 1 <= limit <= 60 or not 0 <= offset <= 1000:
            raise ValueError("Collection limit or offset out of range")
        if not query.strip() and not author:
            raise ValueError("A topic or author is required")
        channels = tuple(sorted({str(value) for value in channel_ids}))
        with self.connect() as db:
            db.execute("BEGIN")
            ids = self._author_ids(db, author, channels, explicit=True) if author else None
            visible, visible_args = self._scope(channels, exclude_message_id=exclude_message_id)
            if ids is not None:
                ids = tuple(value for value in ids if db.execute(
                    f"SELECT 1 FROM messages m WHERE {visible} AND m.author_id = ? LIMIT 1",
                    [*visible_args, value],
                ).fetchone())
            if ids is not None and len(ids) > 1:
                return {"ambiguous_author_ids": list(ids), "hits": []}
            where, args = self._scope(channels, author=ids, exclude_message_id=exclude_message_id)
            where += " AND m.is_bot = 0"
            if query.strip():
                tokens = sorted(set(bigrams(query)))
                if not tokens:
                    return {"message_count": 0, "hits": []}
                expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
                where += " AND m.rowid IN (SELECT rowid FROM message_fts WHERE message_fts MATCH ?)"
                args.append(expression)
            stats = dict(db.execute(
                f"SELECT count(*) message_count, min(created_at) first_seen, "
                f"max(created_at) last_seen FROM messages m WHERE {where}", args,
            ).fetchone())
            stats["author_ids"] = list(ids) if ids is not None else []
            stats["channels"] = [dict(row) for row in db.execute(
                f"SELECT channel_id, max(channel_name) name, count(*) message_count "
                f"FROM messages m WHERE {where} GROUP BY channel_id "
                "ORDER BY count(*) DESC, channel_id",
                args,
            )]
            stats["channel_count"] = len(stats["channels"])
            stats["channels_truncated"] = stats["channel_count"] > 60
            stats["channels"] = stats["channels"][:60]
            rows = list(db.execute(f"""
                WITH matches AS (
                    SELECT m.id, m.channel_id, m.created_at, length(m.text) chars,
                           ntile(8) OVER (ORDER BY created_at, id) time_bin
                    FROM messages m WHERE {where}
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY channel_id, time_bin ORDER BY chars DESC, id
                    ) bucket_rank FROM matches
                ) SELECT m.*, r.time_bin FROM ranked r JOIN messages m ON m.id = r.id
                  WHERE r.bucket_rank = 1
            """, args))
            selected, seen_channels, seen_times = [], set(), set()
            while rows and len(selected) < offset + limit:
                row = max(rows, key=lambda row: (
                    (row["channel_id"] not in seen_channels) + (row["time_bin"] not in seen_times),
                    row["time_bin"] not in seen_times, len(row["text"]), row["id"],
                ))
                rows.remove(row)
                selected.append(row)
                seen_channels.add(row["channel_id"])
                seen_times.add(row["time_bin"])
            stats["next_offset"] = offset + limit if rows and offset + limit <= 1000 else None
            stats["sampling"] = "Longest post per channel/time octile; sample, not all posts"
            stats["hits"] = [_hit(row) for row in selected[offset:]]
            return stats

    def read(self, message_id, *, channel_ids, direction="around", limit=15):
        if direction not in {"around", "before", "after"}:
            raise ValueError("direction must be around, before or after")
        if limit < 1:
            return []
        where, args = self._scope(channel_ids)
        with self.connect() as db:
            db.execute("BEGIN")
            anchor = db.execute(
                f"SELECT m.* FROM messages m WHERE {where} AND m.id = ?",
                [*args, str(message_id)],
            ).fetchone()
            if anchor is None:
                return []

            def side(operator, order, count):
                return db.execute(
                    f"""
                    SELECT m.* FROM messages m WHERE {where} AND m.channel_id = ?
                    AND (m.created_at, m.id) {operator} (?, ?)
                    ORDER BY m.created_at {order}, m.id {order} LIMIT ?
                """,
                    [*args, anchor["channel_id"], anchor["created_at"], anchor["id"], count],
                ).fetchall()

            if direction == "before":
                rows = list(reversed(side("<", "DESC", limit)))
            elif direction == "after":
                rows = side(">", "ASC", limit)
            else:
                previous = side("<", "DESC", limit // 2)
                following = side(">", "ASC", limit - 1 - len(previous))
                if len(previous) + len(following) < limit - 1:
                    previous = side("<", "DESC", limit - 1 - len(following))
                rows = [*reversed(previous), anchor, *following]
            return [_hit(row) for row in rows]

    def recent(self, channel_id, limit):
        where, args = self._scope([channel_id])
        with self.connect() as db:
            rows = db.execute(
                f"SELECT m.* FROM messages m WHERE {where} "
                "ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
                [*args, max(0, limit)],
            ).fetchall()
            return [_hit(row) for row in rows]

    def get_cursor(self, channel_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT message_id FROM backfill_cursors WHERE channel_id = ?", (str(channel_id),)
            ).fetchone()
            return row[0] if row else None

    def set_cursor(self, channel_id, message_id):
        with self.connect() as db:
            self._set_cursor(db, channel_id, message_id)

    def embed_pending(self, *, batch_size=32, progress=None):
        return self._embed_batches(batch_size, progress, Event())

    def embed_pending_hybrid(self, *, batch_size=32, progress=None):
        """Embed pending messages under the hybrid model tag, if configured.

        Keeps the hybrid path's coverage from decaying as new messages
        arrive. Returns 0 when hybrid retrieval is not configured.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.hybrid_model or not (self.hybrid_onnx_dir or self._hybrid_embedder):
            return 0
        with self._embedding_lock:
            try:
                return self._hybrid_index().embed_pending(batch_size, progress, Event())
            except Exception:
                # The hybrid path is best-effort: its failure must never
                # take down primary embedding or search.
                logger.warning("Hybrid embedding pass unavailable", exc_info=True)
                return 0

    def _hybrid_index(self):
        if self._hybrid_dense is None:
            from types import SimpleNamespace

            from nezubot_discrod.server_log_dense import DenseIndex

            shim = SimpleNamespace(
                connect=self.connect,
                model=self.hybrid_model,
                static_model=None,
                max_length=self.max_length,
                embedding_onnx_dir=self.hybrid_onnx_dir,
                hybrid_model=None,
                hybrid_onnx_dir=None,
                _hybrid_embedder=None,
            )
            self._hybrid_dense = DenseIndex(shim, self._hybrid_embedder)
        return self._hybrid_dense

    def enable_dense(self):
        """Load the query backend synchronously without embedding pending messages."""
        with self._dense_init_lock:
            if self._dense is None:
                self._dense_state = "loading"
                logger.info("Loading dense server log backend model=%s max_length=%s",
                            self.model, self.max_length)
                try:
                    from nezubot_discrod.server_log_dense import DenseIndex

                    self._dense = DenseIndex(self, self._embedder)
                except Exception:
                    self._dense_state = "initialization failed"
                    raise
                self._dense_state = "ready"
                logger.info("Dense server log backend ready; vector cache loads on first search")
            return self._dense

    def _embed_batches(self, batch_size, progress, stop):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with self._embedding_lock:
            try:
                return self.enable_dense().embed_pending(batch_size, progress, stop)
            except ImportError:
                logger.warning("Dense server log dependencies unavailable; using lexical")
                return 0

    def start_embedder(self, *, batch_size=32, poll_interval=30):
        if batch_size < 1 or poll_interval <= 0:
            raise ValueError("batch_size and poll_interval must be positive")
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return self._worker
            self._stop.clear()

            def run():
                while not self._stop.is_set():
                    try:
                        self._embed_batches(batch_size, None, self._stop)
                    except Exception:
                        logger.warning("Server log embedder failed; using lexical", exc_info=True)
                    self._stop.wait(poll_interval)

            self._worker = Thread(target=run, name="server-log-embedder", daemon=True)
            self._worker.start()
            return self._worker

    def stop_embedder(self, *, timeout=None):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private Discord server log index")
    parser.add_argument("--db", type=Path, help="Override the private SQLite path")
    parser.add_argument("--guild-id", help="Guild ID for exports that omit guild_id")
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import")
    importer.add_argument("jsonl", type=Path)
    backfill = subparsers.add_parser("backfill-authors")
    backfill.add_argument("jsonl", type=Path)
    for command in ("export-vectors", "import-vectors"):
        subparsers.add_parser(command).add_argument("bundle", type=Path)
    subparsers.add_parser("embed")
    people = subparsers.add_parser("build-people")
    people.add_argument("--rebuild", action="store_true")
    users = subparsers.add_parser("sync-users")
    users.add_argument("--min-messages", type=int, default=50)
    search = subparsers.add_parser("search", help="Search with the bot's dense retrieval backend")
    search.add_argument("query")
    scope = search.add_mutually_exclusive_group(required=True)
    scope.add_argument("--channel-ids", nargs="+", help="Explicit caller-visible channel IDs")
    scope.add_argument("--all-channels", action="store_true",
                       help="Operator search of all archived channels, not Discord permissions")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--author")
    search.add_argument("--channel")
    search.add_argument("--after")
    search.add_argument("--before")
    search.add_argument("--lexical-only", action="store_true", help="Explicit lexical baseline")
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    store = ServerLog(args.db, guild_id=args.guild_id)
    if args.command == "search":
        if args.limit < 1:
            parser.error("--limit must be positive")
        logging.basicConfig(level=logging.INFO)
        try:
            if not args.lexical_only:
                store.enable_dense()
            channels = args.channel_ids
            if args.all_channels:
                with store.connect() as db:
                    channels = [row[0] for row in db.execute(
                        "SELECT DISTINCT channel_id FROM messages ORDER BY channel_id"
                    )]
                print("Scope: all archived channels; NOT live Discord caller permissions",
                      file=sys.stderr)
            hits = store.search(
                args.query, channel_ids=channels, author=args.author, channel=args.channel,
                after=args.after, before=args.before, limit=args.limit,
                require_dense=not args.lexical_only,
            )
        except Exception as exc:
            parser.exit(1, f"Search failed: {exc}. Install nezubot-discrod[rag] for dense "
                        "retrieval, or select --lexical-only explicitly.\n")
        print(re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]",
                     lambda match: f"\\x{ord(match[0]):02x}", format_hits(hits)))
    elif args.command in {"export-vectors", "import-vectors"}:
        from nezubot_discrod.server_log_vectors import transfer

        print(transfer(store, args.command, args.bundle))
    elif args.command == "build-people":
        from nezubot_discrod.server_log_people import build_people

        counts = build_people(
            store, rebuild=args.rebuild,
            progress=lambda n: print(f"Scanned {n} messages", flush=True),
        )
        print(json.dumps(counts, sort_keys=True))
    elif args.command == "sync-users":
        token = os.environ.get("DISCORD_TOKEN")
        if not token:
            parser.exit(1, "DISCORD_TOKEN is not set\n")
        from nezubot_discrod.server_log_people import discord_member_fetcher, sync_users

        result = sync_users(
            store, discord_member_fetcher(token), min_messages=args.min_messages,
            progress=lambda n: print(f"Synced {n} users", flush=True),
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command in {"import", "backfill-authors"}:
        with args.jsonl.open() as source:
            records = (json.loads(line) for line in source if line.strip())
            if args.command == "import":
                count = store.upsert_many(records)
                print(f"Imported {count} messages into {store.path}")
            else:
                count = store.backfill_authors(records)
                print(f"Processed {count} existing author records in {store.path}")
    else:
        count = store.embed_pending(progress=lambda n: print(f"Embedded {n} messages", flush=True))
        if store._dense is None:
            parser.exit(1, "Dense dependencies unavailable; install nezubot-discrod[rag]\n")
        print(f"Embedded {count} messages; index up to date")


if __name__ == "__main__":
    main()
