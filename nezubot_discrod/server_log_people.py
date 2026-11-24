"""Person graph for the server log archive (Phase A: data layer only).

Builds two derived tables inside the server log database:

- ``message_people``: one row per (message, person, relation) where relation
  is ``author`` (the message author), ``mention`` (a ``<@id>`` / ``<@!id>``
  markup in the stored message text), or ``reply`` (the author of the
  message this one replies to, resolved through ``messages.reply_to``).
  Everything is derived from data already stored in ``messages``; a reply
  whose target message is not archived produces no row.
- ``users``: profile snapshots for authors with at least ``min_messages``
  archived messages, refreshed from the Discord guild-member API. Fields
  that cannot be fetched stay empty; nothing is guessed.

The batch build is idempotent and chunked by ``messages.rowid`` so it can
run alongside the live bot without holding long write transactions.
"""

import json
import re
import time
import urllib.error
import urllib.request

MENTION_PATTERN = re.compile(r"<@!?(\d+)>")

PEOPLE_DDL = """
CREATE TABLE IF NOT EXISTS message_people (
    message_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (message_id, user_id, relation)
);
CREATE INDEX IF NOT EXISTS message_people_user_relation
    ON message_people (user_id, relation, message_id);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    display_name TEXT NOT NULL DEFAULT '',
    nickname TEXT NOT NULL DEFAULT '',
    avatar_url TEXT NOT NULL DEFAULT '',
    joined_at TEXT NOT NULL DEFAULT '',
    synced_at TEXT NOT NULL DEFAULT ''
);
"""


def ensure_people_schema(db):
    db.executescript(PEOPLE_DDL)


def mentioned_user_ids(text):
    """User ids referenced by Discord mention markup in stored text."""
    return set(MENTION_PATTERN.findall(text or ""))


def people_rows(message_id, author_id, text, reply_author_id):
    """(message_id, user_id, relation, confidence) rows for one message."""
    rows = [(message_id, author_id, "author", 1.0)]
    for user_id in sorted(mentioned_user_ids(text)):
        rows.append((message_id, user_id, "mention", 1.0))
    if reply_author_id:
        rows.append((message_id, reply_author_id, "reply", 1.0))
    return rows


def sync_message_people(db, row, reply_author_id):
    """Refresh one message's people rows inside an open transaction.

    The caller owns schema creation (ServerLog connection setup or the
    batch entry points call ensure_people_schema first).
    """
    db.execute("DELETE FROM message_people WHERE message_id = ?", (row["id"],))
    if row["deleted"]:
        return
    db.executemany(
        "INSERT OR IGNORE INTO message_people VALUES (?, ?, ?, ?)",
        people_rows(row["id"], row["author_id"], row["text"], reply_author_id),
    )


def drop_message_people(db, message_id):
    db.execute("DELETE FROM message_people WHERE message_id = ?", (message_id,))


def _reply_author(db, reply_to):
    if not reply_to:
        return None
    found = db.execute(
        "SELECT author_id FROM messages WHERE id = ?", (reply_to,)
    ).fetchone()
    return found[0] if found else None


def build_people(store, *, chunk_rows=5000, pause=0.05, rebuild=False, progress=None):
    """(Re)build message_people from messages in short transactions.

    Returns per-relation row counts for the whole table afterwards.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    with store.connect() as db:
        ensure_people_schema(db)
    if rebuild:
        while True:
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "DELETE FROM message_people WHERE rowid IN "
                    "(SELECT rowid FROM message_people LIMIT 100000)"
                )
                removed = db.execute("SELECT changes()").fetchone()[0]
            if not removed:
                break
            if pause:
                time.sleep(pause)
    with store.connect() as db:
        bounds = db.execute("SELECT min(rowid), max(rowid) FROM messages").fetchone()
    low, high = bounds[0] or 0, bounds[1] or 0
    scanned = 0
    start = low
    while start <= high:
        end = start + chunk_rows
        with store.connect() as db:
            rows = db.execute(
                "SELECT id, author_id, text FROM messages "
                "WHERE rowid >= ? AND rowid < ?",
                (start, end),
            ).fetchall()
            db.execute("BEGIN IMMEDIATE")
            values = []
            for row in rows:
                values.extend(
                    people_rows(row["id"], row["author_id"], row["text"], None)
                )
            db.executemany(
                "INSERT OR IGNORE INTO message_people VALUES (?, ?, ?, ?)", values
            )
            db.execute(
                """
                INSERT OR IGNORE INTO message_people
                SELECT m.id, r.author_id, 'reply', 1.0
                FROM messages m JOIN messages r ON r.id = m.reply_to
                WHERE m.rowid >= ? AND m.rowid < ? AND m.reply_to IS NOT NULL
                """,
                (start, end),
            )
        scanned += len(rows)
        if progress is not None:
            progress(scanned)
        start = end
        if pause:
            time.sleep(pause)
    with store.connect() as db:
        counts = dict(
            db.execute(
                "SELECT relation, count(*) FROM message_people GROUP BY relation"
            ).fetchall()
        )
    return {"scanned": scanned, **counts}


def user_targets(store, min_messages):
    """Authors with >= min_messages archived messages, busiest first.

    Each row carries the author fields from that author's most recent
    message in the guild, used as a fallback when the live profile
    cannot be fetched.
    """
    with store.connect() as db:
        ensure_people_schema(db)
        # Group by author alone: grouping by (author_id, guild_id) defeats
        # the author index and forces a full scan + sort of the archive.
        # The guild and fallback fields come from one stored row per
        # author below (an unordered LIMIT 1 rides the author index).
        groups = db.execute(
            """
            SELECT author_id, count(*) AS total FROM messages
            GROUP BY author_id HAVING total >= ?
            ORDER BY total DESC
            """,
            (min_messages,),
        ).fetchall()
        targets = []
        for group in groups:
            latest = db.execute(
                """
                SELECT guild_id, author_username, author_global_name,
                       author_nickname
                FROM messages WHERE author_id = ?
                LIMIT 1
                """,
                (group["author_id"],),
            ).fetchone()
            if latest is None or latest["guild_id"] == "@me":
                continue
            targets.append({
                "author_id": group["author_id"],
                "guild_id": latest["guild_id"],
                "total": group["total"],
                "author_username": latest["author_username"] or "",
                "author_global_name": latest["author_global_name"] or "",
                "author_nickname": latest["author_nickname"] or "",
            })
        return targets


def sync_users(store, fetch_member, *, min_messages=50, pause=0.35,
               now=None, progress=None):
    """Refresh the users table for authors with >= min_messages messages.

    ``fetch_member(guild_id, user_id)`` returns a Discord guild-member
    object or None when the profile cannot be fetched (left the guild,
    API error); stored author fields then fill what they can.
    """
    if now is None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    targets = user_targets(store, min_messages)
    synced = fetched = 0
    for target in targets:
        member = fetch_member(target["guild_id"], target["author_id"])
        if member is not None:
            fetched += 1
            user = member.get("user") or {}
            username = user.get("username") or ""
            display_name = user.get("global_name") or username
            nickname = member.get("nick") or ""
            avatar = user.get("avatar")
            avatar_url = (
                f"https://cdn.discord.com/avatars/{target['author_id']}/{avatar}.png?size=256"
                if avatar else ""
            )
            joined_at = member.get("joined_at") or ""
        else:
            username = target["author_username"] or ""
            display_name = target["author_global_name"] or username
            nickname = target["author_nickname"] or ""
            avatar_url = ""
            joined_at = ""
        with store.connect() as db:
            ensure_people_schema(db)
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    guild_id = excluded.guild_id, username = excluded.username,
                    display_name = excluded.display_name, nickname = excluded.nickname,
                    avatar_url = excluded.avatar_url, joined_at = excluded.joined_at,
                    synced_at = excluded.synced_at
                """,
                (
                    target["author_id"], target["guild_id"], username, display_name,
                    nickname, avatar_url, joined_at, now,
                ),
            )
        synced += 1
        if progress is not None:
            progress(synced)
        if pause:
            time.sleep(pause)
    return {"targets": len(targets), "synced": synced, "fetched": fetched}


def discord_member_fetcher(token, *, api_base="https://discord.com/api/v10",
                           timeout=15, retries=3):
    """Build a fetch_member(guild_id, user_id) callable backed by REST.

    Returns the guild-member object, or None when the member cannot be
    fetched (left the guild, missing access, repeated API errors).
    Honors 429 Retry-After up to ``retries`` attempts.
    """
    def fetch(guild_id, user_id):
        request = urllib.request.Request(
            f"{api_base}/guilds/{guild_id}/members/{user_id}",
            headers={
                "Authorization": f"Bot {token}",
                "User-Agent": "nezubot-discrod server-log people sync",
            },
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 429 and attempt < retries:
                    try:
                        retry_after = float(
                            json.loads(error.read().decode("utf-8"))["retry_after"]
                        )
                    except Exception:
                        retry_after = 1.0
                    time.sleep(min(retry_after, 30.0))
                    continue
                return None
            except (urllib.error.URLError, OSError, ValueError):
                return None
        return None

    return fetch
