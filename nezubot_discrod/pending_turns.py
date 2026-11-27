"""Unanswered-turn recovery: accepted questions survive a restart.

The per-channel turn queue (v1.27.0) and the running turn live only in
the bot process's memory, so a VM replacement or a plain restart used
to silently discard every question that had been accepted but not yet
answered — and a question posted while the process was down never even
reached ``on_message`` (gateway events are not replayed; the archive
backfill records the message, but nothing re-triggers it). Verified on
2026-11-27: two owner questions posted during a deploy-restart gap
were archived by the backfill and never answered.

This module is the durable half of the fix. It owns one table in the
server log DB, ``pending_turns``:

- a turn is registered the moment it is accepted (it passed every gate
  and either took the channel's turn slot or entered its FIFO queue);
- when the answer has been delivered, the row is marked ``answered``
  together with the sent messages' ids;
- at startup the recovery pass in bot.py revisits rows still pending:
  it cross-checks the channel history (an answer posted just before a
  crash, after which the marking never ran, is adopted instead of
  duplicated), drops rows a re-ask already covered — in either
  direction: a later answered re-ask covers the earlier copy, and an
  earlier answered asking covers the re-ask (v1.42.1; checking only
  the later direction let a re-ask be answered twice across two
  recovery passes) — expires rows older than the recovery window,
  and re-runs the rest through the ordinary trigger path, so every
  live gate (blacklist, mute, policy) applies to recovered answers
  exactly as to live ones;
- the same pass also sweeps the archive for trigger-shaped messages
  that arrived while no bot process was running at all (the shape
  actually lost on 2026-11-27) and puts them through the same checks.

Registration and marking are best-effort at their call sites: a DB
hiccup must never break a live turn. The history cross-check is the
backstop for a marking that never ran.
"""

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

PENDING_TURNS_DDL = """
CREATE TABLE IF NOT EXISTS pending_turns (
    message_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    guild_id TEXT,
    author_id TEXT NOT NULL,
    content TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    origin TEXT NOT NULL DEFAULT 'live',
    answer_message_ids TEXT,
    answered_at TEXT,
    updated_at TEXT NOT NULL
)
"""

# States a row can still be recovered from. "recovering" rows were
# claimed by a recovery pass that itself died before resolving them.
RECOVERABLE_STATES = ("pending", "recovering")


def _now():
    return datetime.now(UTC).isoformat()


def ensure_pending_turns_schema(db):
    db.execute(PENDING_TURNS_DDL)


def init_pending_turns(store):
    """Create the pending-turns table if it does not exist yet.

    Safe to call on every startup.
    """
    with store.connect() as db:
        ensure_pending_turns_schema(db)


def normalize_content(text):
    """Whitespace-collapsed form used for duplicate comparison.

    Two turns count as the same question when author, channel and this
    normalized text all match.
    """
    return " ".join((text or "").split())


def register_turn(
    store,
    *,
    message_id,
    channel_id,
    guild_id,
    author_id,
    content,
    origin="live",
):
    """Record an accepted turn as unanswered. Returns True when the
    row was newly inserted (False when a row already exists — e.g. a
    recovery pass re-running a turn it registered itself)."""
    now = _now()
    with store.connect() as db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO pending_turns"
            " (message_id, channel_id, guild_id, author_id, content,"
            "  accepted_at, state, origin, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                str(message_id),
                str(channel_id),
                str(guild_id) if guild_id is not None else None,
                str(author_id),
                content,
                now,
                origin,
                now,
            ),
        )
        return cursor.rowcount > 0


def get_turn(store, message_id):
    with store.connect() as db:
        row = db.execute(
            "SELECT message_id, channel_id, guild_id, author_id, content,"
            " accepted_at, state, origin, answer_message_ids, answered_at"
            " FROM pending_turns WHERE message_id = ?",
            (str(message_id),),
        ).fetchone()
    if row is None:
        return None
    return {
        "message_id": row[0],
        "channel_id": row[1],
        "guild_id": row[2],
        "author_id": row[3],
        "content": row[4],
        "accepted_at": row[5],
        "state": row[6],
        "origin": row[7],
        "answer_message_ids": row[8],
        "answered_at": row[9],
    }


def mark_answered(store, message_id, answer_message_ids):
    """Mark a turn answered, recording the sent messages' ids.

    Only pending/recovering rows move; a row already resolved another
    way (superseded, expired, dropped) is left alone. Returns True
    when the row changed.
    """
    with store.connect() as db:
        cursor = db.execute(
            "UPDATE pending_turns SET state = 'answered',"
            " answer_message_ids = ?, answered_at = ?, updated_at = ?"
            " WHERE message_id = ? AND state IN ('pending', 'recovering')",
            (
                json.dumps([str(value) for value in answer_message_ids]),
                _now(),
                _now(),
                str(message_id),
            ),
        )
        return cursor.rowcount > 0


def trigger_for_answer(store, answer_message_id):
    """The trigger message id an answered bot message belongs to, or
    None when no answered row claims it.

    The reply anchor (v1.43.0) resolves a follow-up that replies to
    one of the bot's own answers through the exchange that produced
    the answer; the bot's answers are plain channel sends (no Discord
    reference back to the question), so this durable record is the
    exact route from an answer to its question. Rows are few (one per
    answered turn), so the JSON membership is confirmed by parsing
    after a quoted-id LIKE pre-filter."""
    wanted = str(answer_message_id)
    with store.connect() as db:
        rows = db.execute(
            "SELECT message_id, answer_message_ids FROM pending_turns"
            " WHERE state = 'answered' AND answer_message_ids LIKE ?",
            (f'%"{wanted}"%',),
        ).fetchall()
    for message_id, payload in rows:
        try:
            ids = [str(value) for value in json.loads(payload)]
        except (TypeError, ValueError):
            continue
        if wanted in ids:
            return str(message_id)
    return None


def set_turn_state(store, message_id, state):
    """Move a pending/recovering row to a terminal state (superseded /
    expired / dropped). Returns True when the row changed."""
    with store.connect() as db:
        cursor = db.execute(
            "UPDATE pending_turns SET state = ?, updated_at = ?"
            " WHERE message_id = ? AND state IN ('pending', 'recovering')",
            (state, _now(), str(message_id)),
        )
        return cursor.rowcount > 0


def claim_turn(store, message_id):
    """Claim a recoverable row for this recovery pass, so a row whose
    live marking lands mid-recovery is never processed twice. Returns
    True when the row was in a recoverable state."""
    with store.connect() as db:
        cursor = db.execute(
            "UPDATE pending_turns SET state = 'recovering', updated_at = ?"
            " WHERE message_id = ? AND state IN ('pending', 'recovering')",
            (_now(), str(message_id)),
        )
        return cursor.rowcount > 0


def list_recoverable(store, *, since_iso, limit=100):
    """Recoverable rows accepted at or after ``since_iso``, in arrival
    order (snowflake order is chronological)."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT message_id, channel_id, guild_id, author_id, content,"
            " accepted_at, state, origin FROM pending_turns"
            " WHERE state IN ('pending', 'recovering')"
            " AND accepted_at >= ?"
            " ORDER BY CAST(message_id AS INTEGER) ASC LIMIT ?",
            (since_iso, int(limit)),
        ).fetchall()
    return [
        {
            "message_id": row[0],
            "channel_id": row[1],
            "guild_id": row[2],
            "author_id": row[3],
            "content": row[4],
            "accepted_at": row[5],
            "state": row[6],
            "origin": row[7],
        }
        for row in rows
    ]


def expire_turns_before(store, cutoff_iso):
    """Expire recoverable rows older than the recovery window: the
    conversation has moved on and a late answer would be noise.
    Returns how many rows were expired."""
    with store.connect() as db:
        cursor = db.execute(
            "UPDATE pending_turns SET state = 'expired', updated_at = ?"
            " WHERE state IN ('pending', 'recovering')"
            " AND accepted_at < ?",
            (_now(), cutoff_iso),
        )
        return cursor.rowcount


def claimed_answer_ids(store, channel_id):
    """Answer message ids already attributed to answered turns in a
    channel. The recovery cross-check excludes these when looking for
    an unclaimed answer after a trigger: they belong to another turn.
    """
    claimed = set()
    with store.connect() as db:
        rows = db.execute(
            "SELECT answer_message_ids FROM pending_turns"
            " WHERE channel_id = ? AND state = 'answered'"
            " AND answer_message_ids IS NOT NULL",
            (str(channel_id),),
        ).fetchall()
    for (payload,) in rows:
        try:
            claimed.update(str(value) for value in json.loads(payload))
        except (TypeError, ValueError):
            continue
    return claimed


def next_trigger_id(store, channel_id, message_id):
    """The smallest registered trigger id after ``message_id`` in the
    channel, or None. Bot messages after that boundary belong to a
    later turn, not to this one."""
    with store.connect() as db:
        row = db.execute(
            "SELECT MIN(CAST(message_id AS INTEGER)) FROM pending_turns"
            " WHERE channel_id = ? AND CAST(message_id AS INTEGER) > ?",
            (str(channel_id), int(message_id)),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def has_answered_duplicate(store, *, channel_id, author_id, content, message_id):
    """True when a LATER turn in the same channel, by the same author,
    with the same normalized content, is already answered — the member
    re-asked the question and got the answer, so the earlier unanswered
    copy must not be answered again."""
    target = normalize_content(content)
    if not target:
        return False
    with store.connect() as db:
        rows = db.execute(
            "SELECT message_id, content FROM pending_turns"
            " WHERE channel_id = ? AND author_id = ? AND state = 'answered'"
            " AND CAST(message_id AS INTEGER) > ?",
            (str(channel_id), str(author_id), int(message_id)),
        ).fetchall()
    return any(
        normalize_content(row_content) == target
        for _row_id, row_content in rows
    )


DISCORD_EPOCH_MS = 1420070400000


def snowflake_time_ms(message_id):
    """Creation time embedded in a Discord snowflake id, in ms."""
    return (int(message_id) >> 22) + DISCORD_EPOCH_MS


def has_answered_predecessor(
    store, *, channel_id, author_id, content, message_id, max_gap_ms=None
):
    """True when an EARLIER turn in the same channel, by the same
    author, with the same normalized content, is already answered —
    this candidate is the re-ask, and the answer to the first asking
    already covers it, so it must not be answered again.

    The mirror image of has_answered_duplicate, and the check whose
    absence produced the 2026-11-27 double answer: the first asking
    was answered by a startup sweep (its answer duly claimed by its
    row, so the history cross-check correctly refuses to adopt it for
    the re-ask), the re-ask itself held no durable supersede verdict,
    and a later process that found or was handed a pending row for
    the re-ask had no rule left to stop it. ``max_gap_ms`` bounds how
    far apart the two askings may be (snowflake time) and still count
    as one question; a same-text question asked much later is a new
    question, not a re-ask.
    """
    target = normalize_content(content)
    if not target:
        return False
    with store.connect() as db:
        rows = db.execute(
            "SELECT message_id, content FROM pending_turns"
            " WHERE channel_id = ? AND author_id = ? AND state = 'answered'"
            " AND CAST(message_id AS INTEGER) < ?",
            (str(channel_id), str(author_id), int(message_id)),
        ).fetchall()
    candidate_ms = snowflake_time_ms(message_id)
    for row_id, row_content in rows:
        if normalize_content(row_content) != target:
            continue
        if (
            max_gap_ms is not None
            and candidate_ms - snowflake_time_ms(row_id) > max_gap_ms
        ):
            continue
        return True
    return False


def sweep_candidates(store, *, bot_id, since_iso, limit=100):
    """Archived messages that look like calls to the bot (a mention of
    the bot, or a reply to one of the bot's messages) at or after
    ``since_iso``, in arrival order.

    These are candidates only: whether each one was actually answered,
    already registered, gated out, or never seen because the process
    was down is decided by the recovery pass, not here.
    """
    bot_id = str(bot_id)
    with store.connect() as db:
        rows = db.execute(
            "SELECT id, channel_id, guild_id, author_id, created_at, text"
            " FROM messages"
            " WHERE is_bot = 0 AND deleted = 0 AND created_at >= ?"
            " AND (text LIKE ? OR text LIKE ?"
            "      OR reply_to IN (SELECT id FROM messages"
            "                      WHERE author_id = ?))"
            " ORDER BY CAST(id AS INTEGER) ASC LIMIT ?",
            (
                since_iso,
                f"%<@{bot_id}>%",
                f"%<@!{bot_id}>%",
                bot_id,
                int(limit),
            ),
        ).fetchall()
    return [
        {
            "message_id": row[0],
            "channel_id": row[1],
            "guild_id": row[2],
            "author_id": row[3],
            "created_at": row[4],
            "text": row[5],
        }
        for row in rows
    ]
