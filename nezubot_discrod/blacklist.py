"""Owner-managed blacklist: listed members get no service at all.

The list lives in the server log DB (table ``user_blacklist``), never
in code: the owner adds, removes and reviews entries at runtime with
the /blacklist slash command, and every enforcement point (slash
commands, n! prefix commands, mention/reply answer generation) reads
the same table through this module. Nothing else about a listed
member changes — their messages are still archived like everyone
else's and the wiki / appellation data is untouched; the list only
gates responses TO that member.

Lookups go through a small per-store cache (30 s TTL) so a check on
the message path costs no query in the common case; add/remove
invalidate the cache entry immediately, so a change made with
/blacklist takes effect on the very next interaction.

The one deploy-time seed below is data, not the list itself: it is
inserted only when the table is first created. If the owner later
removes that member, the removal sticks — the seed never runs again
because the table (and its history) already exists.
"""

import logging
import time
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Members registered when the blacklist table is first created
# (owner's instruction at feature launch, 2026-11-25). Managed with
# /blacklist afterwards like any other entry.
INITIAL_BLACKLIST_IDS = ("123456789012345678",)

BLACKLIST_DDL = """
CREATE TABLE IF NOT EXISTS user_blacklist (
    user_id TEXT PRIMARY KEY,
    added_by TEXT,
    created_at TEXT NOT NULL,
    note TEXT
)
"""

CACHE_TTL_SECONDS = 30.0

# store path -> (frozenset of user_id strings, monotonic expiry)
_cache = {}


def ensure_blacklist_schema(db):
    db.execute(BLACKLIST_DDL)


def _now():
    return datetime.now(UTC).isoformat()


def init_blacklist(store):
    """Create the table and insert the initial seed, once.

    The seed runs only when the table does not exist yet, so an
    owner's later removal of a seeded member is never undone by a
    restart. Safe to call on every startup.
    """
    with store.connect() as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type = 'table' AND name = 'user_blacklist'"
        ).fetchone()
        if exists:
            return
        db.execute("BEGIN IMMEDIATE")
        ensure_blacklist_schema(db)
        for user_id in INITIAL_BLACKLIST_IDS:
            db.execute(
                "INSERT OR IGNORE INTO user_blacklist"
                " (user_id, added_by, created_at, note)"
                " VALUES (?, NULL, ?, ?)",
                (user_id, _now(), "初期登録"),
            )
        logger.info(
            "Blacklist initialised with %d seed entries",
            len(INITIAL_BLACKLIST_IDS),
        )


def _invalidate(store):
    _cache.pop(str(store.path), None)


def _cached_ids(store):
    key = str(store.path)
    now = time.monotonic()
    entry = _cache.get(key)
    if entry is not None and now < entry[1]:
        return entry[0]
    with store.connect() as db:
        ensure_blacklist_schema(db)
        ids = frozenset(
            row[0] for row in db.execute("SELECT user_id FROM user_blacklist")
        )
    _cache[key] = (ids, now + CACHE_TTL_SECONDS)
    return ids


def is_blacklisted(store, user_id):
    """True when the member is on the blacklist (cached lookup)."""
    if user_id is None:
        return False
    return str(user_id) in _cached_ids(store)


def add_to_blacklist(store, user_id, *, added_by=None, note=None):
    """Insert a member. Returns True when newly added."""
    user_id = str(user_id)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_blacklist_schema(db)
        cursor = db.execute(
            "INSERT OR IGNORE INTO user_blacklist"
            " (user_id, added_by, created_at, note) VALUES (?, ?, ?, ?)",
            (
                user_id,
                str(added_by) if added_by is not None else None,
                _now(),
                note,
            ),
        )
        added = cursor.rowcount > 0
    _invalidate(store)
    return added


def remove_from_blacklist(store, user_id):
    """Delete a member's row. Returns True if one existed."""
    user_id = str(user_id)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_blacklist_schema(db)
        cursor = db.execute(
            "DELETE FROM user_blacklist WHERE user_id = ?", (user_id,)
        )
        removed = cursor.rowcount > 0
    _invalidate(store)
    return removed


def list_blacklist(store):
    """All entries as (user_id, added_by, created_at, note), oldest first."""
    with store.connect() as db:
        ensure_blacklist_schema(db)
        return db.execute(
            "SELECT user_id, added_by, created_at, note"
            " FROM user_blacklist ORDER BY created_at, user_id"
        ).fetchall()
