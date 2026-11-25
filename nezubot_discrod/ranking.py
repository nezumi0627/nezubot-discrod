"""Message-count ranking behind the /rank command.

The aggregation is a full scan of the archive (seconds of CPU, much
longer in wall time on the production database under load), so —
exactly like /status — the embed worker periodically writes the
result into a small snapshot table (``rank_stats``) and
collect_ranking reads that row instead. When no snapshot row exists
yet, collect_ranking falls back to the live aggregation.

Counting rules mirror status.py's notion of a surviving message:
``deleted = 0`` and no tombstone row in ``deleted_messages``. Bot
authors are excluded via the per-message ``is_bot`` flag (the users
table carries no bot flag; the flag on the message rows is the
reliable source, and the bot's own messages are flagged too).

The #1 detail block reports, for the top author only: the last
message time (JST), a per-day average (total count over the JST
calendar days from the first to the last message, minimum 1 day),
the longest message's character count, and the most active JST
hours of day (top 3).
"""

import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))

_NOT_DELETED = (
    "m.deleted = 0 AND NOT EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id)"
)
_HUMAN = f"{_NOT_DELETED} AND m.is_bot = 0"

_RANK_STATS_DDL = """
CREATE TABLE IF NOT EXISTS rank_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""


def _display_name(user_row, message_row, author_id):
    """Resolve a name: users nickname/display/username first, then
    the newest message's nickname/global/name snapshot, else the id."""
    for row in (user_row, message_row):
        if not row:
            continue
        for value in row:
            if value and value.strip():
                return value.strip()
    return author_id


def _aggregate_ranking(db, *, limit):
    """Live aggregation on an open connection. Returns the ranking
    dict collect_ranking produces, or None when nobody qualifies."""
    rows = db.execute(
        f"SELECT m.author_id, count(*) AS cnt FROM messages m "
        f"WHERE {_HUMAN} GROUP BY m.author_id ORDER BY cnt DESC, m.author_id "
        f"LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return None
    entries = []
    for author_id, count in rows:
        user_row = db.execute(
            "SELECT nickname, display_name, username FROM users WHERE user_id = ?",
            (author_id,),
        ).fetchone()
        message_row = db.execute(
            "SELECT author_nickname, author_global_name, author_name "
            "FROM messages WHERE author_id = ? ORDER BY created_at DESC LIMIT 1",
            (author_id,),
        ).fetchone()
        entries.append(
            {
                "author_id": author_id,
                "name": _display_name(user_row, message_row, author_id),
                "count": count,
            }
        )
    top = entries[0]
    count, first_at, last_at, max_chars = db.execute(
        f"SELECT count(*), min(m.created_at), max(m.created_at), "
        f"max(length(m.text)) FROM messages m "
        f"WHERE {_HUMAN} AND m.author_id = ?",
        (top["author_id"],),
    ).fetchone()
    hours = db.execute(
        f"SELECT CAST(strftime('%H', m.created_at, '+9 hours') AS INTEGER) AS hour, "
        f"count(*) AS cnt FROM messages m "
        f"WHERE {_HUMAN} AND m.author_id = ? "
        f"GROUP BY hour ORDER BY cnt DESC, hour LIMIT 3",
        (top["author_id"],),
    ).fetchall()
    first_dt = datetime.fromisoformat(first_at).astimezone(JST)
    last_dt = datetime.fromisoformat(last_at).astimezone(JST)
    days = max(1, (last_dt.date() - first_dt.date()).days)
    detail = {
        "author_id": top["author_id"],
        "name": top["name"],
        "count": count,
        "first_at": first_at,
        "last_at": last_at,
        "per_day": count / days,
        "max_chars": max_chars or 0,
        "top_hours": [hour for hour, _cnt in hours],
    }
    return {"entries": entries, "detail": detail}


def _collect_ranking_live(db_path, *, limit):
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("PRAGMA cache_size = -64000")
        return _aggregate_ranking(db, limit=limit)


def _read_ranking_snapshot(db_path, *, limit):
    """The worker-maintained ranking row, or None when never written
    or when it holds fewer entries than requested."""
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
            db.execute("PRAGMA query_only = ON")
            row = db.execute(
                "SELECT payload FROM rank_stats WHERE id = 1"
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        ranking = json.loads(row[0])
    except (ValueError, TypeError):
        return None
    if not isinstance(ranking, dict) or len(ranking.get("entries", ())) < limit:
        return None
    return ranking


def collect_ranking(db_path, *, limit=5):
    """Message-count ranking for the archive at db_path.

    Prefers the maintained snapshot row; falls back to the live full
    aggregation when the snapshot has never been written. Returns
    None when no human messages survive in the archive.
    """
    db_path = Path(db_path)
    snapshot = _read_ranking_snapshot(db_path, limit=limit)
    if snapshot is not None:
        return snapshot
    return _collect_ranking_live(db_path, limit=limit)


def write_ranking_snapshot(db_path, *, limit=5):
    """Run the live aggregation and store it as the snapshot row.

    Called periodically by the embed worker, whose passes already
    walk the archive; the /rank path then never pays the full scan.
    """
    db_path = Path(db_path)
    ranking = _collect_ranking_live(db_path, limit=limit)
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(_RANK_STATS_DDL)
        db.execute(
            "INSERT OR REPLACE INTO rank_stats VALUES (1, ?, ?)",
            (json.dumps(ranking, ensure_ascii=False), time.time()),
        )
        db.commit()
    return ranking


def _format_jst(iso_text):
    dt = datetime.fromisoformat(iso_text).astimezone(JST)
    return f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d}"


def _format_jst_date(iso_text):
    dt = datetime.fromisoformat(iso_text).astimezone(JST)
    return f"{dt.year}年{dt.month}月{dt.day}日"


def build_rank_embed(ranking):
    """Render the ranking dict as the /rank embed."""
    import discord

    embed = discord.Embed(
        title="📊 メッセージ数ランキング（全期間）",
        color=discord.Color.blurple(),
    )
    if ranking is None:
        embed.description = "まだ集計できるメッセージがありません。"
        return embed
    medals = ("🥇", "🥈", "🥉")
    lines = []
    for index, entry in enumerate(ranking["entries"]):
        mark = medals[index] if index < len(medals) else f"{index + 1}."
        lines.append(f"{mark} **{entry['name']}** — {entry['count']:,} 件")
    embed.add_field(name="トップ5", value="\n".join(lines), inline=False)
    detail = ranking["detail"]
    hours_text = "・".join(f"{hour}時台" for hour in detail["top_hours"])
    detail_lines = [
        f"最終メッセージ: {_format_jst(detail['last_at'])}",
        f"1日平均: {detail['per_day']:.1f} 件"
        f"（{_format_jst_date(detail['first_at'])} 〜"
        f" {_format_jst_date(detail['last_at'])}）",
        f"最長メッセージ: {detail['max_chars']:,} 文字",
        f"アクティブな時間帯: {hours_text}",
    ]
    embed.add_field(
        name=f"🥇 1位の詳細: {detail['name']}",
        value="\n".join(detail_lines),
        inline=False,
    )
    return embed
