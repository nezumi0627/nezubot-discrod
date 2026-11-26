"""The /me understanding meter and its priority deepening queue.

A member can ask how well the bot understands them (/me). The score
is a composite of measurable coverage over the archive — how the
person is called (the appellation graph), how much of their speech
is embedded, how complete their identity record is, whether Server Wiki
has pages about them, and how present they are in the person graph.
Nothing is estimated or padded: every point traces to rows.

Using /me also registers the member in a priority tier. A job is
queued for them, and the embed worker — in its spare cycles, never
ahead of answering or embedding — deepens exactly that person:
catching the appellation miner up, rebuilding their alias rows from
the name snapshots stored on their messages, and re-measuring. The
worker also re-queues tier members whose last deepening has gone
stale, so tier members keep being refreshed more often than anyone
else. The next /me then shows a score that moved because the data
under it actually grew.
"""

import logging
import sqlite3
from datetime import UTC, datetime, timedelta

from nezubot_discrod.appellations import (
    mine_appellations,
    qualified_appellations,
    representative_appellation,
)
from nezubot_discrod.server_log import _name

logger = logging.getLogger(__name__)

ME_DDL = """
CREATE TABLE IF NOT EXISTS me_users (
    user_id TEXT PRIMARY KEY,
    first_used_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 1,
    last_score REAL,
    last_processed_at TEXT
)
"""

JOBS_DDL = """
CREATE TABLE IF NOT EXISTS understanding_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    processed_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    detail TEXT NOT NULL DEFAULT ''
)
"""

JOBS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS understanding_jobs_pending
    ON understanding_jobs(status, id)
"""

# Score weights (sum 100). Appellations dominate: being resolvable
# by the names people actually use is what "understanding a person"
# means for this bot.
WEIGHTS = {
    "appellations": 30,
    "embedded": 25,
    "identity": 20,
    "server_wiki": 15,
    "person_graph": 10,
}


def ensure_me_schema(db):
    db.execute(ME_DDL)
    db.execute(JOBS_DDL)
    db.execute(JOBS_INDEX_DDL)


def _now():
    return datetime.now(UTC).isoformat()


def _clamp(value):
    return max(0.0, min(1.0, value))


def collect_understanding(store, user_id, *, server_wiki_path=None):
    """The measured coverage report for one person. Read-only."""
    user_id = str(user_id)
    with store.connect() as db:
        total_messages = db.execute(
            "SELECT count(*) FROM messages WHERE author_id = ? AND deleted = 0",
            (user_id,),
        ).fetchone()[0]
        embedded = db.execute(
            "SELECT count(*) FROM message_vectors mv "
            "JOIN messages m ON m.id = mv.message_id "
            "WHERE m.author_id = ? AND mv.model = ?",
            (user_id, store.model),
        ).fetchone()[0]
        try:
            appellation_rows = db.execute(
                "SELECT target_id, caller_id, appellation, count "
                "FROM appellations WHERE target_id = ?",
                (user_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            appellation_rows = []
        alias_count = db.execute(
            "SELECT count(DISTINCT normalized) FROM author_aliases "
            "WHERE author_id = ?",
            (user_id,),
        ).fetchone()[0]
        profile = db.execute(
            "SELECT username, display_name, nickname FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        graph_links = db.execute(
            "SELECT count(*) FROM message_people WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        tier_row = None
        try:
            tier_row = db.execute(
                "SELECT use_count, last_score, last_processed_at "
                "FROM me_users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            tier_row = None

    # Only established appellations count: raw mined rows are mostly
    # one-off sentence fragments (and, before the 2026-11-25 repair,
    # bot boilerplate), and counting them raw reported 1,033 "names"
    # for one member and pinned this score at full marks for anyone
    # with a busy reply history. The saturation points are set from
    # the measured post-repair magnitudes of well-known members
    # (tens of established uses, around ten distinct callers), so a
    # full score has to be earned across all three components.
    qualified = qualified_appellations(appellation_rows)
    evidence = sum(total for total, _callers in qualified.values())
    callers = len({
        str(row[1])
        for row in appellation_rows if row[2] in qualified
    })
    names = len(qualified)
    top = representative_appellation(qualified)
    appellation_score = _clamp(
        0.4 * min(1.0, evidence / 60.0)
        + 0.3 * min(1.0, callers / 10.0)
        + 0.3 * (1.0 if top else 0.0)
    )
    embedded_score = _clamp(embedded / total_messages) if total_messages else 0.0
    profile_fields = 0
    if profile is not None:
        profile_fields = sum(1 for value in profile if value)
    identity_score = _clamp(
        0.5 * (profile_fields / 3.0) + 0.5 * min(1.0, alias_count / 6.0)
    )

    server_wiki_entries = None
    if server_wiki_path is not None:
        try:
            with sqlite3.connect(f"file:{server_wiki_path}?mode=ro", uri=True) as wiki:
                server_wiki_entries = wiki.execute(
                    "SELECT count(*) FROM server_wiki_entry "
                    "WHERE kind = 'person' AND key = ?",
                    (user_id,),
                ).fetchone()[0]
        except sqlite3.Error:
            server_wiki_entries = None
    if server_wiki_entries is None:
        server_wiki_score = 0.0
    elif server_wiki_entries == 0:
        server_wiki_score = 0.0
    else:
        server_wiki_score = _clamp(0.4 + server_wiki_entries / 25.0)
    graph_score = _clamp(graph_links / 300.0)

    parts = {
        "appellations": {
            "score": appellation_score,
            "detail": (
                f"呼び名 {names} 種・証拠 {evidence} 件・呼び手 {callers} 人"
                + (f"・大半からは「{top}」" if top else "・定着した呼び名はまだなし")
            ),
        },
        "embedded": {
            "score": embedded_score,
            "detail": f"発言 {total_messages:,} 件中 {embedded:,} 件を埋め込み済み",
        },
        "identity": {
            "score": identity_score,
            "detail": f"プロフィール項目 {profile_fields}/3・別名 {alias_count} 件",
        },
        "server_wiki": {
            "score": server_wiki_score,
            "detail": (
                "Server Wiki の人物記事は未確認" if server_wiki_entries is None
                else f"Server Wiki の人物記事 {server_wiki_entries} 件"
            ),
        },
        "person_graph": {
            "score": graph_score,
            "detail": f"返信・メンションの関係 {graph_links:,} 件",
        },
    }
    total = round(
        sum(WEIGHTS[key] * part["score"] for key, part in parts.items()), 1
    )
    return {
        "user_id": user_id,
        "total": total,
        "parts": parts,
        "priority_tier": tier_row is not None,
        "use_count": tier_row[0] if tier_row else 0,
        "last_score": tier_row[1] if tier_row else None,
        "last_processed_at": tier_row[2] if tier_row else None,
    }


def record_me_use(store, user_id):
    """Register a /me use: bump the tier record and queue a deepening
    job unless one is already waiting. Returns (use_count, queued)."""
    user_id = str(user_id)
    now = _now()
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_me_schema(db)
        row = db.execute(
            "SELECT use_count FROM me_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            db.execute(
                "INSERT INTO me_users (user_id, first_used_at, last_used_at, "
                "use_count) VALUES (?, ?, ?, 1)",
                (user_id, now, now),
            )
            use_count = 1
        else:
            use_count = row[0] + 1
            db.execute(
                "UPDATE me_users SET last_used_at = ?, use_count = ? "
                "WHERE user_id = ?",
                (now, use_count, user_id),
            )
        pending = db.execute(
            "SELECT count(*) FROM understanding_jobs "
            "WHERE user_id = ? AND status IN ('pending', 'running')",
            (user_id,),
        ).fetchone()[0]
        queued = False
        if not pending:
            db.execute(
                "INSERT INTO understanding_jobs (user_id, requested_at) "
                "VALUES (?, ?)",
                (user_id, now),
            )
            queued = True
    return use_count, queued


def enqueue_stale_me_users(store, *, stale_hours=24):
    """Re-queue tier members whose deepening has gone stale, so the
    tier keeps being refreshed more often than the general archive.
    Returns how many jobs were queued."""
    cutoff = (datetime.now(UTC) - timedelta(hours=stale_hours)).isoformat()
    now = _now()
    queued = 0
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_me_schema(db)
        rows = db.execute(
            "SELECT user_id FROM me_users WHERE last_processed_at IS NULL "
            "OR last_processed_at < ?",
            (cutoff,),
        ).fetchall()
        for (user_id,) in rows:
            pending = db.execute(
                "SELECT count(*) FROM understanding_jobs "
                "WHERE user_id = ? AND status IN ('pending', 'running')",
                (user_id,),
            ).fetchone()[0]
            if not pending:
                db.execute(
                    "INSERT INTO understanding_jobs (user_id, requested_at) "
                    "VALUES (?, ?)",
                    (user_id, now),
                )
                queued += 1
    return queued


def _rebuild_aliases(store, user_id):
    """Fill author_aliases gaps from the name snapshots stored on the
    person's own messages (the ingest-time alias writer only sees the
    names a record carried when it arrived). Returns rows added."""
    added = 0
    with store.connect() as db:
        rows = db.execute(
            "SELECT DISTINCT channel_id, author_name, author_username, "
            "author_global_name, author_nickname FROM messages "
            "WHERE author_id = ?",
            (user_id,),
        ).fetchall()
        guild_id = store.guild_id
        for channel_id, *names in rows:
            for name in names:
                if not name or not _name(name):
                    continue
                cursor = db.execute(
                    "INSERT OR IGNORE INTO author_aliases VALUES (?, ?, ?, ?, ?)",
                    (guild_id, channel_id, user_id, str(name), _name(name)),
                )
                added += cursor.rowcount
    return added


def process_understanding_jobs(store, *, max_jobs=1, server_wiki_path=None):
    """Run pending deepening jobs, oldest first. Each job: catch the
    appellation miner up, rebuild the member's aliases from stored
    name snapshots, then re-measure and record the score movement.
    Returns the number of jobs completed."""
    done = 0
    for _ in range(max_jobs):
        with store.connect() as db:
            ensure_me_schema(db)
            job = db.execute(
                "SELECT id, user_id FROM understanding_jobs "
                "WHERE status = 'pending' ORDER BY id LIMIT 1"
            ).fetchone()
            if job is None:
                return done
            job_id, user_id = job
            db.execute(
                "UPDATE understanding_jobs SET status = 'running' WHERE id = ?",
                (job_id,),
            )
        try:
            before = collect_understanding(store, user_id, server_wiki_path=server_wiki_path)
            mined = mine_appellations(store)
            aliases_added = _rebuild_aliases(store, user_id)
            after = collect_understanding(store, user_id, server_wiki_path=server_wiki_path)
            detail = (
                f"mined_pairs={mined['pairs']} aliases_added={aliases_added} "
                f"score {before['total']} -> {after['total']}"
            )
            status = "done"
        except Exception as error:  # a failed job must not wedge the queue
            logger.exception("understanding job %s failed", job_id)
            detail = f"{type(error).__name__}: {error}"[:300]
            status = "failed"
            after = None
        with store.connect() as db:
            db.execute(
                "UPDATE understanding_jobs SET status = ?, processed_at = ?, "
                "detail = ? WHERE id = ?",
                (status, _now(), detail, job_id),
            )
            if after is not None:
                db.execute(
                    "UPDATE me_users SET last_score = ?, last_processed_at = ? "
                    "WHERE user_id = ?",
                    (after["total"], _now(), user_id),
                )
        done += 1
    return done


_PART_LABELS = {
    "appellations": "🏷 呼ばれ方",
    "embedded": "🧠 発言の埋め込み",
    "identity": "🪪 名前・別名の記録",
    "server_wiki": "📚 Server Wiki 人物記事",
    "person_graph": "🕸 人物関係",
}


def build_me_embed(report, *, display_name, queued):
    """Render one member's understanding report as the /me embed."""
    import discord

    embed = discord.Embed(
        title=f"{display_name} さんへの理解度",
        description=f"**{report['total']:.0f}%**",
        color=discord.Color.blurple(),
    )
    for key, part in report["parts"].items():
        embed.add_field(
            name=f"{_PART_LABELS[key]}（{WEIGHTS[key]}点満点中 "
                 f"{WEIGHTS[key] * part['score']:.0f}点）",
            value=part["detail"],
            inline=False,
        )
    notes = []
    if report["priority_tier"]:
        notes.append("⭐ 徹底理解の優先対象（/me 利用者）として、裏で継続的に理解を深めています")
    if queued:
        notes.append("いま理解度アップの処理をキューに入れたよ。次に /me したときに反映される")
    elif report["last_processed_at"]:
        notes.append(f"前回の裏処理: {report['last_processed_at'][:10]}"
                     f"（その時点の理解度 {report['last_score']}%)")
    if notes:
        embed.add_field(name="裏機能", value="\n".join(notes), inline=False)
    return embed
