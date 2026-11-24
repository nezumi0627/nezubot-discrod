"""Read-only learning-status aggregation behind the n!status command.

Every query runs against the live archive through a read-only SQLite
connection (the bot keeps writing to the same WAL database) and each
one is an indexed count that returns in about a second; callers cache
the result (see prefix_commands) so repeated n!status calls stay cheap.

"Eligible" mirrors the dense embedder's queue filter in
server_log_dense.embed_pending: not deleted or tombstoned, human
author, at least two non-space characters. The learning rate is the
share of eligible messages that carry a vector from the teacher model;
static-distilled vectors are reported separately because the teacher
has since covered everything they used to fill in for.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

_NOT_DELETED = (
    "m.deleted = 0 AND NOT EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id)"
)
_ELIGIBLE = f"{_NOT_DELETED} AND m.is_bot = 0 AND length(trim(m.text)) >= 2"


def collect_status(db_path, *, teacher_model, static_model=None, server_wiki_path=None):
    """Aggregate the learning stats from the archive at db_path.

    Two index scans total: one over the surviving messages for the
    totals, one over the eligible subset with the vector tables joined
    in for coverage. Measured on the production archive, each scan
    returns in roughly a second even while the bot is writing.
    """
    db_path = Path(db_path)
    stats = {}
    static_param = static_model if static_model else teacher_model
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        db.execute("PRAGMA temp_store = MEMORY")
        db.execute("PRAGMA cache_size = -64000")
        total, people, channels = db.execute(
            f"SELECT count(*), "
            f"count(DISTINCT CASE WHEN m.is_bot = 0 THEN m.author_id END), "
            f"count(DISTINCT m.channel_id) "
            f"FROM messages m WHERE {_NOT_DELETED}"
        ).fetchone()
        eligible, teacher_covered, static_only = db.execute(
            "SELECT count(*), "
            "COALESCE(sum(t.message_id IS NOT NULL), 0), "
            "COALESCE(sum(s.message_id IS NOT NULL AND t.message_id IS NULL), 0) "
            "FROM messages m "
            "LEFT JOIN message_vectors t ON t.message_id = m.id AND t.model = ? "
            "LEFT JOIN message_vectors s ON s.message_id = m.id AND s.model = ? "
            f"WHERE {_ELIGIBLE}",
            (teacher_model, static_param),
        ).fetchone()
        stats["total_messages"] = total
        stats["people"] = people
        stats["channels"] = channels
        stats["eligible"] = eligible
        stats["teacher_covered"] = teacher_covered
        stats["static_only"] = static_only if static_model != teacher_model else 0
    stats["server_wiki_people"] = None
    stats["server_wiki_events"] = None
    if server_wiki_path is not None and Path(server_wiki_path).exists():
        try:
            with closing(sqlite3.connect(f"file:{server_wiki_path}?mode=ro", uri=True)) as db:
                db.execute("PRAGMA query_only = ON")
                by_kind = dict(
                    db.execute("SELECT kind, count(*) FROM server_wiki_entry GROUP BY kind")
                )
            stats["server_wiki_people"] = by_kind.get("person", 0)
            stats["server_wiki_events"] = by_kind.get("event", 0)
        except sqlite3.Error:
            stats["server_wiki_people"] = None
            stats["server_wiki_events"] = None
    stats["db_bytes"] = sum(
        candidate.stat().st_size
        for candidate in (
            db_path,
            db_path.with_name(db_path.name + "-wal"),
            db_path.with_name(db_path.name + "-shm"),
        )
        if candidate.exists()
    )
    return stats


def learning_rate(stats):
    """Teacher coverage of eligible messages, as a percentage."""
    if not stats["eligible"]:
        return 0.0
    return stats["teacher_covered"] / stats["eligible"] * 100


def combined_rate(stats):
    """Teacher + static-only coverage of eligible messages, as a percentage."""
    if not stats["eligible"]:
        return 0.0
    return (stats["teacher_covered"] + stats["static_only"]) / stats["eligible"] * 100


def format_uptime(seconds):
    if seconds is None:
        return "—"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}日{hours}時間{minutes}分"
    if hours:
        return f"{hours}時間{minutes}分"
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def format_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.2f} {unit}"
        value /= 1024


def build_status_embed(stats, *, model, version, uptime_seconds=None):
    """Render the aggregated stats as the n!status embed."""
    import discord

    embed = discord.Embed(
        title="nezubot-discrod 学習ステータス",
        description=f"バージョン v{version} / モデル {model}",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="📨 学習メッセージ数", value=f"{stats['total_messages']:,} 件", inline=True
    )
    embed.add_field(name="👥 学習人数", value=f"{stats['people']:,} 人", inline=True)
    embed.add_field(name="🧵 学習済みチャンネル", value=f"{stats['channels']:,} 件", inline=True)
    rate_text = (
        f"{learning_rate(stats):.2f}%"
        f"（教師ベクトル {stats['teacher_covered']:,} / 対象 {stats['eligible']:,} 件）"
    )
    if stats["static_only"]:
        rate_text += (
            f"\n静的ベクトル込み: {combined_rate(stats):.2f}%"
            f"（静的のみ {stats['static_only']:,} 件）"
        )
    embed.add_field(name="🎓 学習率（埋め込みカバー率）", value=rate_text, inline=False)
    if stats["server_wiki_people"] is None:
        server_wiki_text = "未生成"
    else:
        server_wiki_text = (
            f"{stats['server_wiki_people'] + stats['server_wiki_events']:,} 件"
            f"（人物 {stats['server_wiki_people']:,} / 出来事 {stats['server_wiki_events']:,}）"
        )
    embed.add_field(name="📚 Server Wiki ウィキ記事数", value=server_wiki_text, inline=True)
    embed.add_field(name="💾 DB サイズ", value=format_size(stats["db_bytes"]), inline=True)
    embed.add_field(name="⏱ 稼働時間", value=format_uptime(uptime_seconds), inline=True)
    return embed
