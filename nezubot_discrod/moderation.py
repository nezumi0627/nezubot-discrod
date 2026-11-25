"""Policy moderation: silent points, a 24h no-response list, and
instant blacklisting for the worst categories (see POLICY.md).

Two detection layers feed one recorder:

1. A local pattern layer (``check_message``) screens the text of a
   message that calls the bot. The patterns are deliberately
   conservative — they fire only on unambiguous violations, never on
   ordinary development or security-research vocabulary, because this
   server is a developer community where discussing vulnerabilities,
   exploits (defensively) and reverse engineering is normal.
2. The answering model itself flags a refused request with a hidden
   ``[[policy:<category>]]`` marker at the end of its reply (the
   instruction lives in prompts.POLICY_GUIDANCE). The bot strips the
   marker from the delivered text — in bot.py, on the answer text,
   so both the Codex and the Devin path are covered — and records
   the violation here.

Scoring is silent by design: a point addition produces no message
anywhere, to the member or to the channel. Points decay by one per
violation-free 24 hours (applied lazily on read/update). At 3 points
the member joins the no-response (mute) list for 24 hours and the
points reset; the mute is enforced by the same gates as the
blacklist. A severe category skips points entirely and lands the
member on the blacklist at once. Escalations — mute entry,
blacklist entry (policy or manual) and blacklist removal — are
announced with an embed posted to the channel where they happened;
the owner is never DM'd about moderation. The public embed never
carries the violating text: excerpts live only in the
moderation_violations table.

The owner (settings.owner_id) is exempt from everything here: the
call sites skip moderation for the owner, and record_violation
itself refuses to record an owner violation as a backstop.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import discord

from nezubot_discrod.blacklist import add_to_blacklist

logger = logging.getLogger(__name__)

# --- categories (mirror POLICY.md; keep the two in sync) --------------

SEVERITY_STANDARD = "standard"
SEVERITY_SEVERE = "severe"

CATEGORIES = {
    "sexual_minor": SEVERITY_SEVERE,
    "exploit_abuse": SEVERITY_SEVERE,
    "doxxing": SEVERITY_SEVERE,
    "incitement": SEVERITY_SEVERE,
    "sexual": SEVERITY_STANDARD,
    "discrimination": SEVERITY_STANDARD,
    "other": SEVERITY_STANDARD,
}

CATEGORY_LABELS = {
    "sexual_minor": "未成年が関わる性的内容",
    "exploit_abuse": "他者攻撃のためのエクスプロイト／マルウェア作成",
    "doxxing": "個人情報の暴露・特定の要求",
    "incitement": "暴力の扇動・加害手段の要求",
    "sexual": "下ネタ・性的表現",
    "discrimination": "差別的発言",
    "other": "その他のポリシー違反",
}

POINTS_PER_VIOLATION = 1
MUTE_THRESHOLD = 3
MUTE_DURATION = timedelta(hours=24)
DECAY_PERIOD = timedelta(hours=24)
EXCERPT_LIMIT = 200

# The one fixed reply a layer-1 hit gets. Points, lists and the
# policy itself are never mentioned to the member.
REFUSAL_TEXT = "ごめん、その内容には答えられないよ。"

# --- layer 1: local patterns -------------------------------------------
#
# Conservative on purpose: each pattern pairs a violation noun with a
# request/attack verb (or an equally unambiguous shape), so research
# and defence talk — マルウェア解析、乗っ取り対策、パスワードを盗まれた
# (victim), エロゲー開発, 性的マイノリティの話題 — does not fire.
# Severe categories are checked first.

_PATTERNS = (
    (
        "sexual_minor",
        (
            r"(未成年|児童|小学生|中学生|高校生|女子高生|女子中学生)"
            r".{0,10}(エロ|性的|ポルノ)"
            r".{0,10}(書いて|作って|生成して|見せて|教えて|ください)",
            r"(ロリ|ショタ).{0,10}(エロ|ポルノ|性的)",
            r"(児童|未成年).{0,6}(ポルノ|性的コンテンツ)",
            r"(?i)(child|minor|underage).{0,12}(porn|sexual|nude)",
        ),
    ),
    (
        "exploit_abuse",
        (
            r"(マルウェア|ランサムウェア|トロイの木馬|キーロガー|スティーラー)"
            r".{0,10}(作って|作り方|作成|開発して|書いて|生成して)",
            r"(他人の|人の|誰かの)"
            r".{0,10}(パスワード|アカウント|認証情報)"
            r".{0,8}(盗んで|盗む|抜き取って|乗っ取って)",
            r"(パスワード|認証情報).{0,6}(盗んで|盗む方法|抜き取る方法)",
            r"(アカウント|垢).{0,6}(乗っ取って|乗っ取る方法|乗っ取り方)",
            r"(DDoS|DoS).{0,8}(かけて|仕掛けて|攻撃して|やり方)",
            r"フィッシング.{0,10}(作って|作り方|作成)",
            r"(?i)(write|create|make|build).{0,16}(malware|ransomware|keylogger)",
        ),
    ),
    (
        "doxxing",
        (
            r"(他人の|人の|誰かの)"
            r".{0,10}(住所|電話番号|本名|個人情報)"
            r".{0,8}(晒|暴露|特定|教えて|知りたい|調べて|割り出)",
            r"(住所|電話番号|本名|個人情報)"
            r".{0,8}(晒してくれ|晒してや|暴露して|特定してくれ|特定してや|割り出して)",
        ),
    ),
    (
        "incitement",
        (
            r"(爆弾|爆発物|火炎瓶).{0,8}(作り方|作って|製造法|作る方法)",
            r"(殺し方|殺害方法|人を殺す方法)",
            r"(放火|テロ攻撃).{0,6}(やり方|方法|仕方)",
        ),
    ),
    (
        "sexual",
        (
            r"(エロい|性的な).{0,8}(話|画像|動画|コンテンツ|小説|文章)"
            r".{0,6}(して|話して|書いて|生成して|作って|見せて|ください|欲しい)",
            r"エロ(?!ゲ).{0,8}(話して|書いて|生成して|作って|見せて|ください)",
            r"(ポルノ|アダルト).{0,10}(見せて|教えて|ください|生成|作って)",
            r"下ネタ.{0,8}(言って|話して|やって|出して|ください)",
            r"セックス.{0,8}(話して|したい|話をして)",
            r"(まんこ|ちんこ|ちんぽ)",
        ),
    ),
    (
        "discrimination",
        (
            r"(黒人|白人|中国人|韓国人|朝鮮人|在日|障害者|障碍者|外国人|ユダヤ人)"
            r".{0,12}(死ね|消えろ|いらない|要らない|劣っている|排除|出ていけ|帰れ)",
            r"(死ね|消えろ|排除しろ).{0,8}(黒人|中国人|韓国人|朝鮮人|障害者|外国人)",
        ),
    ),
)

_COMPILED_PATTERNS = tuple(
    (category, tuple(re.compile(p, re.DOTALL) for p in patterns))
    for category, patterns in _PATTERNS
)


@dataclass(frozen=True)
class PolicyMatch:
    category: str
    severity: str


def check_message(text):
    """Layer-1 screen: a PolicyMatch for an unambiguous violation,
    else None. Pure and cheap; safe to run on every calling message."""
    if not text:
        return None
    for category, patterns in _COMPILED_PATTERNS:
        for pattern in patterns:
            if pattern.search(text):
                return PolicyMatch(category, CATEGORIES[category])
    return None


# --- layer 2: the model's hidden marker ---------------------------------

MARKER_RE = re.compile(r"\[\[policy:([a-z_]+)\]\]")


def extract_policy_marker(text):
    """Split a model answer into (cleaned text, category or None).

    The marker is machine-read only: every occurrence is removed from
    the delivered text. An unknown category still counts as a flag —
    the model clearly meant to refuse — and is recorded as ``other``.
    """
    if not text:
        return text, None
    found = MARKER_RE.findall(str(text))
    if not found:
        return text, None
    category = found[0] if found[0] in CATEGORIES else "other"
    cleaned = MARKER_RE.sub("", str(text)).strip()
    return cleaned, category


# --- storage --------------------------------------------------------------

MODERATION_DDL = """
CREATE TABLE IF NOT EXISTS moderation_points (
    user_id TEXT PRIMARY KEY,
    points INTEGER NOT NULL DEFAULT 0,
    last_violation_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS moderation_violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    points_added INTEGER NOT NULL,
    source TEXT NOT NULL,
    excerpt TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS moderation_mutes (
    user_id TEXT PRIMARY KEY,
    until TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reason TEXT
);
"""


def ensure_moderation_schema(db):
    db.executescript(MODERATION_DDL)


def init_moderation(store):
    """Create the moderation tables. Safe to call on every startup."""
    with store.connect() as db:
        ensure_moderation_schema(db)


def _parse_dt(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _decayed(points, last_violation_at, now):
    """Apply point decay: -1 per full violation-free DECAY_PERIOD
    since the last violation, never below zero. Returns
    (points, last_violation_at); the anchor advances by the consumed
    periods so partial progress toward the next decay carries over."""
    if points <= 0 or last_violation_at is None:
        return points, last_violation_at
    periods = int((now - last_violation_at) // DECAY_PERIOD)
    if periods <= 0:
        return points, last_violation_at
    decayed = max(0, points - periods)
    if decayed <= 0:
        return 0, last_violation_at
    return decayed, last_violation_at + periods * DECAY_PERIOD


def _read_points_row(db, user_id):
    row = db.execute(
        "SELECT points, last_violation_at FROM moderation_points"
        " WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if row is None:
        return 0, None
    return row[0], _parse_dt(row[1])


def _write_points(db, user_id, points, last_violation_at, now):
    db.execute(
        "INSERT INTO moderation_points"
        " (user_id, points, last_violation_at, updated_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(user_id) DO UPDATE SET"
        " points = excluded.points,"
        " last_violation_at = excluded.last_violation_at,"
        " updated_at = excluded.updated_at",
        (
            user_id,
            points,
            last_violation_at.isoformat() if last_violation_at else None,
            now.isoformat(),
        ),
    )


def get_points(store, user_id, *, now=None):
    """Current points for a member, with lazy decay applied (and
    persisted when it changed anything)."""
    now = now or datetime.now(UTC)
    user_id = str(user_id)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_moderation_schema(db)
        points, last = _read_points_row(db, user_id)
        decayed, new_last = _decayed(points, last, now)
        if decayed != points or new_last != last:
            _write_points(db, user_id, decayed, new_last, now)
        return decayed


# --- mutes (the no-response list) -----------------------------------------
#
# Cache shape mirrors blacklist.py: per store path, refreshed at most
# every 30 s. Unlike the blacklist's id set, the cache keeps each
# mute's expiry (and the category that tripped it) so an entry stops
# applying at the exact moment it lapses, not at the next refresh,
# and a refusal can say why and until when.

CACHE_TTL_SECONDS = 30.0

# store path -> ({user_id: (until datetime, reason)}, monotonic expiry)
_mute_cache = {}


def _invalidate_mutes(store):
    _mute_cache.pop(str(store.path), None)


def _cached_mutes(store):
    key = str(store.path)
    now_mono = time.monotonic()
    entry = _mute_cache.get(key)
    if entry is not None and now_mono < entry[1]:
        return entry[0]
    with store.connect() as db:
        ensure_moderation_schema(db)
        mutes = {
            row[0]: (_parse_dt(row[1]), row[2])
            for row in db.execute(
                "SELECT user_id, until, reason FROM moderation_mutes"
            )
        }
    _mute_cache[key] = (mutes, now_mono + CACHE_TTL_SECONDS)
    return mutes


def is_muted(store, user_id, *, now=None):
    """True while the member is on the no-response list."""
    if user_id is None:
        return False
    entry = _cached_mutes(store).get(str(user_id))
    if entry is None:
        return False
    return entry[0] > (now or datetime.now(UTC))


def get_mute_until(store, user_id):
    """The member's mute expiry, or None when not muted."""
    if user_id is None:
        return None
    entry = _cached_mutes(store).get(str(user_id))
    return entry[0] if entry is not None else None


@dataclass(frozen=True)
class MuteInfo:
    """A live mute: when it lapses and which category tripped it."""

    until: datetime
    reason: str | None  # category key stored on the mute row


def get_mute_info(store, user_id, *, now=None):
    """The member's MuteInfo while muted, else None (never muted or
    the mute has already lapsed)."""
    if user_id is None:
        return None
    entry = _cached_mutes(store).get(str(user_id))
    if entry is None:
        return None
    until, reason = entry
    if until <= (now or datetime.now(UTC)):
        return None
    return MuteInfo(until, reason)


# --- recording --------------------------------------------------------------


@dataclass(frozen=True)
class ModerationNotice:
    """Everything the channel notice embed needs. Built only on
    escalation (mute entry / policy blacklist entry) — a plain point
    addition has notice=None and stays invisible to everyone. The
    excerpt rides along for the internal record only; the public
    embed builder deliberately never renders it."""

    kind: str  # "mute" | "blacklist"
    user_id: str
    category: str
    severity: str
    excerpt: str
    at: datetime
    mute_until: datetime | None = None


@dataclass(frozen=True)
class ViolationOutcome:
    action: str  # "none" | "muted" | "blacklisted" | "exempt"
    category: str
    severity: str
    points: int  # points after this violation (0 once reset/cleared)
    notice: ModerationNotice | None = None
    mute_until: datetime | None = None


def record_violation(
    store,
    user_id,
    category,
    *,
    excerpt=None,
    source="pattern",
    owner_id=None,
    now=None,
):
    """Record one violation and apply the points/mute/blacklist rules.

    ``source`` is "pattern" (layer 1) or "model" (layer 2 marker).
    The owner is exempt: with owner_id given, recording an owner
    violation writes nothing and returns action="exempt".
    """
    user_id = str(user_id)
    if category not in CATEGORIES:
        category = "other"
    severity = CATEGORIES[category]
    if owner_id is not None and user_id == str(owner_id):
        return ViolationOutcome("exempt", category, severity, 0)
    now = now or datetime.now(UTC)
    excerpt = (excerpt or "")[:EXCERPT_LIMIT]
    severe = severity == SEVERITY_SEVERE
    action = "none"
    points_after = 0
    mute_until = None
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_moderation_schema(db)
        db.execute(
            "INSERT INTO moderation_violations"
            " (user_id, category, severity, points_added, source,"
            " excerpt, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                category,
                severity,
                0 if severe else POINTS_PER_VIOLATION,
                source,
                excerpt,
                now.isoformat(),
            ),
        )
        if severe:
            # Straight to the blacklist: points and any live mute are
            # moot once the stronger list applies.
            db.execute(
                "DELETE FROM moderation_points WHERE user_id = ?", (user_id,)
            )
            db.execute(
                "DELETE FROM moderation_mutes WHERE user_id = ?", (user_id,)
            )
        else:
            points, last = _read_points_row(db, user_id)
            points, last = _decayed(points, last, now)
            total = points + POINTS_PER_VIOLATION
            if total >= MUTE_THRESHOLD:
                mute_until = now + MUTE_DURATION
                db.execute(
                    "INSERT INTO moderation_mutes"
                    " (user_id, until, created_at, reason)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(user_id) DO UPDATE SET"
                    " until = excluded.until,"
                    " created_at = excluded.created_at,"
                    " reason = excluded.reason",
                    (user_id, mute_until.isoformat(), now.isoformat(), category),
                )
                _write_points(db, user_id, 0, now, now)
                action = "muted"
            else:
                _write_points(db, user_id, total, now, now)
                points_after = total
    if severe:
        add_to_blacklist(
            store, user_id, added_by=None, note=f"policy:{category}"
        )
        _invalidate_mutes(store)
        action = "blacklisted"
    elif action == "muted":
        _invalidate_mutes(store)
    notice = None
    if action == "muted":
        notice = ModerationNotice(
            "mute", user_id, category, severity, excerpt, now, mute_until
        )
    elif action == "blacklisted":
        notice = ModerationNotice(
            "blacklist", user_id, category, severity, excerpt, now
        )
    logger.info(
        "Moderation: user %s category=%s severity=%s source=%s -> %s",
        user_id, category, severity, source, action,
    )
    return ViolationOutcome(action, category, severity, points_after, notice, mute_until)


# --- channel notices ----------------------------------------------------------
#
# Escalations are announced in the channel where they happened, never
# by DM to the owner. These embeds are public-channel safe by design:
# the violating text (notice.excerpt) is NEVER rendered — excerpts
# live only in the moderation_violations table.

_JST = ZoneInfo("Asia/Tokyo")


def _format_jst(value):
    return value.astimezone(_JST).strftime("%Y-%m-%d %H:%M JST")


def _format_remaining(delta):
    """A coarse Japanese duration for 'あと約…': hours and minutes,
    never seconds (the exact release time is shown alongside)."""
    total_minutes = max(0, int(delta.total_seconds()) // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}時間{minutes}分"
    if hours:
        return f"{hours}時間"
    return f"{minutes}分"


def format_mute_refusal(info, *, now=None):
    """The ephemeral slash refusal for a muted member: why they are
    restricted, for how much longer, and until when (JST)."""
    now = now or datetime.now(UTC)
    label = CATEGORY_LABELS.get(info.reason, info.reason or "ポリシー違反")
    return (
        f"あなたはポリシー違反（{label}）でポイントが{MUTE_THRESHOLD}点に"
        f"達したため、利用を制限しています。"
        f"あと約{_format_remaining(info.until - now)}"
        f"（{_format_jst(info.until)} まで）制限が続きます。"
    )


def build_escalation_embed(notice, *, user_name=None):
    """The public channel embed for a mute / policy-blacklist entry.

    Carries who, what action, the reason category, and — for a
    mute — the 24h duration and the JST release time. The violating
    excerpt is deliberately NOT included: it stays in the internal
    violations table only.
    """
    label = CATEGORY_LABELS.get(notice.category, notice.category)
    name = user_name or "（名前不明）"
    if notice.kind == "mute":
        title = "モデレーション通知: ミュート（利用制限）"
        color = discord.Color.orange()
        reason = f"{label}（ポリシー違反ポイントが{MUTE_THRESHOLD}点に到達）"
        applied = "24時間利用を制限しています"
        if notice.mute_until is not None:
            applied += f"（解除: {_format_jst(notice.mute_until)}）"
    else:
        title = "モデレーション通知: ブラックリストに登録"
        color = discord.Color.red()
        reason = f"{label}（重大カテゴリのため即時）"
        applied = "ブラックリストに登録しました"
    embed = discord.Embed(title=title, color=color, timestamp=notice.at)
    embed.add_field(
        name="対象ユーザー", value=f"{name}（ID: {notice.user_id}）", inline=False
    )
    embed.add_field(name="理由", value=reason, inline=False)
    embed.add_field(name="適用内容", value=applied, inline=False)
    embed.add_field(name="時刻", value=_format_jst(notice.at), inline=False)
    return embed


def build_blacklist_change_embed(user_id, *, added, user_name=None, at=None):
    """The public channel embed for a manual /blacklist add/remove."""
    at = at or datetime.now(UTC)
    name = user_name or "（名前不明）"
    if added:
        title = "モデレーション通知: ブラックリストに登録"
        color = discord.Color.red()
        applied = "ブラックリストに登録しました（オーナーの操作）"
    else:
        title = "モデレーション通知: ブラックリストから解除"
        color = discord.Color.green()
        applied = "ブラックリストから解除しました（オーナーの操作）"
    embed = discord.Embed(title=title, color=color, timestamp=at)
    embed.add_field(
        name="対象ユーザー", value=f"{name}（ID: {user_id}）", inline=False
    )
    embed.add_field(name="適用内容", value=applied, inline=False)
    embed.add_field(name="時刻", value=_format_jst(at), inline=False)
    return embed
