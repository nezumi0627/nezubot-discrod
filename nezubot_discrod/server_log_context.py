import asyncio
import json
import logging
import os
import queue
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import UTC, datetime
from pathlib import Path
from threading import BoundedSemaphore, Event
from time import perf_counter
from zoneinfo import ZoneInfo

from nezubot_discrod import github_meta
from nezubot_discrod.appellations import AppellationIndex, appellation_list
from nezubot_discrod.appellations import person_summary as build_person_summary
from nezubot_discrod.claim_check import (
    CLAIM_CHECK_HEADER,
    claim_keywords,
    detect_claim_dispute,
)
from nezubot_discrod.messages import Source
from nezubot_discrod.query_rewrite import needs_rewrite
from nezubot_discrod.reading import books_chunk
from nezubot_discrod.server_log import (
    LEXICAL_MEMO,
    REPLY_GRAPH_MAX_NEIGHBORS,
    REPLY_GRAPH_SEED_LIMIT,
    DenseSnapshotUnavailable,
    ServerLog,
    new_lexical_memo,
)
from nezubot_discrod.server_log import (
    _hit as _log_hit,
)
from nezubot_discrod.server_log_timing import CURRENT, Timings, stage
from nezubot_discrod.server_log_tools import LogResult
from nezubot_discrod.server_wiki import ServerWiki
from nezubot_discrod.turn_context import current_turn_id

logger = logging.getLogger(__name__)
_SQL_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="context-sql")
_DENSE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="context-dense")
_WIKI_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="context-wiki")
_MEMBER_WIKI_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="context-member-wiki")
_SQL_SLOTS = BoundedSemaphore(8)
_DENSE_SLOTS = BoundedSemaphore(2)
_WIKI_SLOTS = BoundedSemaphore(2)
_MEMBER_WIKI_SLOTS = BoundedSemaphore(2)
_SELF = re.compile(
    r"(?:おれ|俺|わたし|私|自分|僕|ぼく|うち)(?:の|について|は|が|を|に|と|とか|って|誰|何者|$)|"
    r"\b(?:me|my|myself)\b", re.IGNORECASE,
)
# A question that turns on GitHub authorship or accounts: these get
# the deterministic GitHub metadata block (github_meta), because the
# archive alone cannot distinguish a repo's maker from someone who
# pasted its URL or holds a fork of it.
_GITHUB_TOPIC = re.compile(r"github|ギットハブ", re.IGNORECASE)
# The shape of a question that is about a person even when no name in
# it resolved: used only to flag the turn as person-unresolved, so the
# caller can tell the model to make one resolution attempt of its own
# instead of answering 「特定できない」 from an empty evidence set.
_PERSON_QUESTION = re.compile(
    r"について|って誰|とは|どんな人|誰[？?]|知ってる|知っている|呼ばれ方|って何して|"
    r"のプロフィール|について教え|についておしえ"
)
_APPELLATION_QUESTION = re.compile(r"呼ばれ方|あだ名|ニックネーム|呼び名")
_JST = ZoneInfo("Asia/Tokyo")
# The shape of a question that asks WHEN a person last spoke (or first
# appeared): 「最後いつ話してた？」「久しぶりに顔出したけど前回は」.
# These have a deterministic answer in the archive — the person's own
# max(created_at) — but no retrieval route supplies it: the overview's
# last_seen is the all-time maximum (useless against a 「今日より前」
# boundary once the person has spoken today), its samples are
# time-stratified rather than chronological, and free search is
# relevance-ordered. Left to those, the model composes a date from
# whatever hit it read last; on 2026-11-25 it answered a regular's
# 「今日より前は最後いつ話してた？」 with a timestamp that exists
# nowhere in the archive (2025-05-29 13:41 UTC; the true previous
# activity was 2026-11-23). Questions matching this shape get a
# mechanically aggregated person_activity chunk instead.
_ACTIVITY_QUESTION = re.compile(
    r"最後に?(?:は|いつ|話し|発言|投稿|来)|最終発言|最後の発言|"
    r"いつ.{0,4}(?:話し|発言|投稿|来)た|"
    r"(?:話し|発言|投稿)(?:た|てた)の?(?:は|って)?いつ|"
    r"前回(?:は|いつ)|最近いつ|久しぶり|顔出|"
    r"ラストに(?:話し|発言)"
)
# The mirror shape: a question that asks WHEN a person FIRST spoke
# (「初参加の日にち探して」「最初の発言はいつ？」). The deterministic
# answer — the person's min(created_at) — is likewise supplied by no
# retrieval route: the overview header carries a raw first_seen inside
# a JSON blob (unrendered, unlabelled, and its samples are
# time-stratified rather than chronological), and free search is
# relevance-ordered, so hits near any date are mostly OTHER people's
# messages. Left to those, the model composes a date from whatever hit
# it read last; on 2026-11-26 it answered the owner's 「初参加」 with
# 2026-10-23 11:58 UTC, a timestamp of other people's messages (the
# true earliest for his account is 2026-07-18). Questions matching
# this shape get a mechanically aggregated person_first_activity
# chunk instead.
_FIRST_ACTIVITY_QUESTION = re.compile(
    r"初参加|初発言|初投稿|初メッセージ|"
    r"最初の(?:発言|投稿|メッセージ)|最初に(?:話し|発言|投稿|来)|"
    r"初めての?(?:話し|発言|投稿|メッセージ|参加)|初めて来|"
    r"参加した(?:のは|日)|参加日|いつ参加|加入した(?:のは|日)|"
    r"いつから(?:.{0,10}?(?:いる|居る|参加|来)|ここに)"
)


def _first_activity_signals(text, search_text=None, context_query=None):
    """Detection signals for the first-activity deterministic block.

    direct: the latest message itself asks about a first participation
    / earliest message (the v1.28.0 shape). topic: the CONVERSATION is
    about a first participation even when the latest message does not
    say so — the shape of the 2026-11-26 recurrence, where the owner's
    follow-ups (「ひま」 as a reply to the bot's own wrong
    first-participation answer, then 「しらべてそれを」) carried no
    first-activity wording at all: the topic lived in the reply
    target, whose words the rewritten query and the context query
    already carry into pre_retrieve. A context-carried match only
    counts when the latest message is context-dependent (a
    rewrite-shaped follow-up, a self-reference, or a few bare
    characters hanging on the exchange); a long self-contained
    message about something else does not inherit the topic just
    because a message in the window mentioned one. The third return
    value is that context-dependence flag itself, which also gates
    the caller fallback for the block's subject."""
    body = (text or "").strip()
    direct = bool(_FIRST_ACTIVITY_QUESTION.search(text or ""))
    context_dependent = bool(
        needs_rewrite(text) or _SELF.search(text or "") or len(body) <= 8
    )
    topic = direct or (
        context_dependent
        and bool(
            _FIRST_ACTIVITY_QUESTION.search(search_text or "")
            or _FIRST_ACTIVITY_QUESTION.search(context_query or "")
        )
    )
    return direct, topic, context_dependent


def _github_signals(text, search_text=None, context_query=None):
    """Detection signals for the deterministic GitHub block.

    direct: the latest message itself is about GitHub — it names
    GitHub or links a repo (the v1.35.0 shape). topic: the
    CONVERSATION is about GitHub even when the latest message does
    not say so — the shape of the 2026-11-27 08:45 follow-up
    failure, where the owner's 「タヌキのは」 carried only the new
    person while the GitHub topic lived in the exchange itself,
    whose words the rewritten query and the context query already
    carry into pre_retrieve (the state topic reaches this function
    through those two queries, the same route the first-activity
    signals take). The v1.35.0 scheduling keyed on the raw text
    alone, so the block was never even attempted for the resolved
    person and the model, left with no GitHub fact for them,
    declined to answer. A context-carried match only counts when
    the latest message is context-dependent (a rewrite-shaped
    follow-up, a self-reference, or a few bare characters); a long
    self-contained message about something else does not inherit
    the topic. The third return value is that context-dependence
    flag, mirroring _first_activity_signals."""
    body = (text or "").strip()
    direct = bool(
        _GITHUB_TOPIC.search(text or "")
        or github_meta.extract_github_refs(text)
    )
    context_dependent = bool(
        needs_rewrite(text) or _SELF.search(text or "") or len(body) <= 8
    )
    topic = direct or (
        context_dependent
        and bool(
            _GITHUB_TOPIC.search(search_text or "")
            or _GITHUB_TOPIC.search(context_query or "")
            or github_meta.extract_github_refs(search_text)
            or github_meta.extract_github_refs(context_query)
        )
    )
    return direct, topic, context_dependent


# Surrounding decoration punctuation is not part of the name people
# type: "-aki-" is referred to as "aki". Without the strip, the
# bare form existed only as the stripped core of relay accounts'
# persona aliases ("aki (Matrix)"), so the bare name resolved to
# the relay instead of the person.
_DECORATION_EDGE = "-_.~・=+|/\\:;!?\"'`^ 　"


def _alias_core(alias):
    """The bare name inside a decorated display name: bracketed segments
    (【…】, […], （…）) removed, any ``#suffix`` (さくら#abcd) cut, a
    trailing honorific dropped, and surrounding decoration punctuation
    stripped. People refer to each other by the core, so the primary
    alias match tries it alongside the full alias."""
    core = re.sub(r"【[^】]*】|\[[^\]]*\]|[(（][^)）]*[)）]", "", alias)
    core = re.split(r"[#＃]", core, maxsplit=1)[0].strip()
    core = re.sub(r"(?:ちゃん|くん|さん|氏|さま)$", "", core).strip()
    return core.strip(_DECORATION_EDGE) or core


class _AliasSnapshot:
    """Distinct (normalized, author_id) alias pairs for one channel set,
    with the fallback token keys computed lazily on first use."""

    __slots__ = ("rows", "_keys", "_cores")

    def __init__(self, rows, cores=None):
        self.rows = [(normalized, author_id) for normalized, author_id in rows]
        self._keys = None
        self._cores = cores

    def token_keys(self):
        if self._keys is None:
            by_alias = {}
            keys = []
            for normalized, _ in self.rows:
                if normalized not in by_alias:
                    by_alias[normalized] = _alias_token_keys(normalized)
                keys.append(by_alias[normalized])
            self._keys = keys
        return self._keys

    def core_names(self):
        if self._cores is None:
            by_alias = {}
            cores = []
            for normalized, _ in self.rows:
                if normalized not in by_alias:
                    core = _alias_core(normalized)
                    by_alias[normalized] = core if core != normalized else None
                cores.append(by_alias[normalized])
            self._cores = cores
        return self._cores


_ALIAS_DISK_FORMAT = 2
_ALIAS_DISK_KEEP = 16


def _alias_disk_path(store, key):
    """Where the persisted snapshot for one channel set lives: a side
    directory next to the archive DB, named by the set's hash. None
    when the store has no filesystem path (test doubles)."""
    path = getattr(store, "path", None)
    if path is None:
        return None
    import hashlib

    digest = hashlib.sha256("\n".join(key).encode()).hexdigest()[:24]
    return Path(str(path) + ".alias-snapshots") / f"{digest}.json"


def _alias_snapshot_from_disk(store, key, watermark):
    """The persisted snapshot for this channel set, when its watermark
    still matches the alias table's. A disk hit turns a cold-process
    DISTINCT scan (seconds on the production archive, and the first
    thing the grounding budget's progress handler interrupts) into a
    file read that fits the budget."""
    path = _alias_disk_path(store, key)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != _ALIAS_DISK_FORMAT:
            return None
        if payload.get("watermark") != watermark:
            return None
        rows = payload["rows"]
        cores = payload.get("cores")
        if cores is not None and len(cores) != len(rows):
            cores = None
        return _AliasSnapshot(rows, cores=cores)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _alias_snapshot_to_disk(store, key, watermark, snapshot):
    """Persist a freshly scanned snapshot (best-effort; a cache write
    failure must never fail resolution)."""
    path = _alias_disk_path(store, key)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _ALIAS_DISK_FORMAT,
            "watermark": watermark,
            "rows": [list(row) for row in snapshot.rows],
            "cores": snapshot.core_names(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        files = sorted(path.parent.glob("*.json"), key=lambda item: item.stat().st_mtime)
        for stale in files[: max(0, len(files) - _ALIAS_DISK_KEEP)]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def _alias_snapshot(store, channel_ids, db):
    """Alias pairs for the channel set, cached on the store.

    Aliases are only ever inserted (never updated or deleted), so the
    table's MAX(rowid) is a sound invalidation watermark: while it is
    unchanged the cached pairs are exactly what a fresh scan would read,
    and resolution results are identical. The per-turn cost drops from
    two full alias scans plus one existence query per matching alias to
    one watermark probe plus one batched existence query. A disk copy
    (validated by the same watermark) survives process restarts, so a
    cold process no longer pays the scan inside the grounding budget —
    the interruption that degraded resolution to nobody."""
    key = tuple(sorted({str(channel) for channel in channel_ids}))
    cache = store.__dict__.setdefault("_alias_snapshots", {})
    watermark = db.execute("SELECT MAX(rowid) FROM author_aliases").fetchone()[0]
    entry = cache.get(key)
    if entry is not None and entry[0] == watermark:
        return entry[1]
    snapshot = _alias_snapshot_from_disk(store, key, watermark)
    if snapshot is None:
        rows = db.execute(
            "SELECT DISTINCT normalized, author_id FROM author_aliases "
            f"WHERE channel_id IN ({','.join('?' for _ in key)})",
            [*key],
        ).fetchall()
        snapshot = _AliasSnapshot(rows)
        _alias_snapshot_to_disk(store, key, watermark, snapshot)
    cache[key] = (watermark, snapshot)
    return snapshot


_APPELLATION_TTL_SECONDS = 600.0
_APPELLATION_MISSING_TTL_SECONDS = 60.0


def _appellation_index(store, db):
    """The appellation graph (who calls whom what), cached on the
    store. The table is small and mined incrementally, so a TTL
    refresh is enough; a missing table (mining has not run yet)
    caches as unavailable for a shorter spell."""
    cache = store.__dict__.setdefault("_appellation_cache", {})
    now = perf_counter()
    entry = cache.get("index")
    if entry is not None and now < entry[0]:
        return entry[1]
    try:
        # Load on a dedicated connection: the grounding connection
        # carries the budget's progress handler, and an interrupted
        # load must not poison this cache with "unavailable".
        with store.connect() as load_db:
            index = AppellationIndex.load(load_db)
    except sqlite3.OperationalError:
        index = None
    ttl = _APPELLATION_TTL_SECONDS if index is not None else _APPELLATION_MISSING_TTL_SECONDS
    cache["index"] = (now + ttl, index)
    return index


def _appellation_people(store, db, where, args, normalized, caller_id):
    """People the question addresses by a mined appellation: the
    caller's own name for them first, then the server-wide one.
    Matches are scope-verified exactly like alias matches; an
    appellation that does not verify (or verifies to nobody) simply
    contributes nothing."""
    # Targeted slice first: one indexed lookup for the names the
    # question itself presents, fast even on a cold cache under load
    # (the full index load takes seconds there and blows the whole
    # grounding budget — the タヌキ incident's actual mechanism).
    matched = []
    try:
        targeted = AppellationIndex.load_for_text(db, normalized)
    except sqlite3.OperationalError:
        targeted = None
    index = None
    if targeted is not None:
        matched = targeted.match(normalized, caller_id)
    if not matched:
        index = _appellation_index(store, db)
        if index is None:
            return []
        matched = index.match(normalized, caller_id)
    if not matched:
        # Caller-independent fallback: the caller's own usage met no
        # threshold, but a name the whole server uses heavily for one
        # person is still that person (match_global applies stricter
        # evidence floors to compensate). The cheap targeted slice is
        # tried before the full index.
        for source in (targeted, index):
            if source is not None:
                matched = source.match_global(normalized)
                if matched:
                    break
    if not matched:
        return []
    existing = _existing_authors(db, where, args, matched, non_bot=True)
    if not existing:
        existing = _existing_authors(db, where, args, matched, non_bot=False)
    if len(existing) == 1:
        return sorted(existing)
    if len(existing) > 1:
        dominant = _dominant_author(db, where, args, existing)
        if dominant is not None:
            return [dominant]
    return []


def _appellation_claims_form(store, db, where, args, form):
    """Whether this name string is an established appellation of a
    real (non-bot) person in scope: the appellation graph's dominant
    target for the string, verified against the archive. Relay and
    bridge accounts carry their person's names as decorated aliases
    (the さくら#abcd webhooks are aki's relay personas), so a name
    the alias table attributes only to bots may still be a person's
    name in the evidence that matters — and resolving it to the
    relay, alongside or instead of the person, was the aki failure
    family. Bot-named people (vex) are unaffected: their appellation
    target is the bot itself, which fails the non-bot check here."""
    index = _appellation_index(store, db)
    if index is None:
        return False
    targets = index.by_appellation.get(form)
    if not targets:
        return False
    dominant = AppellationIndex._dominant(targets)
    if dominant is None:
        return False
    return bool(_existing_authors(db, where, args, [dominant], non_bot=True))


def _existing_authors(db, where, args, author_ids, *, non_bot):
    """Subset of author_ids with at least one in-scope message, checked
    with one batched query instead of one query per author."""
    ids = [str(author) for author in author_ids]
    found = set()
    for start in range(0, len(ids), 500):
        batch = ids[start:start + 500]
        query = (
            f"SELECT DISTINCT author_id FROM messages m WHERE {where} "
            f"AND author_id IN ({','.join('?' for _ in batch)})"
        )
        if non_bot:
            query += " AND is_bot = 0"
        found.update(row[0] for row in db.execute(query, [*args, *batch]))
    return found


def _dominant_author(db, where, args, author_ids):
    """One author whose in-scope volume dwarfs the same-name others
    (the main さくら#abcd account has 15x the messages of the next
    bridge account): the name then means that author, not an ambiguity.
    Requires a clear margin so genuinely shared names stay ambiguous."""
    ids = [str(author) for author in author_ids]
    rows = db.execute(
        f"SELECT author_id, count(*) AS total FROM messages m WHERE {where} "
        f"AND author_id IN ({','.join('?' for _ in ids)}) "
        "GROUP BY author_id ORDER BY total DESC",
        [*args, *ids],
    ).fetchall()
    if len(rows) < 2:
        return None
    top, second = rows[0], rows[1]
    if top[1] >= 50 and top[1] >= 5 * second[1]:
        return top[0]
    return None


def resolve_people(store, channel_ids, text, caller_id, message_id=None, seed=None,
                   carryover=None, deadline=None):
    """Resolve the people a question is about, in priority order.

    deadline is an absolute perf_counter() time for the grounding
    route's own budget: the resolution SQL runs under a progress
    handler on this call's dedicated connection and is interrupted
    once the budget is spent, degrading to no people rather than
    eating the dense route's budget too."""
    if not channel_ids:
        return []
    where, args = store._scope(channel_ids, exclude_message_id=message_id)
    salvage = {}
    with store.connect() as db:
        if deadline is not None:
            db.set_progress_handler(
                lambda: 1 if perf_counter() > deadline else 0, 1000
            )
        try:
            return _resolve_people_db(
                db, store, where, args, channel_ids, text, caller_id,
                message_id, seed, carryover, deadline, salvage=salvage,
            )
        except sqlite3.OperationalError as error:
            if deadline is not None and "interrupted" in str(error).lower():
                verified = salvage.get("people") or []
                if verified:
                    # The budget interrupt must not throw away people
                    # already verified before it fired (the タヌキ
                    # incident: a mined appellation had matched, then
                    # the alias snapshot load was interrupted and the
                    # whole resolution degraded to nobody).
                    logger.info(
                        "Person grounding exceeded its budget; "
                        "keeping %d verified people", len(verified))
                    return verified
                logger.info("Person grounding exceeded its budget; resolving to nobody")
                return []
            raise
        finally:
            if deadline is not None:
                db.set_progress_handler(None, 0)


def _resolve_people_db(db, store, where, args, channel_ids, text, caller_id,
                       message_id, seed, carryover, deadline=None, salvage=None):
    def over_budget():
        # Explicit checkpoints complement the progress handler: short
        # queries can finish between its callbacks, so a spent budget
        # is also honoured at each phase boundary.
        return deadline is not None and perf_counter() > deadline

    if over_budget():
        return []
    mentions = re.findall(r"<@!?(\d+)>", text)
    if mentions:
        wanted = list(dict.fromkeys(mentions))
        existing = _existing_authors(db, where, args, wanted, non_bot=True)
        return [value for value in wanted if value in existing]
    if seed is not None and str(seed) != str(caller_id):
        # The conversation layer grounded a demonstrative (この人物…)
        # to this person from the reply chain or the live transcript.
        # Trust it when the person has in-scope messages (bot authors
        # included: a decorated bot can be the referent); otherwise
        # fall through to the text-only resolution below.
        if _existing_authors(db, where, args, [str(seed)], non_bot=False):
            return [str(seed)]
    if caller_id is not None and _SELF.search(text):
        return [str(caller_id)]
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if over_budget():
        return []
    # Appellations (mined nicknames) are checked before the alias
    # snapshot loads: a caller-conditioned appellation is stronger
    # evidence than a display-name substring, and when it resolves,
    # the turn never pays for the snapshot on the alias path's behalf
    # alone. Verified matches merge into every later return.
    appellation_people = _appellation_people(
        store, db, where, args, normalized, caller_id)
    if salvage is not None and appellation_people:
        # Verified before the expensive snapshot load: these survive
        # a budget interrupt fired during any later phase.
        salvage["people"] = list(appellation_people)
    snapshot = _alias_snapshot(store, channel_ids, db)
    # Group matches by the name that hit (full alias or its decorated
    # core): when one name maps to several people and the caller is
    # among them, the caller means themselves, so that name resolves
    # to the caller alone instead of going ambiguous.
    matched = {}
    cores = snapshot.core_names()
    for (alias, author_id), core in zip(snapshot.rows, cores, strict=True):
        for form in (alias, core):
            if form is None or len(form) < 2 or form not in normalized:
                continue
            pattern = re.escape(form)
            if form.isascii():
                pattern = r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])"
            elif len(form) == 2:
                # A two-character Japanese alias must sit at a name
                # position, not inside a longer word: the alias んな
                # (from some decorated display name) matched every
                # 「どんな人？」 question and dragged a stranger into
                # them as a phantom second candidate. Real two-char
                # names （のの) are still matched at the start of the
                # text or after a delimiter, where names occur.
                pattern = r"(?<![ぁ-んァ-ヶー一-龥a-zA-Z0-9_])" + pattern
            if re.search(pattern, normalized):
                matched.setdefault(form, set()).add(author_id)
                # No break: a row whose full decorated name appears in
                # the text ("-aki-って") must ALSO join its bare core's
                # bucket ("aki" matches inside the decoration). With
                # the break, the person himself was missing from the
                # core bucket while his alt and his relay personas —
                # whose full names do not appear — filled it, so the
                # core form resolved to the alt alone and the dominance
                # collapse never saw the person it should collapse to.
    resolved = set()
    for form, form_ids in matched.items():
        if caller_id is not None and str(caller_id) in form_ids and len(form_ids) > 1:
            form_ids = {str(caller_id)}
        existing = _existing_authors(db, where, args, form_ids, non_bot=True)
        if not existing:
            # The name may belong only to bot/bridge accounts (the
            # decorated さくら#abcd accounts are all webhooks);
            # the token fallback already resolves bot authors for
            # ASCII names, so do the same here before giving up —
            # unless the string is an established appellation of a
            # real person, in which case the bots are that person's
            # relay personas and the name is the person's, not theirs.
            if _appellation_claims_form(store, db, where, args, form):
                continue
            existing = _existing_authors(db, where, args, form_ids, non_bot=False)
        if len(existing) > 1:
            dominant = _dominant_author(db, where, args, existing)
            if dominant is not None:
                existing = {dominant}
        resolved.update(existing)
    if salvage is not None and (resolved or appellation_people):
        salvage["people"] = sorted(resolved | set(appellation_people))
    if resolved:
        return sorted(resolved | set(appellation_people))
    if over_budget():
        return appellation_people
    fallback = _token_fallback_people(db, snapshot, where, args, normalized)
    if fallback:
        return sorted(set(fallback) | set(appellation_people))
    if carryover is not None and not appellation_people:
        # A follow-up with no person signal of its own (どこ住み？
        # after asking about someone) continues the previous turn's
        # subject; the caller supplies the person that turn settled
        # on. Only used when nothing in the text resolved at all.
        if _existing_authors(db, where, args, [str(carryover)], non_bot=False):
            return [str(carryover)]
    return appellation_people


def _alias_token_keys(alias):
    """Short keys a speaker may be referred to by: the leading ASCII token of
    a decorated display name ("vex" from "vexちゃん【公式bot】") and the
    bracket-stripped name without an honorific suffix ("vex" from
    "vexちゃん")."""
    if re.fullmatch(r"[a-z0-9_]+", alias):
        # Plain handle: the only possible key is the alias itself.
        return {alias} if len(alias) >= 3 else set()
    keys = set()
    tokens = re.findall(r"[a-z0-9_]+", alias)
    if tokens and len(tokens[0]) >= 3:
        keys.add(tokens[0])
    core = re.sub(r"【[^】]*】|\[[^\]]*\]|[(（][^)）]*[)）]", "", alias).strip()
    core = re.sub(r"(?:ちゃん|くん|さん|氏|さま)$", "", core)
    if core.isascii() and len(core) >= 3:
        keys.add(core)
    return keys


def _token_fallback_people(db, snapshot, where, args, normalized):
    """Fallback when no full alias appears in the text: match a standalone
    ASCII query token against alias token keys. Unlike the primary path this
    also resolves bot authors (a decorated bot name like Vex's can otherwise
    never be resolved: its full alias never appears verbatim in a question
    and the primary existence check requires non-bot messages). Scope and
    visibility checks are identical; only the name matching is looser."""
    query_tokens = {token for token in re.findall(r"[a-z0-9_]{3,}", normalized)}
    if not query_tokens:
        return []
    candidates = {
        author_id
        for (_, author_id), keys in zip(snapshot.rows, snapshot.token_keys(), strict=True)
        if keys & query_tokens
    }
    if not candidates:
        return []
    return sorted(_existing_authors(db, where, args, candidates, non_bot=False))


# The overview sample is query-independent (the person's longest posts
# per channel/time octile), yet it is selected before the query-driven
# routes for a resolved person and the selection loop takes chunks
# whole. Measured on the complex-QA set (2026-11-26/27): the sample's
# median size was 3,921 chars -- 65% of the 6,000-char material budget
# -- leaving a median of 1,990 chars for every later route, so
# question-specific evidence chunks (median ~1,700 chars, up to ~3,400)
# routinely no longer fit and were dropped whole. Cap the rendered
# sample at 2,000 chars (header + first hits, whole hits only): the
# two observed cases where a sample hit WAS the gold evidence sit at
# 690 and 1,660 cumulative chars, so the cap keeps them while freeing
# roughly 1,900 chars for the query-driven routes.
OVERVIEW_SAMPLE_CHAR_LIMIT = 2000


def _person_chunks(log, person, member_wiki, guild_id, text, message_id, part):
    if part == "overview":
        overview = log.store.collect(channel_ids=log.channel_ids, author=person, limit=8,
                                     exclude_message_id=message_id)
        hits = overview.pop("hits")
        metadata = {key: overview.get(key) for key in (
            "author_ids", "message_count", "first_seen", "last_seen", "channel_count",
        )}
        excerpts = LogResult.from_hits(hits)
        header = ("Speaker overview (sample, not biography): "
                  + json.dumps(metadata, ensure_ascii=False) + "\n")
        while len(hits) > 1 and len(header) + len(excerpts) > OVERVIEW_SAMPLE_CHAR_LIMIT:
            hits = hits[:-1]
            excerpts = LogResult.from_hits(hits)
        result = LogResult(header + excerpts, sources=excerpts.sources)
        # Lets the selection loop drop already-delivered sample messages
        # and rebuild the chunk from the header plus the fresh excerpts.
        result.hits = hits
        result.prefix = header
        yield result
        return
    if (part == "member_wiki" and member_wiki is not None and guild_id is not None
            and person == log.caller_id):
        try:
            pages = member_wiki.for_member(int(person), guild_id).context(text)
            seen = set()
            for page in pages:
                body = page["excerpt"].strip()
                if body and body not in seen:
                    seen.add(body)
                    yield LogResult(f"Member wiki (guild {guild_id}, author {person}), "
                                    f"{page['page']}: {body[:240]}")
        except (OSError, sqlite3.Error, ValueError):
            logger.warning("Optional member wiki context unavailable", exc_info=True)
    if part != "server_wiki":
        return
    seen = set()
    for page in ServerWiki(log.store).lookup("person", person, log.channel_ids):
        if page["id"] != person:
            continue
        for entry in page["entries"]:
            if message_id is not None and any(
                source["url"].rsplit("/", 1)[-1] == str(message_id) for source in entry["sources"]
            ):
                continue
            summary = " ".join(entry["summary"].split())
            if not summary or summary.casefold() in seen:
                continue
            seen.add(summary.casefold())
            sources = tuple(Source(source["url"], title=f"Server Wiki · {person}",
                                   evidence=summary)
                            for source in entry["sources"][:1])
            yield LogResult(f"Server Wiki person {person}: {summary[:240]} "
                            + " ".join(sources), sources=sources)
            if len(seen) == 4:
                return


def _activity_line(label, row):
    created = datetime.fromisoformat(row["created_at"])
    jst = created.astimezone(_JST)
    stamp = (f"{jst.year}年{jst.month}月{jst.day}日 "
             f"{jst.hour}:{jst.minute:02d}（JST）")
    excerpt = " ".join((row["text"] or "").split())[:40]
    line = f"- {label}: {stamp} #{row['channel_name']}「{excerpt}」"
    return line, created


def _person_activity(store, channel_ids, person, *, message_id=None, now=None):
    """Deterministic latest-activity facts for one resolved person.

    Two scoped lookups by author id, newest first — no ranking, no
    sampling: the person's latest message overall, and their latest
    message before today (Japan time). The second entry is what a
    「今日より前は最後いつ話してた？」 question actually asks for, and
    it is exactly what the overview's all-time last_seen cannot say
    once the person has spoken today. Dates are rendered in JST with
    the channel and an excerpt so the model can quote them verbatim
    instead of converting (or inventing) a timestamp."""
    person = str(person)
    now = now or datetime.now(_JST)
    today_start_utc = (
        now.replace(hour=0, minute=0, second=0, microsecond=0)
        .astimezone(UTC)
    )
    with store.connect() as db:
        where, args = store._scope(
            channel_ids, author=(person,), exclude_message_id=message_id)
        latest = db.execute(
            f"SELECT m.created_at, m.channel_name, m.text FROM messages m "
            f"WHERE {where} ORDER BY m.created_at DESC, m.id DESC LIMIT 1",
            args,
        ).fetchone()
        if latest is None:
            return None
        where_before, args_before = store._scope(
            channel_ids, author=(person,), exclude_message_id=message_id,
            before=today_start_utc.isoformat(),
        )
        before_today = db.execute(
            f"SELECT m.created_at, m.channel_name, m.text FROM messages m "
            f"WHERE {where_before} ORDER BY m.created_at DESC, m.id DESC "
            f"LIMIT 1",
            args_before,
        ).fetchone()
    lines = [
        "発言日時（確定情報）: 以下はアーカイブの実レコードから機械的に集計した"
        "この人物の発言日時。サンプルや検索ヒットの日付から推測せず、"
        "「最後に話したのはいつ」型の質問にはこの値で答えること。"
    ]
    latest_line, latest_at = _activity_line("最新の発言", latest)
    lines.append(latest_line)
    if before_today is not None and before_today["created_at"] != latest["created_at"]:
        line, _ = _activity_line(
            f"今日（{now.month}月{now.day}日）より前の最新の発言", before_today)
        lines.append(line)
    elif latest_at.astimezone(_JST).date() < now.date():
        lines.append(
            "- 今日より前の最新の発言も上記の発言（今日はまだ発言していない）")
    return LogResult("\n".join(lines))


def _person_first_activity(store, channel_ids, person, *, message_id=None):
    """Deterministic earliest-activity fact for one resolved person.

    One scoped lookup by author id, oldest first — the mirror of
    _person_activity for the 「初参加／最初の発言はいつ」 shape. The
    result is the earliest message THIS ACCOUNT has in the archive,
    which is not necessarily the person's absolute first participation
    (an account change can predate it) and is not Discord's server-join
    timestamp itself, so the header instructs the model to present it
    with that reservation instead of as an unqualified join date."""
    person = str(person)
    with store.connect() as db:
        where, args = store._scope(
            channel_ids, author=(person,), exclude_message_id=message_id)
        first = db.execute(
            f"SELECT m.created_at, m.channel_name, m.text FROM messages m "
            f"WHERE {where} ORDER BY m.created_at ASC, m.id ASC LIMIT 1",
            args,
        ).fetchone()
    if first is None:
        return None
    lines = [
        "最初の発言（確定情報）: 以下はアーカイブの実レコードから機械的に集計した"
        "このアカウントの最古の発言。検索ヒットやサンプルの日付から初参加日を"
        "組み立てず、「初参加／最初の発言はいつ」型の質問にはこの値で答えること。"
        "ただしこれはDiscordのサーバー参加日時そのものではなく、アーカイブ内で"
        "このアカウントについて確認できる最古の発言であり、アカウント変更等で"
        "それ以前の参加があり得るため、「アーカイブで確認できる最古は」と留保を"
        "付けて答えること。"
    ]
    line, _ = _activity_line("最初の発言", first)
    lines.append(line)
    return LogResult("\n".join(lines))


_QUOTED_PHRASE = re.compile(r"[「『]([^」』]{2,60})[」』]")
_ASCII_RUN = re.compile(r"[A-Za-z0-9]+(?:[ ._-]+[A-Za-z0-9]+)*")


def _event_candidates(text):
    """Distinctive names a question may use for an Server Wiki event: quoted
    phrases and Latin-script runs. Event titles are descriptive Japanese
    sentences, so a verbatim title match is hopeless; the Server Wiki lookup
    instead matches the candidate as a substring of a title, which works
    when the asker names the event. Short lowercase runs (vex, bot) are
    excluded: they substring-match half the catalogue."""
    candidates = [quoted.strip() for quoted in _QUOTED_PHRASE.findall(text)]
    candidates += [
        run for run in _ASCII_RUN.findall(text)
        if len(run) >= 3 and any(char.isalpha() for char in run)
        and (len(run) >= 5 or any(char.isupper() or char.isdigit() for char in run))
    ]
    return [candidate for candidate in dict.fromkeys(candidates) if candidate][:3]


def _event_chunks(log, candidates, message_id):
    """Server Wiki event evidence for a question that names an event, delivered
    with the first-pass context so the model does not have to spend a
    read_server_wiki round on the lookup its guidance otherwise demands."""
    seen = set()
    delivered = 0
    for candidate in candidates:
        if delivered >= 4:
            return
        for page in ServerWiki(log.store).lookup("event", candidate, log.channel_ids):
            for entry in page["entries"]:
                if delivered >= 4:
                    return
                if message_id is not None and any(
                    source["url"].rsplit("/", 1)[-1] == str(message_id)
                    for source in entry["sources"]
                ):
                    continue
                summary = " ".join(entry["summary"].split())
                if not summary or summary.casefold() in seen:
                    continue
                seen.add(summary.casefold())
                title = " ".join(entry["title"].split())
                sources = tuple(Source(source["url"], title=f"Server Wiki · {title}",
                                       evidence=summary)
                                for source in entry["sources"][:1])
                yield LogResult(f"Server Wiki event {title}: {summary[:240]} "
                                + " ".join(sources), sources=sources)
                delivered += 1


# Per-route retrieval budgets (milliseconds). The single shared 850ms
# pool let the lexical stage eat everything and the dense route almost
# never ran (measured: deadline 73%, dense executed 2/44 turns). Each
# route now owns a slice — grounding, lexical, dense (query embedding
# included), aux/fusion — with the remainder held as reserve for chunk
# selection and materialization. pre_retrieve still enforces the total
# through its timeout argument as a safety net.
ROUTE_BUDGETS_MS = {
    # Sized from production measurements, not the design sketch: on the
    # 3.8GB archive a warm resolve_people takes 10-1300ms (an alias
    # snapshot rebuild is seconds), the FTS stage's own soft deadline
    # is 450ms, and a dense query embedding alone can take ~300ms on
    # this CPU. The budgets keep one route from starving the others
    # inside the caller's overall timeout; they are not latency goals.
    "grounding_ms": 400,
    "lexical_ms": 400,
    "dense_ms": 500,
    "aux_ms": 250,
    "reserve_ms": 100,
}

# Budgets for an extension round: when the first pass finishes with
# evidence that is not sufficient, only the routes that came up short
# are re-run, with a larger slice than the first pass gave them. The
# extension is bounded by pre_retrieve's hard cap, not by these values.
EXTENSION_BUDGETS_MS = {
    "grounding_ms": 1200,
    "lexical_ms": 900,
    "dense_ms": 1200,
    "aux_ms": 400,
}


def _prefetch_covers(prefetch_hits, prefetch_subject, people):
    """Whether prefetched evidence already covers the settled person:
    the prefetch was keyed to that person and at least two of its hits
    are theirs, so the dense job would mostly re-fetch the same
    messages."""
    if prefetch_subject is None or len(people) != 1:
        return False
    if str(people[0]) != str(prefetch_subject):
        return False
    return sum(
        1 for hit in prefetch_hits
        if str(getattr(hit, "author_id", "")) == str(prefetch_subject)
    ) >= 2


def _assess_sufficiency(available, people, *, dense_available=True,
                        snapshot_unavailable=False, grounding_cut=False,
                        has_alt_query=False, dense_extended=False,
                        summary_retried=False, github_topic=False,
                        github_retried=False):
    """Judge whether the evidence gathered so far is sufficient, and
    which routes are weak when it is not.

    Sufficiency is about the evidence, not the clock (the hard cap is
    only a safety valve): a settled person's own messages reached the
    result, or several routes agree on a healthy number of hits. The
    weak list names the routes an extension round may re-run — a route
    that cannot change the outcome (no dense backend, a snapshot that
    is not published, grounding that completed cleanly) is never
    listed, so extension rounds only spend time where it can help."""
    authors = set()
    total = set()
    delivered_kinds = set()
    for kind, chunks in available.items():
        for chunk in chunks:
            hits = getattr(chunk, "hits", None) or []
            if hits:
                delivered_kinds.add(kind)
            for hit in hits:
                total.add(hit.id)
                if getattr(hit, "author_id", None) is not None:
                    authors.add(str(hit.author_id))
    count = len(total)
    person = str(people[0]) if len(people) == 1 else None
    target_reached = person is not None and person in authors
    dense_delivered = bool(
        delivered_kinds & {"dense", "context", "author", "prefetch"}
    )
    overview_delivered = "overview" in available
    # A person question is not sufficiently answered until the
    # structured identity line (person_summary) has actually arrived:
    # without it the model cannot tell the question's name and the
    # evidence's author are the same person, and falls back to
    # 「特定できない」 even with the person's messages in hand (the
    # A-5/A-6 pattern). The summary stage is cheap and deterministic,
    # so a missing one is always worth one extension retry — but only
    # one: a person with no users row and no appellations has no
    # summary to give, and retrying forever would tax every turn.
    summary_delivered = "person_summary" in available
    summary_missing = person is not None and not summary_delivered
    sufficient = False
    reasons = []
    if person is not None and target_reached and (count >= 3 or overview_delivered):
        sufficient, reasons = True, ["target_evidence"]
    elif person is None and count >= 4 and dense_delivered:
        sufficient, reasons = True, ["multi_route_evidence"]
    elif count >= 8:
        sufficient, reasons = True, ["ample_evidence"]
    if summary_missing and not summary_retried:
        sufficient = False
        reasons = []
    # The deterministic GitHub block gets the summary's one-retry
    # treatment: on a GitHub-topic turn with a settled person it is
    # the decisive evidence for any account/authorship answer, and
    # the phase-1 job can miss delivery under deadline pressure
    # (on 2026-11-27 08:44 the stage ran, the phase hit its
    # deadline, and no chunk reached the result). A person with no
    # GitHub footprint yields no block however often it is built,
    # so — like the summary — the retry is spent at most once.
    github_missing = (
        github_topic and person is not None
        and "github_facts" not in available and not github_retried
    )
    if github_missing:
        sufficient = False
        reasons = []
    weak = []
    if not sufficient:
        if summary_missing and not summary_retried:
            weak.append("person_summary")
        if github_missing:
            weak.append("github")
        if person is not None and not target_reached:
            weak.append("author")
        if person is None and not people and grounding_cut:
            weak.append("grounding")
        # A "dense" chunk whose dense half was discarded as over
        # budget carries only lexical hits, so a thin result counts
        # as dense-missing too — unless an extension round already
        # re-ran dense at full budget and this is all it found.
        if (dense_available and not snapshot_unavailable
                and (not dense_delivered or count < 3)
                and not (dense_extended and dense_delivered)):
            weak.append("dense")
        if count < 3 and has_alt_query:
            weak.append("lexical")
    return {
        "sufficient": sufficient, "reasons": reasons, "weak": weak,
        "total_hits": count, "target_reached": target_reached,
        "dense_delivered": dense_delivered,
    }


async def pre_retrieve(log, text, *, member_wiki=None, guild_id=None, message_id=None,
                       seed_person=None, carryover_person=None, timeout=0.85, budget=6000,
                       budgets=None, search_text=None, context_query=None,
                       hard_cap=2.5, adaptive=True, prefetch_hits=None,
                       prefetch_subject=None):
    """Add person history and Server Wiki; private member-wiki excerpts are caller-only.

    search_text replaces the question for the archive searches (a
    contextual rewrite that makes a demonstrative/follow-up question
    self-contained); people resolution and event matching keep the
    original text. context_query, when given, drives one extra dense
    search built from the live conversation, so relevant past messages
    surface from the flow of the discussion rather than only from the
    question's own words.

    With route budgets passed and adaptive on, the first pass is only
    the opening move: if the evidence it gathers is not sufficient
    (see _assess_sufficiency), the routes that came up short are
    re-run with larger budgets until the evidence suffices or the
    hard_cap safety valve (seconds, total wall clock) is reached.
    prefetch_hits are archive hits fetched ahead of the question from
    the channel's conversation state; they are delivered as their own
    chunk, count toward sufficiency, and — when keyed to the person
    grounding settles on — stand in for the dense job."""
    started = perf_counter()
    timings = Timings()
    token = CURRENT.set(timings)
    memo_token = LEXICAL_MEMO.set(new_lexical_memo())
    stopped = Event()
    chunks = queue.Queue()
    skipped = []
    failures = []
    people = []
    # Aux (hybrid) gate mode, read per turn: "always" (the v1.29.0
    # default) never closes the gate, so aux fires even when grounding
    # resolves a person; "grounding" restores the pre-v1.29 behaviour
    # where work() closes the gate once people resolve (rollback).
    aux_gate_mode = (
        "grounding"
        if os.environ.get("SERVER_LOG_AUX_GATE", "always").strip().lower()
        == "grounding"
        else "always"
    )
    hybrid_gate = {"allow": True}
    aux_stats = {"fired": False, "hits": 0}
    reply_graph_enabled = (
        os.environ.get("SERVER_LOG_REPLY_GRAPH", "on").strip().lower() != "off"
    )
    reply_graph_stats = {"seeds": 0, "neighbors": 0}
    route_budgets = None
    grounding_deadline = None
    if budgets is not None:
        route_budgets = {
            "lexical_ms": budgets.get("lexical_ms"),
            "dense_ms": budgets.get("dense_ms"),
            "aux_ms": budgets.get("aux_ms"),
        }
        if budgets.get("grounding_ms") is not None:
            grounding_deadline = started + budgets["grounding_ms"] / 1000.0
    effective_text = search_text or text
    # First-activity detection (v1.31.0): the v1.28.0 block keyed on
    # the latest message's own wording only, so a follow-up carrying
    # the topic in the conversation instead of in its own words never
    # re-armed it, and the model restated the wrong date the
    # transcript held (the bot's own pre-fix answer). The topic
    # signal also reads the rewritten query and the context query.
    (first_activity_direct, first_activity_topic,
     first_activity_context_dependent) = _first_activity_signals(
        text, search_text, context_query)
    fa_subject = {"source": None, "person": None}
    fa_scheduled = set()
    # GitHub detection (v1.37.0): the v1.35.0 block keyed on the
    # latest message's own wording only, so an elliptical follow-up
    # that only names a new person (「タヌキのは」) never armed it
    # even though the conversation was settled on GitHub and the
    # person resolved — the model then declined to answer rather
    # than mix in the previous person's account. The topic signal
    # also reads the rewritten query and the context query, the
    # same follow-the-topic route as the first-activity block.
    (github_direct, github_topic,
     github_context_dependent) = _github_signals(
        text, search_text, context_query)
    github_state = {"retried": False}

    def first_activity_fallback_subject():
        # The block's subject when grounding resolved nobody (in the
        # incident it hit its deadline under load on both follow-up
        # turns, and the seed branch of resolve_people skips a seed
        # that IS the caller, so a marker-less follow-up about the
        # caller's own first participation had no path to a subject):
        # the conversation's own pointers first — the person the
        # reply chain grounded, then the previous turn's subject —
        # and, for a context-dependent turn, the caller, whose own
        # first participation is what such a follow-up continues.
        # A direct question about an unknown third party (not
        # context-dependent, no pointers) gets no fallback: there is
        # still no subject to describe.
        if seed_person is not None:
            return str(seed_person), "seed"
        if carryover_person is not None:
            return str(carryover_person), "carryover"
        if first_activity_context_dependent and log.caller_id is not None:
            return str(log.caller_id), "caller"
        return None, None

    # Evidence-verification gate (claim_check.py): a turn shaped like
    # a dispute over the record gets a targeted verification search,
    # so the model answers from the archive instead of conceding an
    # unchecked claim (the vex-duel #19/#30 surrender pattern).
    claim_dispute = detect_claim_dispute(text)
    # Filled by claim_part (author-scoped pass only): which disputed
    # keywords occur in the speaker's own messages. Consumed by the
    # Assistant's deterministic floor via the diagnostics dict.
    claim_author_hits = []
    prefetch_list = [
        hit for hit in (prefetch_hits or []) if str(hit.id) != str(message_id)
    ]

    def publish(kind, chunk):
        if chunk and not stopped.is_set():
            chunks.put((kind, chunk))

    def search(query_text, author, limit, lexical_only=False, kind=None,
               budgets_override=None, capture=None):
        with stage("lexical_search" if lexical_only else "dense_search"):
            try:
                hits = log.store.search(query_text[:2000], channel_ids=log.channel_ids,
                                        author=author, limit=limit, require_dense=not lexical_only,
                                        infer_author=author is not None,
                                        exclude_message_id=message_id,
                                        lexical_only=lexical_only, snapshot_only=not lexical_only,
                                        allow_hybrid=hybrid_gate["allow"],
                                        route_budgets=(budgets_override
                                                       if budgets_override is not None
                                                       else route_budgets))
            except DenseSnapshotUnavailable:
                skipped.append("dense_snapshot")
                if capture is not None:
                    capture["hits"] = []
                return
            # Aux instrumentation carried on the fusion path's result
            # (plain lists on early-return paths default to no aux).
            if getattr(hits, "aux_fired", False):
                aux_stats["fired"] = True
            aux_stats["hits"] += int(getattr(hits, "aux_hits", 0) or 0)
            if capture is not None:
                capture["hits"] = hits
                return
            if hits:
                if kind is None:
                    kind = "lexical" if lexical_only else "author" if author else "dense"
                chunk = LogResult.from_hits(hits)
                chunk.hits = hits
                publish(kind, chunk)

    async def run(name, function, *args, dense=False):
        slots, pool = (_DENSE_SLOTS, _DENSE_POOL) if dense else (_SQL_SLOTS, _SQL_POOL)
        if name in ("server_wiki", "server_wiki_event"):
            slots, pool = _WIKI_SLOTS, _WIKI_POOL
        elif name == "member_wiki":
            slots, pool = _MEMBER_WIKI_SLOTS, _MEMBER_WIKI_POOL
        if not slots.acquire(blocking=False):
            skipped.append(name)
            return None
        def execute():
            try:
                if not stopped.is_set():
                    with stage(name):
                        return function(*args)
            except Exception as error:
                failures.append({"stage": name, "error": type(error).__name__})
                logger.warning("Automatic retrieval stage %s unavailable", name, exc_info=True)
            finally:
                slots.release()
        return await asyncio.shield(asyncio.get_running_loop().run_in_executor(
            pool, copy_context().run, execute,
        ))

    def person_part(person, part):
        for chunk in _person_chunks(log, person, member_wiki, guild_id, text, message_id, part):
            publish(part, chunk)

    def summary_part(person):
        # The structured identity/appellation line for a resolved
        # person (Discord name, the name most people use, the name
        # this caller uses), so generation can answer 「Xって知って
        # る？」 about them directly instead of asking back.
        try:
            with log.store.connect() as db:
                index = _appellation_index(log.store, db)
                summary = build_person_summary(db, index, person, log.caller_id)
        except (OSError, sqlite3.Error):
            summary = None
            index = None
        if summary:
            publish("person_summary", LogResult(summary))
        if index is not None and _APPELLATION_QUESTION.search(text or ""):
            # 「呼ばれ方を教えて」 asks for the whole list, not the
            # single top name the summary line carries (A-6: a 94k
            # token turn still answered with three names because the
            # graph's list never reached the payload).
            try:
                names = appellation_list(index, person, log.caller_id)
            except Exception:
                names = []
            if names:
                listing = "、".join(
                    f"「{name}」（{count}回）" for name, count in names
                )
                publish("appellations", LogResult(
                    f"呼び名一覧（ログで実際に呼ばれた回数つき）: {listing}"))

    def activity_part(person):
        # Deterministic latest-activity facts for a 「最後いつ話し
        # てた？」 question: the one date-bearing answer the archive
        # can give exactly, computed by author id instead of left to
        # the model's free search (which composed a nonexistent
        # timestamp on 2026-11-25). Cheap scoped SQL, same slot pool
        # as the other person parts.
        try:
            chunk = _person_activity(
                log.store, log.channel_ids, person, message_id=message_id)
        except (OSError, sqlite3.Error):
            chunk = None
        if chunk is not None:
            publish("person_activity", chunk)

    def first_activity_part(person):
        # Deterministic earliest-activity fact for a 「初参加はいつ？」
        # question: the mirror of activity_part for the person's
        # first word instead of their last. Without it the model
        # composed a first-participation date from other people's
        # search hits on 2026-11-26.
        try:
            chunk = _person_first_activity(
                log.store, log.channel_ids, person, message_id=message_id)
        except (OSError, sqlite3.Error):
            chunk = None
        if chunk is not None:
            publish("person_first_activity", chunk)

    def event_part(candidates):
        for chunk in _event_chunks(log, candidates, message_id):
            publish("server_wiki", chunk)

    def books_part():
        # The reading-time library: for development questions only,
        # up to three kept books (title + one-line summary + URL) as
        # one small chunk. books_chunk gates on the question being
        # about development and on the library existing, so ordinary
        # turns pay nothing for this route.
        chunk = books_chunk(log.store.path.with_name("reading.db"), text)
        if chunk:
            publish("books", LogResult(chunk))

    def github_part(person):
        # Deterministic GitHub metadata for a GitHub-shaped question:
        # the resolved person's account and repos (fork-traced), or,
        # with no settled person, the repos the question itself links.
        # The metadata is cache-first in the archive DB, so inside the
        # retrieval budgets this is normally a few local lookups; a
        # cold cache spends a bounded number of API calls. The block
        # exists because the 2026-11-27 incident turned on facts no
        # log search can supply (a fork's holder is not its maker).
        try:
            block = None
            if person is not None:
                block = github_meta.person_github_block(
                    log.store, log.channel_ids, person)
            if block is None:
                block = github_meta.text_github_block(log.store, text)
            if block is None and search_text and search_text != text:
                # A follow-up's rewrite makes it self-contained: a
                # repo URL the demonstrative pointed at (the reply
                # target's link) rides in on the rewritten query
                # even when the raw words carry none.
                block = github_meta.text_github_block(
                    log.store, search_text)
        except Exception:
            logger.warning("GitHub metadata block unavailable", exc_info=True)
            block = None
        if block:
            publish("github_facts", LogResult(block))

    def claim_part():
        # The verification search behind the gate: the claim's
        # keywords against the speaker's own messages first (a
        # self-contradiction in their own words is the decisive
        # evidence), then against the whole archive (who actually
        # said it). Lexical only, so the check stays inside the
        # existing route budgets and works even while the dense
        # snapshot is unavailable. The first chunk carries the
        # header (via the prefix mechanism, like the overview) that
        # marks the claim itself as unverified.
        keywords = claim_keywords(text)
        query = " ".join(keywords) if keywords else text
        header_pending = True
        for author in (log.caller_id, None):
            try:
                hits = log.store.search(
                    query[:2000], channel_ids=log.channel_ids,
                    author=author, limit=5, require_dense=False,
                    infer_author=author is not None,
                    exclude_message_id=message_id,
                    lexical_only=True, snapshot_only=False,
                    allow_hybrid=False,
                    # The store's default 450 ms lexical deadline is
                    # calibrated for question-shaped queries; a claim
                    # query's bigrams chunk into several scans and the
                    # decisive (rare-term) chunk may sit behind common
                    # ones, so the gate gets a larger — still bounded —
                    # share of pre_retrieve's hard cap.
                    route_budgets={"lexical_ms": 1200},
                )
            except Exception:
                logger.warning(
                    "Claim check search unavailable", exc_info=True)
                return
            if hits:
                if author is not None:
                    for keyword in keywords:
                        matching = [h for h in hits if keyword in h.text]
                        for hit in matching[:2]:
                            entry = {
                                "keyword": keyword,
                                "date": str(hit.date)[:10],
                                "text": hit.text[:80],
                            }
                            # The search can surface the same message
                            # twice (chunk overlap); one quote per
                            # distinct text is enough.
                            if entry not in claim_author_hits:
                                claim_author_hits.append(entry)
                        if len({h["keyword"]
                                for h in claim_author_hits}) >= 3:
                            break
                chunk = LogResult.from_hits(hits)
                chunk.hits = hits
                if header_pending:
                    # Exactly one header, on the first chunk that
                    # actually delivers evidence.
                    chunk.prefix = CLAIM_CHECK_HEADER
                    header_pending = False
                publish("claim_check", chunk)

    def graph_part():
        # Reply-graph expansion (v1.29.0): the fused top hits
        # (REPLY_GRAPH_SEED_LIMIT) are seeds; each seed's parent (its
        # reply_to) and its children
        # (messages whose reply_to points at the seed) are delivered
        # as one extra chunk, so a hit's conversational context comes
        # with it instead of depending on the rankers surfacing the
        # neighbouring messages on their own.
        if not reply_graph_enabled:
            return
        capture = {}
        search(effective_text, None, REPLY_GRAPH_SEED_LIMIT, capture=capture)
        seed_hits = list(capture.get("hits") or [])
        reply_graph_stats["seeds"] = len(seed_hits)
        if not seed_hits:
            return
        seed_ids = [str(hit.id) for hit in seed_hits]
        seed_set = set(seed_ids)
        try:
            with log.store.connect() as db:
                seed_rows = db.execute(
                    "SELECT id, reply_to FROM messages "
                    f"WHERE id IN ({','.join('?' for _ in seed_ids)})",
                    seed_ids,
                ).fetchall()
                reply_by_seed = {
                    str(row["id"]): (
                        str(row["reply_to"]) if row["reply_to"] else None
                    )
                    for row in seed_rows
                }
                parent_ids = sorted({
                    parent for parent in reply_by_seed.values()
                    if parent and parent not in seed_set
                    and parent != str(message_id)
                })
                where, args = log.store._scope(
                    log.channel_ids, exclude_message_id=message_id)
                parent_by_id = {}
                if parent_ids:
                    parent_by_id = {
                        str(row["id"]): row
                        for row in db.execute(
                            f"SELECT m.* FROM messages m WHERE {where} "
                            f"AND m.id IN ({','.join('?' for _ in parent_ids)})",
                            [*args, *parent_ids],
                        ).fetchall()
                    }
                child_rows = db.execute(
                    f"SELECT m.* FROM messages m WHERE {where} "
                    f"AND m.reply_to IN ({','.join('?' for _ in seed_ids)}) "
                    "ORDER BY m.created_at, m.id",
                    [*args, *seed_ids],
                ).fetchall()
        except (OSError, sqlite3.Error):
            logger.warning("Reply graph expansion unavailable", exc_info=True)
            return
        children_by_seed = {}
        for row in child_rows:
            if str(row["id"]) in seed_set:
                continue
            children_by_seed.setdefault(str(row["reply_to"]), []).append(row)
        neighbors = []
        seen = set(seed_set)
        if message_id is not None:
            seen.add(str(message_id))
        for seed_id in seed_ids:
            if len(neighbors) >= REPLY_GRAPH_MAX_NEIGHBORS:
                break
            parent_id = reply_by_seed.get(seed_id)
            if (
                parent_id and parent_id not in seen
                and parent_id in parent_by_id
            ):
                neighbors.append(_log_hit(parent_by_id[parent_id]))
                seen.add(parent_id)
                if len(neighbors) >= REPLY_GRAPH_MAX_NEIGHBORS:
                    break
            for row in children_by_seed.get(seed_id, []):
                child_id = str(row["id"])
                if child_id in seen:
                    continue
                neighbors.append(_log_hit(row))
                seen.add(child_id)
                if len(neighbors) >= REPLY_GRAPH_MAX_NEIGHBORS:
                    break
        reply_graph_stats["neighbors"] = len(neighbors)
        if not neighbors:
            return
        chunk = LogResult.from_hits(neighbors)
        chunk.hits = neighbors
        chunk.prefix = (
            "Reply graph expansion: parent and reply messages of the "
            "search hits above, added by reply-graph expansion.\n"
        )
        publish("reply_graph", chunk)

    def dense_search(people):
        search(effective_text, None, 3 if len(people) == 1 else 8)
        if len(people) == 1 and not stopped.is_set():
            # Five of the person's own messages: the wider author sample
            # is what the model otherwise fetches with a follow-up
            # search_server_log round; the budget gate in the selection
            # loop still decides how much of it is delivered.
            search(effective_text, people[0], 5)
        if context_query and not stopped.is_set():
            # Context-linked recall (the Sakana/vex pattern): one dense
            # search whose query is the live conversation itself, so
            # past messages the discussion is circling around surface
            # even when the question never names them. It shares the
            # dense route's budget inside store.search.
            search(context_query, None, 3, kind="context")

    def dense_search_ext():
        # Extension re-run of the dense route alone, with the larger
        # extension budgets: the first pass's dense result was missing
        # or discarded as over-budget, and the lexical half of the call
        # is served from this turn's memo, so the time goes to the
        # query embedding and the snapshot scan.
        search(effective_text, None, 8, budgets_override=EXTENSION_BUDGETS_MS)
        if context_query and not stopped.is_set():
            search(context_query, None, 3, kind="context",
                   budgets_override=EXTENSION_BUDGETS_MS)

    def author_search_ext(person):
        if "dense_snapshot" in skipped or log.store._dense is None:
            # No snapshot to fuse against: an author-scoped lexical
            # search still delivers the person's own words.
            search(effective_text, person, 5, lexical_only=True, kind="author",
                   budgets_override=EXTENSION_BUDGETS_MS)
        else:
            search(effective_text, person, 5,
                   budgets_override=EXTENSION_BUDGETS_MS)

    def alt_query():
        """A query the first pass has not already run lexically: the
        original wording when the rewrite drove the first pass, else
        the conversation-context query."""
        if search_text and search_text != text:
            return text
        if context_query:
            return context_query
        return None

    def drain(target):
        while True:
            try:
                kind, chunk = chunks.get_nowait()
            except queue.Empty:
                return target
            target.setdefault(kind, []).append(chunk)

    summary_state = {"retried": False}
    claim_state = {"retried": False}

    async def extension_round(weak):
        nonlocal people
        jobs = []
        if (claim_dispute and not claim_state["retried"]
                and "claim_check" not in available):
            # The verification search missed the first pass (its
            # unchunked lexical scan is the slowest job here under
            # load); it is the turn's decisive evidence, so it gets
            # the same one-retry treatment as the person summary.
            claim_state["retried"] = True
            jobs.append(run("claim_check", claim_part))
        if (github_topic and not github_state["retried"]
                and "github_facts" not in available and len(people) == 1):
            # The GitHub block missed the first pass (the phase-1
            # job can miss delivery under deadline pressure); on a
            # GitHub-topic turn it is the decisive evidence, so it
            # gets the same one-retry treatment as the claim check.
            github_state["retried"] = True
            jobs.append(run("github", github_part, people[0]))
        if "person_summary" in weak and len(people) == 1:
            # The identity line did not survive the first pass (slot
            # contention or the phase deadline); fetch it on its own.
            summary_state["retried"] = True
            jobs.append(run("person_summary", summary_part, people[0]))
        if "grounding" in weak:
            retry = await run(
                "people_ext", resolve_people, log.store, log.channel_ids,
                text, log.caller_id, message_id, seed_person, carryover_person,
                perf_counter() + EXTENSION_BUDGETS_MS["grounding_ms"] / 1000.0,
            ) or []
            if retry:
                people = retry
                if aux_gate_mode == "grounding":
                    hybrid_gate["allow"] = False
                if len(people) == 1:
                    jobs.append(run("dense_ext", author_search_ext, people[0],
                                    dense=True))
                    jobs.append(run("person_summary", summary_part, people[0]))
                    if github_topic and "github_facts" not in available:
                        # Grounding settled only on the retry, so the
                        # first pass built at most the text block:
                        # fetch the newly resolved person's block.
                        jobs.append(run("github", github_part, people[0]))
                    if _ACTIVITY_QUESTION.search(text or ""):
                        jobs.append(run("person_activity", activity_part,
                                        people[0]))
                    if first_activity_topic and \
                            "person_first_activity" not in available:
                        if fa_subject["source"] is None or \
                                fa_subject["person"] == people[0]:
                            fa_subject["source"] = "resolved"
                            fa_subject["person"] = people[0]
                        jobs.append(run("person_first_activity",
                                        first_activity_part, people[0]))
                    jobs.extend(run(part, person_part, people[0], part)
                                for part in ("overview", "server_wiki", "member_wiki"))
                else:
                    publish("ambiguity", LogResult(
                        "Ambiguous people; request an explicit Discord ID: "
                        + ", ".join(people[:8])))
        if "dense" in weak and log.store._dense is not None and len(people) != 1:
            jobs.append(run("dense_ext", dense_search_ext, dense=True))
        if "author" in weak and len(people) == 1:
            jobs.append(run("dense_ext", author_search_ext, people[0], dense=True))
            if "overview" not in available:
                jobs.append(run("overview", person_part, people[0], "overview"))
        if "lexical" in weak:
            alt = alt_query()
            if alt:
                jobs.append(run("lexical_ext", search, alt, None, 8, True, None,
                                EXTENSION_BUDGETS_MS))
        if jobs:
            await asyncio.gather(*jobs)

    async def work():
        nonlocal people
        lexical = asyncio.create_task(
            run("lexical_fallback", search, effective_text, None, 8, True)
        )
        early = []
        if first_activity_topic and "<@" not in (text or "") and (
                not first_activity_direct or _SELF.search(text or "")):
            # Speculative delivery for the conversational shapes: the
            # subject comes from the conversation's pointers or the
            # caller, never from grounding, so the job is scheduled
            # alongside the lexical fallback instead of after the
            # people await — under load that await eats the whole
            # phase budget (both incident follow-ups hit the grounding
            # deadline), and a job scheduled only after it would never
            # run. run() shields the executor call, so the chunk still
            # lands if the phase is cut short. An explicit @-mention
            # in the text keeps grounding authoritative instead: no
            # speculative delivery then.
            subject, source = first_activity_fallback_subject()
            if subject is not None:
                fa_subject["source"] = source
                fa_subject["person"] = subject
                fa_scheduled.add(subject)
                early.append(asyncio.create_task(
                    run("person_summary", summary_part, subject)))
                early.append(asyncio.create_task(
                    run("person_first_activity", first_activity_part, subject)))
        try:
            people = await run("people", resolve_people, log.store, log.channel_ids,
                               text, log.caller_id, message_id, seed_person,
                               carryover_person, grounding_deadline) or []
            if people and aux_gate_mode == "grounding":
                hybrid_gate["allow"] = False
            jobs = [lexical, *early]
            jobs.append(run("books", books_part))
            if github_topic:
                # The resolved person's block, fetched for THIS
                # person in THIS turn — including when the GitHub
                # topic is carried by the conversation rather than
                # the latest words (「タヌキのは」). With nobody
                # settled, github_part falls back to the repos the
                # question itself links.
                jobs.append(run(
                    "github", github_part,
                    people[0] if len(people) == 1 else None))
            if claim_dispute:
                jobs.append(run("claim_check", claim_part))
            if log.store._dense is not None:
                if _prefetch_covers(prefetch_list, prefetch_subject, people):
                    # The prefetch already delivered this person's
                    # evidence; the dense job would re-fetch the same
                    # messages and spend the budget doing it.
                    skipped.append("dense_prefetch")
                else:
                    jobs.append(run("dense", dense_search, people, dense=True))
            if reply_graph_enabled and log.store._dense is not None:
                jobs.append(run("reply_graph", graph_part, dense=True))
            if len(people) == 1:
                if people[0] not in fa_scheduled:
                    jobs.append(run("person_summary", summary_part, people[0]))
                if _ACTIVITY_QUESTION.search(text or ""):
                    jobs.append(run("person_activity", activity_part,
                                    people[0]))
                if first_activity_topic and people[0] not in fa_scheduled:
                    fa_subject["source"] = "resolved"
                    fa_subject["person"] = people[0]
                    jobs.append(run("person_first_activity",
                                    first_activity_part, people[0]))
                jobs.extend(run(part, person_part, people[0], part)
                            for part in ("overview", "server_wiki", "member_wiki"))
            elif people:
                publish("ambiguity", LogResult("Ambiguous people; request an explicit Discord ID: "
                                               + ", ".join(people[:8])))
            elif first_activity_topic and fa_subject["source"] is None:
                # Grounding resolved nobody, but the conversation is
                # settled on a first-activity topic: deliver the
                # deterministic fact for the fallback subject instead
                # of leaving the model to restate whatever date the
                # transcript happens to carry. The person's summary
                # rides along so the fact is attributed to a named
                # account, not floated as a bare date. (The speculative
                # early path above covers most shapes; this branch
                # remains for direct, context-dependent shapes the
                # early path leaves to grounding first.)
                subject, source = first_activity_fallback_subject()
                if subject is not None:
                    fa_subject["source"] = source
                    fa_subject["person"] = subject
                    jobs.append(run("person_summary", summary_part, subject))
                    jobs.append(run("person_first_activity",
                                    first_activity_part, subject))
            if len(people) != 1:
                candidates = _event_candidates(text)
                if candidates:
                    jobs.append(run("server_wiki_event", event_part, candidates))
            await asyncio.gather(*jobs)
        finally:
            lexical.cancel()
            for task in early:
                task.cancel()

    status = "complete"
    phase1_status = None
    verdict = None
    extended_routes = []
    extension_rounds = 0
    cap_hit = False
    available = {}
    try:
        if not isinstance(log.store, ServerLog) or not log.channel_ids:
            status = "empty_scope"
        else:
            if prefetch_list:
                prefetch_chunk = LogResult.from_hits(prefetch_list)
                prefetch_chunk.hits = prefetch_list
                publish("prefetch", prefetch_chunk)
            try:
                await asyncio.wait_for(work(), timeout=min(max(timeout, 0), 2))
                phase1_status = "complete"
            except TimeoutError:
                phase1_status = "deadline"
            status = phase1_status
            if adaptive and budgets is not None:
                # Adaptive extension: while the evidence is not
                # sufficient and the hard cap leaves room, re-run only
                # the weak routes. The cap is a safety valve; the
                # sufficiency verdict is what normally stops the loop.
                while extension_rounds < 2:
                    drain(available)
                    stage_values, _ = timings.snapshot()
                    grounding_cut = (
                        phase1_status == "deadline"
                        or (budgets.get("grounding_ms") is not None
                            and stage_values.get("people", 0)
                            >= budgets["grounding_ms"] * 0.95)
                    )
                    verdict = _assess_sufficiency(
                        available, people,
                        dense_available=log.store._dense is not None,
                        snapshot_unavailable="dense_snapshot" in skipped,
                        grounding_cut=grounding_cut,
                        has_alt_query=bool(alt_query()),
                        dense_extended=bool(
                            {"dense", "author"} & set(extended_routes)),
                        summary_retried=summary_state["retried"],
                        github_topic=github_topic,
                        github_retried=github_state["retried"],
                    )
                    claim_pending = (
                        claim_dispute and not claim_state["retried"]
                        and "claim_check" not in available
                    )
                    if (verdict["sufficient"] or not verdict["weak"]) \
                            and not claim_pending:
                        break
                    remaining = started + hard_cap - perf_counter() - 0.1
                    if remaining <= 0.05:
                        cap_hit = True
                        break
                    stopped.clear()
                    try:
                        await asyncio.wait_for(
                            extension_round(verdict["weak"]), timeout=remaining)
                    except TimeoutError:
                        cap_hit = True
                    finally:
                        stopped.set()
                    extended_routes.extend(verdict["weak"])
                    extension_rounds += 1
                drain(available)
                verdict = _assess_sufficiency(
                    available, people,
                    dense_available=log.store._dense is not None,
                    snapshot_unavailable="dense_snapshot" in skipped,
                    grounding_cut=phase1_status == "deadline",
                    has_alt_query=bool(alt_query()),
                    summary_retried=summary_state["retried"],
                    github_topic=github_topic,
                    github_retried=github_state["retried"],
                )
                if verdict["sufficient"]:
                    status = ("complete" if extension_rounds == 0
                              and phase1_status == "complete" else "extended")
                elif cap_hit or (phase1_status == "deadline" and extension_rounds == 0):
                    status = "deadline"
                elif extension_rounds > 0:
                    status = "extended"
                else:
                    status = "complete"
    except asyncio.CancelledError:
        status = "cancelled"
    finally:
        stopped.set()
        CURRENT.reset(token)
        LEXICAL_MEMO.reset(memo_token)
    drain(available)
    if "dense" in available:
        available.pop("lexical", None)
    selected, sources, remaining, counts = [], [], budget, {}
    delivered = set()
    if len(people) == 1:
        # A settled person question is answered from who the person
        # is and what they themselves said: the identity line and the
        # person's own messages take the shared budget first, and the
        # generic routes fill what is left. The old order let dense /
        # context chunks (other people's talk about everything) spend
        # the whole 6,000 chars before the person's own evidence was
        # reached, so resolved questions still answered thin (A-5/A-6).
        selection_order = (
            "claim_check", "person_summary", "github_facts", "person_activity",
            "person_first_activity", "appellations", "author",
            "overview", "member_wiki", "server_wiki", "books",
            "ambiguity", "dense", "context", "reply_graph", "prefetch", "lexical",
        )
    else:
        selection_order = (
            "claim_check", "github_facts", "dense", "context", "reply_graph",
            "books", "prefetch",
            "lexical", "ambiguity",
            "person_summary", "person_activity", "person_first_activity",
            "appellations", "overview",
            "member_wiki", "server_wiki", "author",
        )
    for kind in selection_order:
        for chunk in available.get(kind, []):
            hits = getattr(chunk, "hits", None)
            if hits is not None:
                # The same message can surface in several chunks (the
                # dense search, the author-scoped search, the overview
                # sample); deliver each message once and spend the
                # budget on new evidence. Chunks without hits (Server Wiki,
                # member wiki) carry derived text, not the source
                # message, and are exempt.
                fresh = [hit for hit in hits if hit.id not in delivered]
                if kind == "lexical" and len(people) == 1:
                    fresh = fresh[:3]
                if not fresh:
                    continue
                rebuilt = LogResult.from_hits(fresh)
                prefix = getattr(chunk, "prefix", None)
                chunk = (LogResult(prefix + rebuilt, sources=rebuilt.sources)
                         if prefix else rebuilt)
            if len(chunk) + 2 <= remaining:
                selected.append(str(chunk))
                sources.extend(chunk.sources)
                remaining -= len(chunk) + 2
                counts[kind] = counts.get(kind, 0) + 1
                if hits is not None:
                    delivered.update(hit.id for hit in fresh)
    result = LogResult("\n\n".join(selected), sources=sources)
    values, active = timings.snapshot()
    values = {name: 0.0 for name in (
        "people", "overview", "server_wiki", "member_wiki", "query_embedding", "query_lock_wait",
        "cache_lock_wait", "cache_sync", "dense", "lexical",
    )} | values
    result.diagnostics = {
        # The calling turn's correlation id (None outside a turn,
        # e.g. the CLI): lets this line be joined with the turn's
        # Codex/Devin and Assistant completion lines in bot.log.
        "turn_id": current_turn_id(),
        "elapsed_ms": round((perf_counter() - started) * 1000, 2), "status": status,
        "stages_ms": values, "active": active, "skipped": skipped,
        "errors": list(failures),
        "chunks": counts, "chars": len(result), "sources": len(sources),
        "people": [str(person) for person in people],
        "budgets": budgets is not None,
        "rewritten": bool(search_text and search_text != text),
        "context_recall": bool(context_query),
        "adaptive": bool(adaptive and budgets is not None),
        "phase1_status": phase1_status,
        "sufficient": verdict["sufficient"] if verdict else None,
        "weak_routes": verdict["weak"] if verdict else [],
        "extended_routes": extended_routes,
        "extension_rounds": extension_rounds,
        "prefetch_hits": len(prefetch_list),
        "prefetch_subject": str(prefetch_subject) if prefetch_subject else None,
        "prefetch_skip_dense": "dense_prefetch" in skipped,
        "summary_delivered": "person_summary" in counts,
        "activity_question": bool(_ACTIVITY_QUESTION.search(text or "")),
        "activity_delivered": "person_activity" in counts,
        "first_activity_question": first_activity_direct,
        "first_activity_topic": first_activity_topic,
        "first_activity_subject": fa_subject["source"],
        "first_activity_delivered": "person_first_activity" in counts,
        "books_delivered": "books" in counts,
        "github_question": github_direct,
        "github_topic": github_topic,
        "github_delivered": "github_facts" in counts,
        "claim_dispute": claim_dispute,
        "claim_check_delivered": "claim_check" in counts,
        "claim_author_hits": claim_author_hits,
        "person_unresolved": bool(
            not people and _PERSON_QUESTION.search(text or "")
        ),
        "aux_gate_mode": aux_gate_mode,
        "aux_gate_allowed": bool(hybrid_gate["allow"]),
        "aux_fired": bool(aux_stats["fired"]),
        "aux_hits": int(aux_stats["hits"]),
        "reply_graph_seeds": int(reply_graph_stats["seeds"]),
        "reply_graph_neighbors": int(reply_graph_stats["neighbors"]),
        "reply_graph_delivered": "reply_graph" in counts,
    }
    logger.info("Automatic retrieval %s", json.dumps(result.diagnostics, sort_keys=True))
    if status == "cancelled":
        raise asyncio.CancelledError
    return result


if __name__ == "__main__":
    from nezubot_discrod.server_log_context_cli import main

    main()
