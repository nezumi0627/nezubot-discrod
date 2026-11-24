import asyncio
import json
import logging
import queue
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import BoundedSemaphore, Event
from time import perf_counter

from nezubot_discrod.messages import Source
from nezubot_discrod.server_log import (
    LEXICAL_MEMO,
    DenseSnapshotUnavailable,
    ServerLog,
    new_lexical_memo,
)
from nezubot_discrod.server_log_timing import CURRENT, Timings, stage
from nezubot_discrod.server_log_tools import LogResult
from nezubot_discrod.server_wiki import ServerWiki

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
    r"(?:おれ|俺|わたし|私|自分|僕|ぼく|うち)(?:の|について|は|が|を|に|と|とか|って|$)|"
    r"\b(?:me|my|myself)\b", re.IGNORECASE,
)


def _alias_core(alias):
    """The bare name inside a decorated display name: bracketed segments
    (【…】, […], （…）) removed, any ``#suffix`` (さくら#abcd) cut, and a
    trailing honorific dropped. People refer to each other by the core,
    so the primary alias match tries it alongside the full alias."""
    core = re.sub(r"【[^】]*】|\[[^\]]*\]|[(（][^)）]*[)）]", "", alias)
    core = re.split(r"[#＃]", core, maxsplit=1)[0].strip()
    core = re.sub(r"(?:ちゃん|くん|さん|氏|さま)$", "", core).strip()
    return core


class _AliasSnapshot:
    """Distinct (normalized, author_id) alias pairs for one channel set,
    with the fallback token keys computed lazily on first use."""

    __slots__ = ("rows", "_keys", "_cores")

    def __init__(self, rows):
        self.rows = [(normalized, author_id) for normalized, author_id in rows]
        self._keys = None
        self._cores = None

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


def _alias_snapshot(store, channel_ids, db):
    """Alias pairs for the channel set, cached on the store.

    Aliases are only ever inserted (never updated or deleted), so the
    table's MAX(rowid) is a sound invalidation watermark: while it is
    unchanged the cached pairs are exactly what a fresh scan would read,
    and resolution results are identical. The per-turn cost drops from
    two full alias scans plus one existence query per matching alias to
    one watermark probe plus one batched existence query."""
    key = tuple(sorted({str(channel) for channel in channel_ids}))
    cache = store.__dict__.setdefault("_alias_snapshots", {})
    watermark = db.execute("SELECT MAX(rowid) FROM author_aliases").fetchone()[0]
    entry = cache.get(key)
    if entry is not None and entry[0] == watermark:
        return entry[1]
    rows = db.execute(
        "SELECT DISTINCT normalized, author_id FROM author_aliases "
        f"WHERE channel_id IN ({','.join('?' for _ in key)})",
        [*key],
    ).fetchall()
    snapshot = _AliasSnapshot(rows)
    cache[key] = (watermark, snapshot)
    return snapshot


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
    with store.connect() as db:
        if deadline is not None:
            db.set_progress_handler(
                lambda: 1 if perf_counter() > deadline else 0, 1000
            )
        try:
            return _resolve_people_db(
                db, store, where, args, channel_ids, text, caller_id,
                message_id, seed, carryover, deadline,
            )
        except sqlite3.OperationalError as error:
            if deadline is not None and "interrupted" in str(error).lower():
                logger.info("Person grounding exceeded its budget; resolving to nobody")
                return []
            raise
        finally:
            if deadline is not None:
                db.set_progress_handler(None, 0)


def _resolve_people_db(db, store, where, args, channel_ids, text, caller_id,
                       message_id, seed, carryover, deadline=None):
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
            if re.search(pattern, normalized):
                matched.setdefault(form, set()).add(author_id)
                break
    resolved = set()
    for form_ids in matched.values():
        if caller_id is not None and str(caller_id) in form_ids and len(form_ids) > 1:
            form_ids = {str(caller_id)}
        existing = _existing_authors(db, where, args, form_ids, non_bot=True)
        if not existing:
            # The name may belong only to bot/bridge accounts (the
            # decorated さくら#abcd accounts are all webhooks);
            # the token fallback already resolves bot authors for
            # ASCII names, so do the same here before giving up.
            existing = _existing_authors(db, where, args, form_ids, non_bot=False)
        if len(existing) > 1:
            dominant = _dominant_author(db, where, args, existing)
            if dominant is not None:
                existing = {dominant}
        resolved.update(existing)
    if resolved:
        return sorted(resolved)
    if over_budget():
        return []
    fallback = _token_fallback_people(db, snapshot, where, args, normalized)
    if fallback:
        return fallback
    if carryover is not None:
        # A follow-up with no person signal of its own (どこ住み？
        # after asking about someone) continues the previous turn's
        # subject; the caller supplies the person that turn settled
        # on. Only used when nothing in the text resolved at all.
        if _existing_authors(db, where, args, [str(carryover)], non_bot=False):
            return [str(carryover)]
    return []


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


async def pre_retrieve(log, text, *, member_wiki=None, guild_id=None, message_id=None,
                       seed_person=None, carryover_person=None, timeout=0.85, budget=6000,
                       budgets=None, search_text=None, context_query=None):
    """Add person history and Server Wiki; private member-wiki excerpts are caller-only.

    search_text replaces the question for the archive searches (a
    contextual rewrite that makes a demonstrative/follow-up question
    self-contained); people resolution and event matching keep the
    original text. context_query, when given, drives one extra dense
    search built from the live conversation, so relevant past messages
    surface from the flow of the discussion rather than only from the
    question's own words."""
    started = perf_counter()
    timings = Timings()
    token = CURRENT.set(timings)
    memo_token = LEXICAL_MEMO.set(new_lexical_memo())
    stopped = Event()
    chunks = queue.Queue()
    skipped = []
    failures = []
    people = []
    # The hybrid (aux) path is enabled only while grounding has not
    # settled on anyone; work() closes the gate once people resolve.
    hybrid_gate = {"allow": True}
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

    def publish(kind, chunk):
        if chunk and not stopped.is_set():
            chunks.put((kind, chunk))

    def search(query_text, author, limit, lexical_only=False, kind=None):
        with stage("lexical_search" if lexical_only else "dense_search"):
            try:
                hits = log.store.search(query_text[:2000], channel_ids=log.channel_ids,
                                        author=author, limit=limit, require_dense=not lexical_only,
                                        infer_author=author is not None,
                                        exclude_message_id=message_id,
                                        lexical_only=lexical_only, snapshot_only=not lexical_only,
                                        allow_hybrid=hybrid_gate["allow"],
                                        route_budgets=route_budgets)
            except DenseSnapshotUnavailable:
                skipped.append("dense_snapshot")
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

    def event_part(candidates):
        for chunk in _event_chunks(log, candidates, message_id):
            publish("server_wiki", chunk)

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

    async def work():
        nonlocal people
        lexical = asyncio.create_task(
            run("lexical_fallback", search, effective_text, None, 8, True)
        )
        try:
            people = await run("people", resolve_people, log.store, log.channel_ids,
                               text, log.caller_id, message_id, seed_person,
                               carryover_person, grounding_deadline) or []
            if people:
                hybrid_gate["allow"] = False
            jobs = [lexical]
            if log.store._dense is not None:
                jobs.append(run("dense", dense_search, people, dense=True))
            if len(people) == 1:
                jobs.extend(run(part, person_part, people[0], part)
                            for part in ("overview", "server_wiki", "member_wiki"))
            elif people:
                publish("ambiguity", LogResult("Ambiguous people; request an explicit Discord ID: "
                                               + ", ".join(people[:8])))
            if len(people) != 1:
                candidates = _event_candidates(text)
                if candidates:
                    jobs.append(run("server_wiki_event", event_part, candidates))
            await asyncio.gather(*jobs)
        finally:
            lexical.cancel()

    status = "complete"
    try:
        if not isinstance(log.store, ServerLog) or not log.channel_ids:
            status = "empty_scope"
        else:
            await asyncio.wait_for(work(), timeout=min(max(timeout, 0), 2))
    except TimeoutError:
        status = "deadline"
    except asyncio.CancelledError:
        status = "cancelled"
    finally:
        stopped.set()
        CURRENT.reset(token)
        LEXICAL_MEMO.reset(memo_token)
    available = {}
    while not chunks.empty():
        kind, chunk = chunks.get_nowait()
        available.setdefault(kind, []).append(chunk)
    if "dense" in available:
        available.pop("lexical", None)
    selected, sources, remaining, counts = [], [], budget, {}
    delivered = set()
    for kind in ("dense", "context", "lexical", "ambiguity", "overview",
                 "member_wiki", "server_wiki", "author"):
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
        "elapsed_ms": round((perf_counter() - started) * 1000, 2), "status": status,
        "stages_ms": values, "active": active, "skipped": skipped,
        "errors": list(failures),
        "chunks": counts, "chars": len(result), "sources": len(sources),
        "people": [str(person) for person in people],
        "budgets": budgets is not None,
        "rewritten": bool(search_text and search_text != text),
        "context_recall": bool(context_query),
    }
    logger.info("Automatic retrieval %s", json.dumps(result.diagnostics, sort_keys=True))
    if status == "cancelled":
        raise asyncio.CancelledError
    return result


if __name__ == "__main__":
    from nezubot_discrod.server_log_context_cli import main

    main()
