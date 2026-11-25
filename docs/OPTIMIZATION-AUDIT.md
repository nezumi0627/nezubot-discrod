# Optimization Audit — nezubot-discrod

- **Date / basis:** 2026-11-24 (JST). Static audit of the archived `origin/main`
  base `5e34b0c` (#27) as copied into this workspace. No source code was changed
  for this audit; no network calls were made; no live database was available, so
  every latency figure is either (a) measured numbers quoted from
  `docs/rag-benchmark.md` / `HANDOFF.md`, or (b) an explicitly labelled estimate
  derived from the code path. Line numbers refer to the base tree above.
- **Scope read in full:** all of `nezubot_discrod/*.py` (~6,600 lines incl.
  `features/ping.py`), `docs/rag-benchmark.md`, `HANDOFF.md` (known issues),
  `HISTORY.md`, and both pending patches:
  - `pending/web-skip-preretrieval/` — tip `f2c98d5`, "Skip automatic private
    context for explicit web lookups" (below: **Patch W**).
  - `pending/retrieval-snapshots/` — tip `26bb38c`, "WIP: optimize Server Wiki
    validation and background dense snapshots" (below: **Patch R**).
    HANDOFF states Patch R is unreviewed and its full test suite was **not**
    run; Patch W's branch suite passed in the handoff environment
    (1,037 passed) but it has no live-model/Discord verification.
- **Scale assumptions (from HANDOFF / benchmark doc, not re-measured):**
  archive `server_log.db` ≈ 3.7 GB; host inventory in the benchmark doc:
  777,818 archived messages, 642,542 embedding-eligible; the handoff narrative
  says ~1.2 M messages total. Dense matrix at ~640 k eligible × 1,024 dims
  float32 ≈ 2.6 GB resident. Retrieval deadline for automatic pre-retrieval is
  **850 ms** (`server_log_context.py:121`, `timeout=0.85`).

## How the 850 ms deadline is lost (summary of the mechanism)

`pre_retrieve` (`server_log_context.py:120-213`) fans out to thread pools, but
the dense stage's cost is dominated by work that is *not* the ranking itself:

1. `DenseIndex.search` holds one global `cache_lock` for its **entire** body —
   cache synchronisation, coverage, query embedding (ONNX inference), the
   matrix scan, and the SQL re-validation loop (`server_log_dense.py:362-407`).
   Any second dense search — including one from the *next* Discord turn — waits
   for all of that. This is the `cache_lock_wait` stage.
2. Inside the lock, `_sync` runs **in the query path**. Any archive write since
   the last search (and in a live server, messages arrive constantly, and every
   write bumps the dense sequence via triggers, see F5) forces an incremental
   sync before the query can proceed, and clears the whole scope-coverage
   cache. HANDOFF measured `cache_sync` ≈ 677–712 ms and `cache_lock_wait` up
   to 807 ms — i.e. either one alone can consume the deadline.
3. The 850 ms `asyncio.wait_for` does **not** stop the underlying executor
   threads (`asyncio.shield` + `run_in_executor`, `server_log_context.py:155-170`).
   A timed-out dense search keeps holding `cache_lock` (and its pool slot /
   semaphore) until the thread actually finishes, poisoning the next turn.
4. For comparison, ranking itself is comparatively cheap at benchmark scale:
   warm unique-query p50 ≈ 100–113 ms and cached-query p50 ≈ 23–35 ms on the
   20,124-message real store; 550 k-vector synthetic warm p50 ≈ 151 ms
   (`docs/rag-benchmark.md`). The deadline problem is synchronisation and
   serialisation, not BM25/RRF/matmul ranking quality.

## Ranked findings

| # | Finding | Area | Expected impact | Patch R | Patch W |
|---|---------|------|-----------------|---------|---------|
| F1 | Dense `cache_lock` held across sync + embedding + matmul + SQL; timed-out threads keep holding it | RAG latency | Very high | Partial | No |
| F2 | `_sync` in the query path; any write invalidates sync + all coverage cache | RAG latency | Very high | **Yes (main target)** | No |
| F3 | Query embedding shares one embedder lock with the document embedder worker | RAG latency | High | Partial | No |
| F4 | Codex: one global lock serialises all turns; per-turn subprocess spawn, `--version` exec, catalog re-read/re-parse | General / Codex | High (concurrency + per-turn latency) | No | No |
| F5 | Per-message write amplification: 9 dense triggers, FTS delete+insert, ≤5 alias inserts, vector delete, dead `log_state` write, double SHA-256 | DB write load | High (backfill + live ingest) | No | No |
| F6 | `connect()` re-validates guarded paths (resolves + lstats) and opens a fresh SQLite connection on every operation; MemberWiki repeats DDL on every connect | DB / all | High (multiplies every other cost) | No | No |
| F7 | Backfill flushes every 100 messages but commits/cursor use separate connections & transactions; channels backfilled strictly serially | DB write load | High (rebuild time) | No | No |
| F8 | Server Wiki `lookup` reloads and SHA-256-hashes every candidate batch before dedup/page caps are applied | Server Wiki | High per person lookup | **Yes (mostly)** | No |
| F9 | `resolve_people` / `_speaker_query`: `instr()` alias scan + one SQL query per alias / per word boundary | RAG latency | Medium–high | No | No |
| F10 | `collect()` overview: window functions over all matches + O(n²) Python sample selection | RAG latency | Medium–high | No | No |
| F11 | Pre-retrieval runs up to 3 full searches per turn; lexical fallback is computed then usually discarded | RAG latency / CPU | Medium | Partial | Partial (web turns only) |
| F12 | Full-matrix scan with several O(N) NumPy passes per query; single-thread BLAS by design | RAG latency | Medium (grows with corpus) | No | No |
| F13 | Embedder pending-selection is a full archive scan per 1,024-row window; per-row revalidation + double hash per batch row | DB / embedder | Medium (rebuild/ETA) | No | No |
| F14 | Missing SQLite tuning: `synchronous` stays FULL, no `wal_autocheckpoint`/`cache_size`/`mmap_size`; MemberWiki DB is not in WAL at all | DB write load | Medium | No | No |
| F15 | Bot event path: archive awaited before reply handling; bulk-delete = N connections/transactions; per-turn permission recomputation over all channels | General / bot | Medium | No | No |
| F16 | Assistant serialises independent I/O: member-wiki context, then recent-messages, each a separate threaded DB call | General | Low–medium | No | Partial (web turns only) |
| F17 | Web tools: `asyncio.run` + fresh `ClientSession`/connector + uncached DNS on every call | General | Low–medium (web turns) | No | No |
| F18 | Micro-hotspots: `bigrams()` per-token `re.fullmatch`, `_fusion` re-bigramming, Server Wiki build full-corpus digest pass, `_first_object` retry loop, `split_message` per-char UTF-16 encode | General | Low individually | No | No |

---

## 1. DB write load — `server_log.py` ingestion / backfill

### F5 — Per-message write amplification in `_upsert`

- **Where:** `server_log.py:336-355` (trigger creation), `365-433` (`_upsert`),
  `435-438` (`upsert`), `481-493` (`mark_deleted`), `28-38` (`bigrams`).
- **What's slow and why:** one live message (or one backfill row) costs far
  more than one row write:
  - **9 dense-change triggers** are installed (3 tables × INSERT/DELETE/UPDATE).
    A `messages` UPDATE fires statements for both OLD and NEW: each does
    `UPDATE dense_state SET sequence = sequence + 1` **plus** an
    `INSERT … ON CONFLICT` into `dense_changes`. A plain INSERT fires one pair;
    an upsert that also deletes a stale vector (`message_vectors` DELETE) and/or
    inserts a tombstone (`deleted_messages` INSERT) fires several more. Every
    one of these is an extra indexed write and dirties the single-row
    `dense_state` page — a hot page that also serialises writers.
  - **FTS maintenance per row:** `DELETE FROM message_fts WHERE rowid = ?` then
    a fresh `INSERT` of the bigram token string (`server_log.py:424-431`).
    FTS5 delete+insert on a regular (non-contentless) FTS table is one of the
    most expensive per-row operations in this schema, and no
    `INSERT INTO message_fts(message_fts) VALUES('optimize')` maintenance exists
    anywhere, so segment bloat accumulates over ~10⁶ insert/delete cycles.
  - **Aliases:** `_aliases` runs up to 5 `INSERT OR IGNORE` statements per
    message (id, name, username, global name, nickname), each preceded by an
    NFKC/casefold normalisation in Python (`_name`, `server_log.py:363-371, 386-390`).
  - **Redundant hashing:** the unchanged-check compares the full previous row in
    Python, and the vector-invalidation check computes `text_hash(previous)` and
    `text_hash(row)` — two SHA-256 hashes over `author_name + ": " + text` —
    even for brand-new messages where `previous is None` short-circuits only
    part of the work (`server_log.py:415-433`).
  - **Dead write:** `UPDATE log_state SET generation = generation + 1`
    (`server_log.py:433`, also `481-493`, `server_log_dense.py:198`,
    `server_log_vectors.py:142`) — `generation` is never read by production
    code (only tests assert it increases). It is one more hot-row update in
    every write transaction.
  - `upsert()` (live path) wraps each single message in its own connection +
    `BEGIN IMMEDIATE` transaction (`server_log.py:435-438`), so all of the
    above is also one WAL commit (with FULL sync, see F14) per Discord message.
- **Proposed fix:** (1) Collapse the trigger bodies: bump `dense_state` once
  per statement, not per source row, or replace the per-row journal write with
  a cheaper epoch counter plus a periodic journal compaction (the journal is
  already keyed by `message_id`, so multiple bumps per transaction for the same
  id are pure waste — dedupe within `_upsert` by writing the journal once per
  affected id at the Python layer and dropping the `message_vectors` /
  `deleted_messages` triggers in favour of explicit journal writes at the few
  call sites). (2) Skip FTS delete when the rowid is new; batch FTS work during
  backfill (see F7). (3) Compute `text_hash` once per row and reuse it for both
  the comparison and the stored value. (4) Remove the `log_state.generation`
  update or document who consumes it. (5) Add a periodic FTS `optimize`.
- **Expected impact:** estimate 20–40 % fewer SQL statements and WAL bytes per
  ingested message; the bigger win is indirect — fewer sequence bumps per
  transaction means less spurious dense-sync invalidation (F2). No benchmark
  exists for ingest throughput in the repo; this is the first thing to measure
  before/after.
- **Risk:** medium. The dense journal is the correctness backbone of the
  incremental cache (deletions/edits must never be served stale). Any trigger
  change needs the existing `test_server_log*.py` invalidation tests plus a
  cross-instance write test. FTS changes risk ranking drift if tokenisation
  changes — it must not.
- **Pending patches:** neither addresses this. Patch R *depends* on the journal
  semantics staying exactly as they are.

### F6 — `connect()` overhead is paid on every single operation

- **Where:** `server_log.py:224-262` (`_guarded_path` + `connect`),
  `member_wiki.py:132-180` (`_guarded_path` + `connect`).
- **What's slow and why:** `ServerLog.connect()` is a context manager used by
  *every* public method (`upsert`, `search`, `collect`, `read`, `recent`,
  `get_cursor`, `set_cursor`, `mark_deleted`, …). Each call:
  - acquires `_schema_lock` (a threading lock contended by all pool threads),
  - calls `_guarded_path()`, which calls `guard_host_write` (two
    `Path.resolve()` chains, each doing realpath syscalls) for the directory
    **and again for each of the 4 file suffixes** (`""`, `-journal`, `-wal`,
    `-shm`), plus `mkdir`, `lstat` on the directory and on every existing file,
    and `os.geteuid()` checks — roughly 10–20 filesystem syscalls before SQLite
    is even opened;
  - opens a brand-new `sqlite3.connect(..., timeout=30)` (no connection reuse
    anywhere in the codebase).
  A single answered Discord turn opens on the order of 8–12 connections:
  `resolve_people` (1), lexical search (1), dense search (1, plus its internal
  validation queries on the same connection), author search (1), `collect`
  overview (1), Server Wiki lookup (2 — archive + wiki), `recent` (1), member-wiki
  context (1+). `MemberWiki.connect()` is worse: it additionally takes a global
  `_CREATE_LOCK`, attempts an `O_CREAT|O_EXCL` open, and re-executes
  `CREATE TABLE IF NOT EXISTS` on **every** connect, with no equivalent of
  ServerLog's `_schema_ready` fast path.
- **Proposed fix:** cache the guarded-path validation per process (validate
  once at startup / first use; re-validate only on `PermissionError` or on a
  slow timer), keeping the security property (symlink/hardlink checks) at
  open-of-database granularity rather than per-query granularity. Give
  `MemberWiki` the same `_schema_ready` pattern. Consider a small pool of
  read-only connections for the search path (WAL makes this safe) while keeping
  one writer connection discipline for ingest.
- **Expected impact:** removes a fixed per-query tax (estimate single-digit ms
  per connection on local SSD, more under load / on the 3.7 GB DB's host) and,
  more importantly, removes `_schema_lock` + filesystem jitter from the tail
  latency of every retrieval stage inside the 850 ms budget.
- **Risk:** low–medium. The guarded-path checks are a deliberate security
  control (private files, no symlinks into the checkout). Caching must not
  weaken the *first-open* checks, and a swapped symlink after startup would no
  longer be caught per-operation — that trade-off should be explicit and
  owner-approved.
- **Pending patches:** neither addresses this. Patch R's snapshot path still
  goes through `store.connect()` for refresh and validation.

### F7 — Backfill: small flushes, separate cursor transactions, serial channels

- **Where:** `server_log_discord.py:112-162` (`backfill_channel`),
  `164-199` (`backfill_guild`); `server_log.py:440-454` (`upsert_many`),
  `771-786` (`get_cursor` / `set_cursor`).
- **What's slow and why:**
  - `backfill_channel` flushes every **100** messages, but `upsert_many`'s own
    commit batching is 1,000 — so during backfill each flush is its own
    connection + transaction of only 100 rows, and then `set_cursor` opens
    **another** connection and transaction just to write one cursor row.
    Rebuilding ~777 k–1.2 M messages therefore costs ~8–12 k transactions and
    ~8–12 k extra cursor connections, each paying F6's connect overhead and a
    WAL commit.
  - `backfill_guild` visits channels/threads **strictly serially**
    (`await visit(channel)` in loops), including archived-thread enumeration.
    Discord history fetching is I/O-bound; serial visiting leaves most of the
    available rate-limit budget unused and makes a from-scratch rebuild
    (exactly the situation after a DB rebuild) take wall-clock far longer than
    the API requires. (Per-channel ordering/cursor semantics are per-channel,
    so channels are independent.)
  - `message_record()` per message builds embed dicts etc. — minor, but it runs
    in the event-loop thread while iterating `channel.history`, not in the
    worker thread with the DB write.
- **Proposed fix:** fold the cursor update into the same transaction/connection
  as the batch it describes (add an optional `cursor=` parameter to
  `upsert_many`, or a combined `upsert_many_with_cursor`); raise the flush size
  to match the 1,000-row commit batch (or flush on whichever of 1,000 rows /
  ~1 s comes first); run a bounded number of channels concurrently (e.g. 3–4,
  respecting 429 back-off per channel as today). Keep per-channel
  oldest-first order and resume semantics unchanged.
- **Expected impact:** estimate 3–10× faster full rebuild, dominated by fewer
  commits (FULL-sync fsyncs, F14) and parallel history fetching. This is the
  single biggest lever for the planned DB rebuild.
- **Risk:** medium. Cursor-in-same-transaction is strictly *safer* than today
  (today a crash between batch commit and cursor commit re-imports a batch —
  idempotent but wasteful). Concurrency must respect discord.py rate limits;
  start at 2–3 channels and watch 429s.
- **Pending patches:** neither addresses backfill.

### F14 — SQLite tuning left on defaults

- **Where:** `server_log.py:250-276` (only `journal_mode=WAL` at schema init
  and `temp_store=MEMORY` per connection); `member_wiki.py:158-180` (no
  pragmas except `temp_store`; no WAL; default 5 s busy timeout from
  `sqlite3.connect(path)` with no `timeout=`).
- **What's slow and why:** `synchronous` remains at the SQLite default (FULL in
  WAL mode), so every commit — including each 100-message backfill flush and
  each live single-message upsert — pays a full fsync. No `cache_size`,
  `mmap_size`, or `wal_autocheckpoint` is configured, so the 3.7 GB archive is
  read with the default 2 MB page cache and the WAL can grow unchecked between
  default checkpoints. The member-wiki DB is in rollback-journal mode, so a
  wiki write blocks readers entirely and vice versa.
- **Proposed fix:** set `PRAGMA synchronous = NORMAL` for the archive and wiki
  (safe against application crash in WAL mode; only an OS/power crash can lose
  the last transactions — for a rebuildable archive and a derived wiki this is
  the standard trade-off, but it must be an explicit owner decision), raise
  `cache_size` (e.g. −64 MB) and `mmap_size` for the search connections, set an
  explicit `wal_autocheckpoint`, and put `wiki.db` into WAL with
  `timeout=30` like the archive.
- **Expected impact:** commit latency on fsync-bound hosts typically drops
  several-fold with NORMAL; combined with F7 this is the difference between a
  rebuild measured in hours vs. tens of hours on slow virtualised storage.
  Estimate only — measure with the actual host disk.
- **Risk:** low for cache/mmap/autocheckpoint; **durability trade-off** for
  `synchronous=NORMAL` (see above) — flag for owner approval, do not change
  silently.
- **Pending patches:** neither touches pragmas.

---

## 2. RAG retrieval latency — the 850 ms deadline

### F1 — One lock for the whole dense search; deadlines don't release it

- **Where:** `server_log_dense.py:358-407` (`DenseIndex.search`),
  `server_log_context.py:146-170` (pool execution via `asyncio.shield`),
  `182-213` (deadline handling).
- **What's slow and why:** `search()` acquires `self.cache_lock` at entry and
  releases it only in the `finally` at the end. Inside the critical section it
  performs, in order: `_sync` (SQLite reads, hashing, pending-mirror updates),
  `_coverage` (SQLite count on the in-memory pending DB + O(N) NumPy masking),
  `_query_vector` → `encode` (ONNX inference for a cache-miss query), the
  matmul, and a widening loop of SQL validation queries against the archive DB.
  Dense search is therefore globally serialised at an effective concurrency of
  1 despite `_DENSE_POOL` having 2 workers and `_DENSE_SLOTS` being 2
  (`server_log_context.py:21,25`). Worse, when `pre_retrieve`'s 850 ms
  `wait_for` fires, the asyncio side gives up but the executor thread keeps
  running under `asyncio.shield`; it retains both `cache_lock` and its dense
  slot until it finishes naturally. A slow turn thus directly manufactures the
  next turn's `cache_lock_wait` (HANDOFF: up to 807 ms — consistent with
  waiting behind one full sync+embed+scan cycle).
- **Proposed fix:** shrink the critical section to state swap only: sync into a
  *new* state object (or copy-on-write snapshot — the direction Patch R takes)
  and publish it under the lock; do query embedding, matmul, and SQL validation
  against the published immutable state **without** holding the lock. Add a
  generation check so a search that started before a publish can finish on its
  own snapshot. Separately, track abandoned (deadline-exceeded) work: if the
  consumer is gone, the worker should stop at the next stage boundary instead
  of completing a full sync whose result nobody will read.
- **Expected impact:** removes queueing as a deadline factor in steady state;
  `cache_lock_wait` should fall from up-to-807 ms to ~0 ms except during an
  actual snapshot publish (a pointer swap). This and F2 are the two changes
  that make the 850 ms deadline routinely achievable.
- **Risk:** medium–high. Concurrency correctness (torn reads of matrix/ids/
  active flags) is exactly what the current coarse lock prevents by brute
  force. Needs the snapshot tests in Patch R plus stress tests with concurrent
  search + ingest.
- **Pending patches:** **Patch R partially addresses this.** `search_snapshot`
  runs against a previously published snapshot object with its *own*
  `cache_lock`, and `synchronize=False` skips `_sync`. But: the snapshot's
  `search()` still holds its lock across embedding + matmul + validation; the
  live object's `search()` (still used by the model-facing
  `search_server_log` tool, which does not pass `snapshot_only`) is unchanged;
  and refresh itself takes the live `cache_lock` for the whole `_sync`
  (`refresh_snapshot`), so a refresh still blocks live-path searches. The
  abandoned-thread problem is not addressed in either patch.

### F2 — Cache synchronisation inside the query path; coverage cache invalidated by any write

- **Where:** `server_log_dense.py:274-320` (`_sync`), `279-280`
  (coverage clear), `332-356` (`_coverage`); trigger source at
  `server_log.py:336-355`.
- **What's slow and why:** every `DenseIndex.search` first calls `_sync(db)`.
  If the dense sequence moved — which happens on **every** archived message
  insert/edit/delete because of the F5 triggers — the incremental branch:
  loads all changed ids since the cached sequence, issues one
  `pending.execute("DELETE …")` **per changed id** (Python loop, not
  `executemany`), reloads the changed rows with full `SELECT m.*, v.vector …`,
  and for each row `_load` recomputes `text_hash` (SHA-256 over the full text)
  and renormalises the vector. On a busy server the sequence has *always*
  moved, so this is per-query work, and HANDOFF's observed 677–712 ms
  `cache_sync` is what it costs when the changed set is large (e.g. after
  backfill bursts or the embedder publishing a 1,024-row window — each vector
  insert also fires the triggers). Compounding it, `_sync` clears the entire
  `scope_coverage` cache whenever the sequence changes
  (`server_log_dense.py:279-280`), so `_coverage`'s pending-count SQL + NumPy
  masking is also redone per query in live traffic. Cold start is the extreme
  case: the first search after process start streams the **whole archive**
  (`ORDER BY channel_id, created_at, id`, `fetchmany(512)`) and hashes every
  row's text before answering.
- **Proposed fix:** move `_sync` off the query path entirely — a background
  refresher (embedder-worker cadence or sequence-watermark polling) that
  publishes snapshots (Patch R's design); queries read the last published
  snapshot and re-validate candidates against current DB state (Patch R does
  this for snapshot searches via vector-version comparison). Make coverage
  cache invalidation granular (key by scope + sequence bucket) instead of
  clear-all. Batch the pending-mirror deletes with `executemany`. Avoid
  re-hashing unchanged text in `_load` by trusting the stored `text_hash` when
  the row's hash column already matches the vector's hash (hash only when they
  differ — currently `text_hash(row)` is computed unconditionally at
  `server_log_dense.py:238`).
- **Expected impact:** removes the measured 677–712 ms `cache_sync` from the
  query path (Patch R's stated goal); steady-state query latency should return
  to the benchmark's warm numbers (~25 ms cached / ~100 ms unique at 20 k
  scale, ~150 ms at 550 k synthetic scale) plus validation.
- **Risk:** medium. Snapshot staleness is the price: a message archived 100 ms
  ago may be missing from dense results until the next refresh (lexical
  fallback still covers it in `pre_retrieve`'s fusion, and Patch R marks
  results `partial` with `coverage=None` when the snapshot sequence is behind).
  Deletion safety must not depend on the snapshot — Patch R correctly
  re-checks current rows/vectors in SQL before returning an id.
- **Pending patches:** **Patch R is precisely this fix** for the
  `pre_retrieve` path (`snapshot_only=True` there; `search_snapshot` +
  `refresh_snapshot` + `request_refresh` throttled to 1/s and triggered from
  `embed_pending`). Gaps to review before merging: (a) tool-path searches
  (`LogTools.call` → `store.search` without `snapshot_only`) still sync inline;
  (b) until the first `refresh_snapshot()` completes there is *no* snapshot and
  dense pre-retrieval silently degrades to lexical/`DenseSnapshotUnavailable`
  (the diagnose CLI gains `--warmup` for this, but bot startup does not warm
  up); (c) `refresh_snapshot` shallow-copies the index (`copy(self)`), so the
  snapshot shares the matrix, embedder, and `lock` with the live object until a
  `_reserve` reallocates — the immutability invariant is maintained by the new
  `snapshot_mode` append-only logic in `_load`, which is subtle and is the
  part most in need of review; (d) snapshot validation hashes every candidate
  vector blob (`hashlib.sha256(row["vector"])` per candidate, up to `fetch`
  ≥ 200 blobs of 2 KB) on every snapshot search — cheaper than a sync, not
  free; (e) full tests were not run for this branch (HANDOFF).

### F3 — Query embedding waits behind document embedding

- **Where:** `server_log_dense.py:106-130` (`encode` under `self.lock`),
  `131-141` (`_embedding_order` also takes `self.lock` for tokenisation),
  `143-210` (`embed_pending` loop), `322-330` (`_query_vector`).
- **What's slow and why:** one `threading.Lock` guards the FastEmbed model for
  both passage embedding (the background embedder, which encodes batches of 32
  back-to-back while catching up — potentially for hours after a rebuild) and
  query embedding (one text, on the critical path of every dense search).
  A query that arrives mid-batch waits for the batch; a query that arrives
  during catch-up competes with a worker that immediately re-acquires the lock
  for the next batch (no fairness/priority). E5-large single-query inference is
  itself ~80–160 ms (benchmark doc: E5 p50 161 ms standalone; production
  128-token setting ~82–100 ms warm unique) — a large slice of 850 ms even
  without contention. The 128-entry query-vector LRU helps only exact-repeat
  queries.
- **Proposed fix:** give queries priority: a separate query encoder instance
  (FastEmbed sessions are not that heavy to duplicate, ~2.1 GB model files but
  shared on-disk; RAM cost is the ONNX session) or a priority-aware lock
  (embedder checks a "query waiting" flag between batches and yields). At
  minimum, chunk passage encoding so the lock is released between smaller
  sub-batches during catch-up. Consider capping embedder ONNX threads to
  leave a core for query inference on the 8-core host (embedder currently
  created with `threads=8`, `server_log_dense.py:80-84`).
- **Expected impact:** removes unbounded `query_lock_wait` spikes during
  embedding catch-up; typical saving is one batch-encode duration per affected
  query (tens to hundreds of ms at batch 32).
- **Risk:** low–medium. A second encoder instance doubles ONNX session RAM;
  measure on the target host. Thread reallocation may slow bulk embedding
  slightly while speeding queries — the right trade-off for an interactive bot.
- **Pending patches:** Patch R makes `_query_vector` thread-safe with a
  dedicated `query_cache_lock`, but the actual `encode` still goes through the
  shared `self.lock`, and the snapshot shares the same embedder — so the
  contention itself is **not** fixed, only the cache bookkeeping.

### F9 — Speaker resolution does per-alias and per-boundary SQL

- **Where:** `server_log_context.py:34-67` (`resolve_people`),
  `server_log.py:530-541` (`_author_ids`), `543-574` (`_speaker_query`).
- **What's slow and why:** `resolve_people` runs on every pre-retrieval:
  it selects **all** aliases for the visible channels with
  `instr(?, normalized) > 0` — a full scan of the channel-scoped alias rows
  that cannot use the `author_alias_lookup` index (the predicate is on the
  query string containing the alias, not an equality) — then for each matching
  alias runs a regex search in Python **and** a separate
  `SELECT 1 FROM messages …` existence query. `_speaker_query` (run inside
  every `store.search` with author inference) tries every whitespace boundary
  of the query string as a potential speaker name — for a long query that is
  O(words) boundaries × up to 2 `_author_ids` SQL queries each (plain and
  honorific-stripped). Both are bounded by small constants in the common case
  (short queries, few aliases) but degrade linearly-to-quadratically with
  query length and alias count, all inside the 850 ms budget and all through
  F6's per-call connections.
- **Proposed fix:** cache the channel-set alias table in memory (it changes
  only when a new author/name appears — invalidate on the same dense/generation
  watermark or a short TTL), and resolve speakers against the in-memory table
  with a single existence check per *resolved* author, not per alias. In
  `_speaker_query`, cap boundary attempts (e.g. first 4 tokens / 32 chars —
  real speaker prefixes are short) — the current patterns already only succeed
  for genuine prefixes.
- **Expected impact:** turns a per-turn O(aliases + words) SQL fan-out into
  O(1) queries; estimate 5–50 ms saved per person-question, more on cold
  caches / large alias sets.
- **Risk:** low. Alias cache staleness only affects brand-new speakers' first
  minutes; a TTL bounds it.
- **Pending patches:** neither addresses this.

### F10 — `collect()` overview: heavy SQL + O(n²) Python sampling

- **Where:** `server_log.py:656-760` (`collect`), used by the "overview" stage
  at `server_log_context.py:70-84`.
- **What's slow and why:** for a person overview, `collect` runs (on the same
  scope) a count/min/max query, a per-channel group-by, **and** a window-function
  query (`ntile(8)` + `row_number()` over *all* matching rows — for a prolific
  author that can be tens of thousands of rows sorted twice). It then selects
  the diverse sample in Python with a `while rows: row = max(rows, key=…);
  rows.remove(row)` loop (`server_log.py:710-715`) — O(n²) with an expensive
  key function (set lookups + `len(text)`) recomputed for every remaining row
  on every pick, for up to `offset + limit` ≤ 1,060 picks. This runs inside
  pre-retrieval's deadline on the shared `_SQL_POOL`.
- **Proposed fix:** push the sampling into SQL (the `ranked` CTE already
  computes `bucket_rank`; select `bucket_rank = 1` and then do the
  channel/time diversity pass over at most 8 × #channels pre-filtered rows —
  which the query already materialises — instead of over all matches), or cap
  the Python loop's input by ordering/limiting in SQL first. The current SQL
  already filters to `bucket_rank = 1` before returning rows, so the O(n²)
  loop's input is (#channels × 8) rows — the real cost driver is the window
  functions over the full match set; add a covering index or pre-limit by
  taking the longest-per-bucket via a cheaper plan, and cache the overview per
  (author, scope, generation) for a few minutes — overviews change slowly.
- **Expected impact:** overview stage from potentially deadline-eating
  (prolific authors) to a bounded ~10–30 ms; estimate, needs a profile on the
  real archive.
- **Risk:** low. Sampling semantics ("longest per channel/time octile") must
  be preserved exactly; it is explicitly a sample, not an exhaustive profile.
- **Pending patches:** neither addresses this.

### F11 — Pre-retrieval does up to three full searches per turn

- **Where:** `server_log_context.py:134-196` (`search` helper, `dense_search`,
  lexical fallback task).
- **What's slow and why:** per turn, `pre_retrieve` launches a lexical-only
  search (limit 8) as a fallback **and** a dense search (limit 3 or 8) **and**,
  for a single resolved person, an author-filtered dense search (limit 2).
  Each `store.search` call re-opens a connection (F6), re-derives scope SQL,
  re-bigrammes the query, runs the FTS BM25 query, and (for the dense ones)
  enters the F1/F2 machinery. If dense succeeds, the lexical result is thrown
  away (`available.pop("lexical")`) — it is insurance for the deadline, paid
  in CPU on every turn, on a host where the embedder may already be saturating
  cores. The two dense calls also serialise on `cache_lock` (F1), so the
  second one pays lock-wait for the first one's sync.
- **Proposed fix:** after F1/F2 land, re-evaluate whether the parallel lexical
  insurance is still needed — a snapshot-backed dense search with a fast
  lexical *merge* (the partial-fusion path already interleaves lexical hits)
  may make the separate fallback redundant except when dense is unavailable.
  Share one connection and one scope computation across the stages of a turn.
  Run the author-filtered search only if the general dense search returned
  nothing for that author.
- **Expected impact:** up to ~⅓–½ of pre-retrieval CPU per turn; reduces pool
  contention that feeds F1's lock waits.
- **Risk:** medium. The lexical fallback exists because dense used to miss the
  deadline entirely; removing it before F1/F2 are proven live would regress
  answer quality on deadline hits. Sequence this **after** Patch R.
- **Pending patches:** Patch R makes the dense calls cheaper (snapshot) but
  keeps the fan-out. Patch W eliminates pre-retrieval entirely for explicit
  web-lookup turns (also skipping member-wiki context and recent-messages —
  a real latency saving on those turns, in addition to its privacy purpose).

### F12 — Dense scan: several O(N) passes over a multi-GB matrix per query

- **Where:** `server_log_dense.py:358-407` (search body), `332-356` (coverage).
- **What's slow and why:** per query the code builds `permitted` via
  `np.isin` over the full channel array (and again over authors when filtered),
  copies/masks it in `_coverage`, computes `matrix[:size] @ vector` with BLAS
  limited to **1 thread** (deliberate, to avoid oversubscription — benchmark
  doc), negates the permission mask (`scores[~permitted] = -inf`, another
  full-size temporary), `argpartition`s, then sorts the ≥200-element fetch
  window in Python with a tuple key. At ~640 k × 1,024 dims the matmul alone is
  ~0.65 GFLOP and the auxiliary passes touch ~5 full-size arrays (≈13 MB of
  bool/float temporaries at 640 k, plus the 2.6 GB matrix read). The benchmark
  doc's 550 k synthetic warm p50 of ~151 ms is the floor this produces; it
  grows linearly with corpus size and is paid on every unique query.
- **Proposed fix:** this is inherent to exact brute-force search; options are
  (a) accept it (it is not the current deadline bottleneck — F1/F2 are),
  (b) precompute a combined permission bitmask per (channel-set) and cache it
  keyed by snapshot sequence + scope, so `isin`/mask passes happen once per
  scope per snapshot, (c) longer term, evaluate an ANN index (e.g. IVF) — a
  ranking-change project, not an optimisation, and out of scope for a drop-in
  fix. Also consider float16 matmul or splitting the scan by channel shard so
  out-of-scope shards are never touched (channels are already the primary
  partition: `_sync` loads ordered by channel, so a channel-range index would
  let the scan skip most of the matrix for narrow scopes).
- **Expected impact:** (b) saves the non-matmul O(N) passes (estimate 10–30 %
  of dense time); channel-sharding could save much more for single-channel
  scopes, which are the common case in `pre_retrieve` (audience channels).
- **Risk:** (b) low; (c) high (recall changes — would need the full benchmark
  harness re-run).
- **Pending patches:** neither addresses this; Patch R's snapshot search
  inherits the identical scan.

---

## 3. Server Wiki — `server_wiki.py`

### F8 — Lookup validates batches it will never use, at full hashing cost

- **Where:** `server_wiki.py:437-470` (`lookup`, up to 200 candidates),
  `471-503` (`_pages`).
- **What's slow and why:** in the base code, `_pages` processes candidates in
  order and, for each candidate whose batch hasn't been seen, **first** loads
  the batch's messages from the archive (`_ROW_SQL` over the batch's rowid
  range — up to 400 messages), converts and sorts them (`_messages`), and
  computes `_digest(messages)` — a SHA-256 over the JSON serialisation of the
  *full text of every message in the batch* — and only **after** that checks
  whether the entry is a duplicate, whether the page cap (5 pages) is reached,
  or whether the page already has 8 entries/aliases. A person with entries in
  many batches therefore triggers dozens of full-batch loads + hashes per
  lookup, most of them discarded. This lookup runs inside pre-retrieval's
  "server_wiki" stage (850 ms budget) and in the `/wiki` command and
  `read_server_wiki` tool. There is also no caching: every turn re-opens both
  DBs read-only and re-validates from scratch.
- **Proposed fix:** exactly Patch R's reordering (dedup/cap checks before any
  batch load) plus: cache validated batch digests keyed by
  (channel, start, stop, digest) across lookups in-process — a batch that
  validated once stays valid until its digest in `server_wiki_batch` changes; longer
  term, store a cheap validation token (e.g. max rowid + count + a hash of
  ids/timestamps only) at build time so validation doesn't need full text.
- **Expected impact:** Patch R's own test asserts ≤ 9 validations for a lookup
  that previously could validate up to 200 batches' worth of entries; expect
  the server_wiki stage to drop from a potential deadline-buster to tens of ms in
  the common case.
- **Risk:** low for the reordering (behaviour-identical outputs, verified by
  Patch R's new test which also checks edits/deletions still invalidate).
  Patch R also adds `NOT INDEXED` to the batch query and a rowid subquery for
  profile samples — query-plan changes that should be `EXPLAIN QUERY PLAN`
  checked on a large archive, since `NOT INDEXED` forces the rowid-range scan
  even if the planner would prefer the channel index.
- **Pending patches:** **Patch R addresses the main cost** (early dedup/cap,
  indexed validation). It does **not** add cross-lookup caching, and batches
  that *are* used still pay full JSON+SHA-256 validation per lookup.

### Server Wiki build path (secondary observations)

- **Full-corpus digest pass per build:** `build()` (`server_wiki.py:311-435`)
  streams the entire archive via `batches()` (500-row `fetchall` windows,
  `248-272`) and computes `_digest(messages)` (JSON + SHA-256 over full texts)
  for **every** batch just to discover it is unchanged, plus one
  `SELECT … FROM server_wiki_batch` per batch to check the stored digest
  (`392-396`). A no-op rebuild of a ~1.2 M-message archive is therefore an
  O(corpus) serialisation+hashing pass before any model call. Fix: keep a
  row-level change journal like `dense_changes` (or compare
  count/max-rowid/max-edited_at per batch range first and only hash ranges
  whose cheap fingerprint moved). Risk: medium (resume correctness); impact:
  rebuild/resume checks go from O(corpus) hashing to O(changed).
- **`_PROFILE_ROWS`** (`server_wiki.py:56-77`) runs two window functions over the
  whole `messages` table filtered by `guild_id`, for which there is **no
  leading index** (existing indexes are `(channel_id, created_at, id)` and
  `(author_id, channel_id, created_at, id)`); it also evaluates
  `NOT EXISTS (deleted_messages …)` per row twice. It runs once per build, so
  impact is bounded, but on 1.2 M rows it is a multi-second full scan + sort.
  An index on `(guild_id, author_id, channel_id)` would also serve
  `batches()`' guild filter. Patch R does not touch this.
- **`_compact_text`** (`server_wiki.py:180-191`) runs two compiled regex subs and
  a Python `any(char.isalnum() …)` scan per message per build — acceptable,
  and the regexes here *are* module-level compiled (unlike F18's hot spots).
- **`_first_object`** (`server_wiki.py:89-100`) retries `JSONDecoder.raw_decode`
  at every `{` in model output until one parses — worst case O(n²) on
  adversarial/broken output, but outputs are capped small; negligible.

---

## 4. General — orchestration, Codex, bot, and hot spots

### F4 — Codex provider: global serialisation + per-turn process/setup costs

- **Where:** `codex.py:586-621` (`create` under `self.lock`), `507-525`
  (`_verify_version`), `539-563` (`server` context manager), `565-584`
  (`direct_catalog`), `322-334` (`initialize`), `483-498` (`prepare_runtime`).
- **What's slow and why:**
  - `CodexClient.create` holds a single `asyncio.Lock` for the **whole turn**
    (up to the 600 s timeout). The bot allows 4 concurrent AI slots
    (`bot.py`, `self.slots = asyncio.Semaphore(4)`), but with the Codex
    provider every turn serialises behind the previous one — a slow turn
    (tool loops, sandbox commands) blocks all other channels' replies. This
    is a deliberate-looking safety choice (auth-refresh races), but it caps
    throughput at 1 concurrent conversation.
  - Per turn, `create` additionally: spawns the Codex binary just to check
    `--version` (`_verify_version` — a subprocess exec + wait per turn);
    re-reads and `json.loads`-parses the model catalog file **twice-worth**
    (once inside `direct_catalog` even when the catalog already exists:
    `catalog.read_text()` + parse at `581`, after the exists-check at `567`);
    creates a fresh temporary workspace directory; and spawns a **fresh
    app-server subprocess** (`server()`), then does `initialize` +
    `account/read` + `thread/start` + `turn/start` round-trips before the
    model sees the prompt. Process spawn + handshake is plausibly 100–500 ms
    per turn (estimate; no measurement in repo) — paid on *every* message,
    including trivial ones.
- **Proposed fix:** cache the version check per client (binary path + mtime);
  cache the parsed catalog in memory keyed by file mtime instead of re-parsing
  per turn; investigate a pooled/persistent app-server with per-thread
  isolation (the protocol is thread-based already) or at least a warm spare
  process; if the global lock is only about auth refresh, narrow it to the
  refresh critical section (as `warm_server_wiki_auth` already does separately) and
  allow N concurrent turns with per-turn runtimes.
- **Expected impact:** per-turn fixed overhead down by the spawn+handshake
  cost; concurrency from 1 → up to the bot's 4 slots. For a busy server this
  is the difference between replies queueing for minutes and replying in
  parallel.
- **Risk:** **high — security-sensitive.** The per-turn fresh runtime,
  isolated `CODEX_HOME`, and strict config are deliberate isolation controls
  (HANDOFF §Codex). Any pooling must preserve: no user config/MCP/plugins,
  `auth.json`-only copying, approval rejection, and tool registration scoping.
  Do not trade these away for speed without owner review. The version/catalog
  caching sub-fixes are low-risk by comparison.
- **Pending patches:** neither touches `codex.py` (Patch W only changes the
  privacy-guard call path in `assistant.py`).

### F15 — Bot event handling (`bot.py`, `server_log_discord.py`)

- **Where:** `bot.py:165-172` (`archive`), `174-199` (delete/edit handlers),
  `199-236` (`on_message`), `server_log_discord.py:20-46`
  (`allowed_channel_ids`), `49-110` (`audience_channel_ids`).
- **What's slow and why:**
  - `on_message` **awaits** `archive(message)` (a threaded single-message
    upsert paying F5+F6 costs) before the mention/reply check and before the
    assistant is invoked — archival latency is on the response critical path
    for messages that will be answered. (Archiving *all* messages is by
    design; awaiting it inline is the issue.)
  - `on_raw_bulk_message_delete` loops `mark_deleted` once per id, each with
    its own connection + transaction (F6) — a 100-message purge is 100
    connections. Add a `mark_deleted_many` with one transaction.
  - `on_raw_message_edit` does a `fetch_message` API call per edit event
    (necessary to get new content, but uncached/undeduped — edit storms hit
    the API per event).
  - Per answered message, `allowed_channel_ids` iterates **every** guild
    channel/thread and calls discord.py `permissions_for` on each (overwrite
    resolution in Python), with a potential `fetch_member` API call per
    private thread; `audience_channel_ids` then repeats permission resolution
    against the default role for the same channel set. No caching across
    messages, although permissions change rarely. In a large guild this is
    O(channels) Python work (plus possible API calls) per turn, before any
    retrieval starts.
- **Proposed fix:** fire-and-forget archival through a bounded queue with a
  single writer task (which also enables micro-batching live upserts — see
  F5/F7); batch bulk deletes; cache permission-derived channel lists per
  (member, guild) with invalidation on channel/role/member update events and
  a short TTL backstop.
- **Expected impact:** removes upsert latency (ms-scale, but under WAL
  contention much more) from every reply; bulk deletes go from O(n)
  transactions to O(1); permission computation from per-turn O(channels) to
  amortised ~0.
- **Risk:** medium. A queued archiver must preserve ordering guarantees that
  `recent` / pre-retrieval implicitly rely on (a message should ideally be
  archived before retrieval for the *next* message runs — a short flush
  interval or flush-on-reply preserves this). Permission caching must fail
  closed on invalidation bugs (privacy boundary, #25 in HANDOFF).
- **Pending patches:** neither addresses this.

### F16 — Assistant orchestration serialises independent reads

- **Where:** `assistant.py:108-340` (`reply`), esp. `164-198`.
- **What's slow and why:** pre-retrieval is correctly started as a task early
  (`164`), but the member-wiki context (`177-186`, threaded DB call) and the
  recent-messages fetch (`188-197`, threaded DB call) are then awaited
  **sequentially**, each paying F6 connect costs, before the retrieval task is
  awaited. They are independent of each other; serialising them adds their
  latencies instead of overlapping them (they already overlap with
  pre-retrieval, which is why this is only low–medium impact). The tool loop
  itself dispatches each tool call via `asyncio.to_thread` on the shared
  default executor — the same executor used by archival upserts and backfill
  flushes — so heavy tool use and heavy ingest contend for the same threads.
- **Proposed fix:** `asyncio.gather` the wiki-context and recent fetches (or
  start them as tasks alongside `retrieval_task`); consider a dedicated
  executor for DB reads so ingest writes can't starve tool reads.
- **Expected impact:** saves roughly one DB round-trip latency (wiki context
  *or* recent, whichever is smaller) per turn — ms to tens of ms.
- **Risk:** low.
- **Pending patches:** Patch W removes both calls entirely on explicit
  web-lookup turns; the general path is unchanged.

### F17 — Web tools rebuild their HTTP stack per call

- **Where:** `web_tools.py:151-170` (`fetch_text`), `374-400` (`call`).
- **What's slow and why:** `WebTools.call` runs `asyncio.run(...)` per tool
  call (new event loop), and `fetch_text` builds a new `TCPConnector` +
  `ClientSession` per request with `use_dns_cache=False` and `force_close=True`
  — so every search/read pays DNS resolution + TCP + TLS handshake with no
  connection reuse, even for consecutive Wikipedia API calls in one turn.
- **Proposed fix:** keep the DNS behaviour (the uncached, all-IPs-vetted
  resolution is an anti-rebinding security control — do **not** "optimise" it
  into a cached resolver), but reuse a session/connector *within* a single
  turn's batch of web calls, or accept the cost: it is bounded by the 10 s
  fetch timeout and web turns are not the common case.
- **Expected impact:** saves one TLS handshake (~50–150 ms, estimate) per
  additional web call in a turn. Low priority.
- **Risk:** low if scoped to a turn; **do not** weaken the resolver/IP vetting.
- **Pending patches:** neither addresses this (Patch W only reduces how often
  web calls are *blocked*, not their cost).

### F18 — Micro-hotspots (individually small, worth a cleanup pass)

- `bigrams()` (`server_log.py:28-38`) calls `re.fullmatch` **per token** to
  test whether a token is a Japanese run — the findall pattern already
  guarantees which alternative matched; branch on the first character's range
  instead. `bigrams` runs per ingested message (FTS) and per query, and
  `_fusion` (`server_log.py:154-175`) re-bigrammes both the query and the full
  text of the top lexical hit on every search to compute coverage/anchors.
  Module-level compiled patterns would also help (`re`'s internal cache makes
  this minor, but the per-token `fullmatch` is algorithmic, not cache, waste).
- `messages.py:203-224` (`split_message`) encodes **every character** to
  UTF-16-LE to count Discord code units — fine at ≤ 8,000-char answers, but
  `split_reply` + `source_lines` repeat similar per-character/per-source work;
  a `len(text.encode('utf-16-le')) // 2` fast path with a per-character
  fallback only near the limit would do. Low impact.
- `sandbox.py` / `selfmod.py`: `run_command`'s Docker-per-command model is a
  security design, not a bug; note only that `workspace_user()` shells out to
  `docker info` per selfmod command — selfmod is rare, ignore.
- `policy.py:validate_workspace` walks up to 10,000 files per selfmod
  validation (twice per proposal) — one-off cost, ignore.
- Regex compilation overall is **not** a systemic problem in this codebase:
  the genuinely hot modules mostly use module-level compiled patterns
  (`server_wiki.py:53-55`, `messages.py:38-62`, `server_log_context.py:28-31`,
  `member_wiki.py:19`). The exceptions are the string-pattern calls inside
  `bigrams` / `_speaker_query` noted above and per-call `re.fullmatch` in
  argument validation (`web_tools.py:392`, `member_wiki.py:209-213`) — all
  bounded by small inputs.

---

## 5. What the pending patches already cover — consolidated

| Patch | Covers | Does not cover / watch items |
|-------|--------|-------------------------------|
| **W** `web-skip-preretrieval` (`f2c98d5`) | HANDOFF known issue #1 (web reads blocked after private context) by skipping pre-retrieval, member-wiki context, and recent-messages on explicit web-intent turns, and by not tainting `private_read` for empty retrieval. Side benefit: those turns also skip F11/F16 costs entirely. | Nothing else in this audit. The regex intent detector is a new per-turn cost (negligible, single compiled pattern — in the patch it is compiled at module level). History taint correctly remains. |
| **R** `retrieval-snapshots` (`26bb38c`) | F2 (background snapshot refresh instead of per-query `_sync`, for the pre-retrieval path), F8 (Server Wiki validation reordering + indexed batch reads), part of F1 (snapshot searches don't wait for refresh). | F1's lock scope within a search; tool-path (`search_server_log`) sync; cold-start warmup at bot startup; F3 embedder/query lock sharing; F5/F6/F7/F14 (all DB write/connect costs); F4/F15/F16/F17. **Risks to review:** shallow-copy snapshot sharing matrix/embedder/locks; append-only `snapshot_mode` invariant in `_load`; per-candidate SHA-256 of vector blobs during snapshot validation; `coverage=None → partial` semantics feeding `_fusion`; `NOT INDEXED` plan change in Server Wiki; full test suite not run (HANDOFF). |

## 6. Suggested sequencing

1. **Land Patch R after review + full tests** (F2, F8, part of F1) — it is
   already written and targets the measured deadline killers. Add bot-startup
   snapshot warmup and extend `snapshot_only` to the tool path, or explicitly
   accept the tool path's different freshness/latency trade-off.
2. **DB rebuild prep: F7 + F14 + F5's cheap parts** (cursor-in-transaction,
   larger flushes, channel-parallel backfill, `synchronous=NORMAL` decision,
   single-hash in `_upsert`) — these decide whether the planned rebuild takes
   hours or days, and they are independent of Patch R.
3. **F6 connect-overhead caching** (with owner sign-off on the security
   trade-off) and **F3 query-embedding priority** — steady-state latency.
4. **F4 Codex** version/catalog caching (low-risk parts only); treat app-server
   pooling / lock narrowing as a separate security-reviewed project.
5. **F9/F10/F11/F15/F16** cleanups, then re-run `server_log_context diagnose`
   (HANDOFF) and the benchmark harness to verify the deadline is met with
   headroom before considering any F12 ANN work.

*End of audit. No source files were modified.*
