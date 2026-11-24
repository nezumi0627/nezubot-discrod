# Optimization Worklog — RAG review & structural cleanup

- **Date:** 2026-11-24 (JST). Operator: optimization subagent, supervised by the
  main agent. Production processes (bot / Server Wiki / embed worker / relay) were
  never restarted; `deploy.env` untouched; all production-DB access was
  read-only (`mode=ro` connections, snapshot search only).
- **Baseline:** pytest 1054 passed / 13 skipped, `ruff check` clean,
  line length 100. HEAD at start: `d5000b0`.
- **Measurement harness:** `~/workspace/measure_rag.py` (outside the repo)
  drives the real `pre_retrieve` against the production DB with the
  production 850 ms budget, all-channel scope (906 channels), caller
  testuser0627, warm dense snapshot (689,746 vectors, 100% teacher).
  Numbers below are from that harness unless noted. VM is a shared
  1.5-core box also running the bot + Server Wiki, so treat timings as
  comparative (before/after on the same harness), not absolute.

## Baseline measurements (before any change)

| Query | elapsed (2 reps) | status | context chars | chunks | sources |
|---|---|---|---|---|---|
| 俺について簡単にまとめて | 852 / 785 ms | complete | 5,893 | dense 1, overview 1, server_wiki 4, author 1 | 17 |
| vexってどう思う？ | 852 / 818 ms | complete | 692 | dense 1, overview 1 | 3 |
| サーバーオーナーおしえて | 757 / 372 ms | complete | 1,618 | dense 1 | 8 |

Stage detail (self query): `dense_search` 782 ms, `lexical` (FTS, summed
over all `store.search` calls) 474 ms, `lexical_fallback` 213 ms,
`overview` 236 ms, `people` 1.6 ms. Vex query: `dense_search` 715 ms,
`lexical` 933 ms, `people` 101 ms, `skipped=[dense_snapshot]` (the
author-scoped snapshot search found no candidates — see A4).

Fixed per-turn context (same harness): system prompt SYSTEM+LOG+WIKI
guidance 7,756 chars; LOG+WIKI+WEB tool schemas 7,203 chars;
`recent_messages` (15 hits, quiet channel) 2,709 chars. The retrieved
context budget is 6,000 chars and the self query nearly saturates it
(5,893). A full turn's 29k–44k input tokens are this fixed+retrieved
context multiplied by the number of model rounds (usage is summed over
rounds in `Assistant.reply`), so retrieved-context savings multiply.

## RAG audit findings

- **A1 — The same FTS query runs up to 3× per turn.** `pre_retrieve`
  starts a lexical-fallback search in parallel, and each of the two
  dense-path `store.search` calls (unscoped + author-scoped) runs the
  identical standalone-bm25 FTS query internally before fusing. The
  fallback's lexical result and the unscoped search's lexical result are
  the *same rows* (same query, scope, pool=50); when dense succeeds the
  fallback copy is discarded. On a 1.5-core VM the three FTS scans over
  common-bigram posting lists (213–933 ms summed per turn) also contend
  with each other and with the dense stage, which is the critical path
  (715–782 ms of the 850 ms budget).
- **A2 — No cross-chunk deduplication in `pre_retrieve`.** Chunks are
  selected in kind order until the 6,000-char budget fills, but the same
  message can appear in several chunks. Measured on the self query:
  author search hit `123456789012345678` is also the dense chunk's #1
  hit, so its ~350-char formatted line is delivered twice (and consumes
  two of the ≤17 sources). Overview samples did not overlap in the
  measured queries (collect's channel/time diversity works), but nothing
  prevents it.
- **A3 — Snapshot-missing path burns FTS before failing.** In
  `ServerLog.search`, the lexical query runs *before* the dense attempt.
  With `snapshot_only + require_dense` (the pre-retrieval dense job), a
  missing/stale snapshot raises `DenseSnapshotUnavailable` only after a
  full FTS scan — twice per turn (unscoped + author attempts), exactly
  in the post-restart window where the lexical fallback is the only
  evidence and CPU contention is worst.
- **A4 — Bot-person author search is guaranteed empty.** The dense
  index never loads bot messages (`is_bot` rows are skipped in
  `DenseIndex._load`, and `embed_pending` never embeds them), and
  `collect()` filters `is_bot = 0`. For a bot person (e.g. vex), the
  author-scoped dense search therefore always ends in
  `DenseSnapshotUnavailable` (observed: `skipped=[dense_snapshot]`) and
  the overview sample is always empty; the unscoped dense chunk carries
  the evidence (other people's messages about the bot). This is
  functionally acceptable — answers about vex measured fine — but the
  author attempt still pays a full FTS + mask pass before failing.
  A3's fix removes the FTS part of that waste.
- **A5 — `resolve_people` cost is spiky.** 1.6 ms (self) vs 101 ms
  (vex): the alias `instr()` scan plus one existence query per matching
  alias, and the token fallback loads *all* alias rows for the channel
  set into Python. It sits serially before the dense job. Tolerable at
  current scale; noted, not changed (prior audit F9).
- **A6 — Duplicate guidance text.** Citation/source-footer rules are
  stated in SYSTEM, LOG_GUIDANCE and again in the LOG_TOOLS
  descriptions (~7.2k chars of schemas resent every round). Trimming is
  possible but behavior-sensitive and hard to verify offline; left as a
  future candidate, deliberately not touched here.
- **A7 — Failure modes inside the deadline are graceful.** Deadline
  expiry keeps partial chunks (`status=deadline`), a stuck dense stage
  does not block later turns (slot semaphores + snapshot reads), and a
  failed snapshot refresh never discards the published snapshot. The
  changes below preserve all three properties (the timeout/partial
  results test is the guard).

## Structural audit findings

- **S1 — `Assistant.reply` re-sends the full context every tool round**
  (inherent to the chat-completions shape; the Codex path likewise).
  Mitigation is context size (A1/A2), not structure.
- **S2 — Dead-ish state:** `log_state.generation` is still incremented
  by writers although the F5 comment in `_upsert` records that nothing
  in production reads it. Removing the writes touches the ingest hot
  path and the vector-import tooling for zero user-visible gain; left
  alone deliberately.
- **S3 — Config sprawl is contained:** `SERVER_LOG_*` env vars are read
  in `ServerLog.__init__` only; model selection (fastembed vs local
  ONNX) is decided in `DenseIndex.__init__` from `store` fields. No
  change needed.
- **S4 — Test coverage of the retrieval seam is good** (context,
  snapshots, integration suites pin the deadline/partial behavior, the
  fusion gate, and snapshot revalidation). The thin spot was the
  lexical-recomputation seam (A1) — it had no test because the lexical
  query was inlined in `search()`; Change 1 factors it out and tests
  the sharing directly.

## Improvements

### Change 1 — Share one FTS computation per pre-retrieval turn (A1)

- **Aim:** run the identical standalone-bm25 FTS query once per turn
  instead of up to three times; cut SQL-pool contention on the dense
  critical path without changing any delivered result.
- **Change:** factor the lexical query in `ServerLog.search` into
  `_lexical_query()`; add a `LEXICAL_MEMO` contextvar. When a memo is
  active, searches with the same (expression, scope, pool) reuse the
  first computation (concurrent callers wait on an event for the owner
  thread's rows; if the owner fails, a waiter computes its own).
  `pre_retrieve` installs a fresh memo per call (same mechanism as the
  existing `Timings` contextvar, propagated to executor threads via
  `copy_context`). No memo is active outside `pre_retrieve`, so tool
  calls and CLI searches are unchanged.
- **Tests:** new `test_lexical_query_is_shared_within_a_memo_context`
  (two identical searches → one `_lexical_query` call with a memo, two
  without). Full suite + ruff after the change.
- **Measurement:** see the after table below.

### Change 2 — Deduplicate hits across pre-retrieval chunks (A2)

- **Aim:** never deliver the same message twice in one turn's context;
  spend the 6,000-char budget and the source slots on unique evidence.
- **Change:** in `pre_retrieve`'s selection loop, track delivered
  message ids. Search chunks (which carry `.hits`) are rebuilt from
  only their not-yet-delivered hits (the single-person lexical trim to
  3 applies to the *fresh* hits). An overview chunk whose every source
  message was already delivered is skipped (its excerpts are the same
  raw messages; its stats header is the only loss). Server Wiki and
  member-wiki chunks are intentionally exempt: their text is derived
  summaries/excerpts, not the source message itself.
- **Tests:** new `test_pre_retrieval_deduplicates_hits_across_chunks`.
- **Measurement:** see the after table below.

### Change 3 — Probe the dense snapshot before FTS when the caller requires dense (A3)

- **Aim:** in the snapshot-missing window, fail the dense attempts
  immediately instead of after a full FTS scan each.
- **Change:** in `ServerLog.search`, when `snapshot_only and
  require_dense` and dense is enabled, attempt `search_snapshot`
  first; `None` raises `DenseSnapshotUnavailable` without running FTS,
  and a generic dense exception falls through to the existing
  lexical-fallback flow unchanged. When the probe succeeds, FTS runs
  afterwards for fusion exactly as before, so fused results are
  identical. All other call shapes keep the original order.
- **Tests:** existing snapshot/integration suites pin this seam; new
  `test_require_dense_snapshot_probe_skips_fts_without_snapshot`.
- **Measurement:** failure-path timing is hard to reproduce against
  the live DB (snapshot is warm); verified by unit test + code
  inspection. Normal-path numbers in the after table.

## After measurements

Same harness, warm reps (last of 3; the box's load varies with Server Wiki
phases, so first reps still touch the deadline under load — the
steady-state reps are the comparison):

| Query | elapsed | dense_search stage | lexical total | chars | sources |
|---|---|---|---|---|---|
| 俺について簡単にまとめて | 785 → **361 ms** | 782 → 356 ms | 474 → 242 ms | 5,893 → 5,696 | 17 → 16 |
| vexってどう思う？ | 818 → **219 ms** | 715 → 126 ms | 933 → 224 ms | 692 → 509 | 3 → 3 |
| サーバーオーナーおしえて | 372 → **170 ms** | 348 → 151 ms | 190 → 124 ms | 1,618 → 1,618 | 8 → 8 |

Reading of the deltas:

- The duplicated message (`123456789012345678`, dense #1 and author
  #1 on the self query) is delivered once: −1 source, −197 chars, and
  the freed budget stays available for unique evidence.
- Vex chars drop by 183 because the overview chunk is now dropped when
  its sample is empty: for a bot person `collect()` always returns
  zero hits (A4), so the chunk previously contributed only a
  zero-count stats header and "No messages found." — misleading next
  to vex's 1,130 archived (bot-flagged) messages. Deliberate side
  effect of Change 2, recorded here.
- The vex author attempt now fails in ~0 ms of FTS (Change 3's probe),
  and the unscoped FTS runs once per turn instead of twice (Change 1),
  which is most of the `lexical` and `dense_search` stage reduction;
  the rest is reduced CPU contention knock-on.
- Owner query output is byte-identical (chars and sources unchanged),
  only faster — the expected signature for Changes 1+3.

Commits: `50135f4` (Change 1), `f35fe1b` (Change 2), `10b09c5`
(Change 3). Full suite after all three: **1057 passed / 13 skipped**
(baseline 1054 + 3 new tests), ruff clean.

## Deployment

All three changes are library-level (`server_log.py`,
`server_log_context.py`); no schema, config or `deploy.env` changes.
The running bot loaded the old modules at startup, so **a bot restart
is required to deploy** (relay / Server Wiki / embed worker can keep running;
restarting the embed worker too is optional — it imports
`server_log.py` but none of its changed paths serve embedding).
No production process was restarted during this work.

## Left for later

- A6 (guidance duplicated between system prompt and tool schemas) —
  needs live answer-quality checks to trim safely.
- A5 (`resolve_people` alias-scan spikes, up to ~130 ms for some
  names) — index/prefilter work, prior audit F9.
- The dense stage still pays one FTS per scoped search (fusion needs
  the ranks); a shared *author-scoped* lexical is impossible because
  the scope differs per search. The remaining big lever on input
  tokens is multi-round re-sends (S1), i.e. fewer tool rounds via
  better first-pass evidence, not context plumbing.
- Phase B (article-level embeddings / context-driven recall) remains
  design-only, out of scope here as instructed.

---

# Round 2 — 2026-11-24 (A5 / A6 / S1 + cold-fill follow-up)

Baseline at start: 1057 passed / 13 skipped, ruff clean. All changes
are local commits; no production process was restarted, `deploy.env`
untouched, no schema changes.

## A5 — resolve_people alias scans (commit fd6f7c2, follow-up 33ba416)

Problem, measured on the production DB (harness label `before2`):
every person resolution ran two full alias scans (an `instr()`
substring scan and a full scoped table load in the token fallback)
plus one existence query per matching alias, serially before the
dense job. People stage: **795–894 ms for "vexってどう思う？"** (the
whole 850 ms budget; reps delivered 0 chunks), 66–166 ms for the
owner query.

Fix (`server_log_context.py`): aliases are only ever INSERTed
(verified: no UPDATE/DELETE on `author_aliases` anywhere), so
`MAX(rowid)` is a sound watermark. The scoped DISTINCT
(normalized, author_id) pairs (8,889 pairs over 26,761 rows) are
cached on the store per channel-set and invalidated by the watermark;
matching (substring + ASCII word-boundary regex, then the token-key
fallback) runs in memory over the snapshot, and existence checks are
one batched IN query per path (500-id chunks; primary keeps
is_bot = 0, fallback keeps bots allowed, mentions keep mention
order). Resolved sets are provably identical — the existing
identity/boundary/decorated-bot tests pin the semantics, plus two new
tests (snapshot reuse/watermark invalidation; mention order).

Component timings on the production DB, isolated (unloaded moment):
watermark 0.1 ms, scoped DISTINCT select 24.4 ms, snapshot build
3.1 ms, token keys 16.6 ms, primary scan 1.5 ms, fallback scan
1.0 ms → **cold fill ≈ 45 ms once per channel set per process, warm
resolution ≈ 2.5 ms + one batched query**. Direct probe (same store,
4 reps): vex [cold, then 6.6–11.9 ms warm] → ['333333333333333333'];
self 0.3–0.9 ms; owner 1.8–12 ms → []. Follow-up commit 33ba416
cheapened the cold fill itself: token keys are memoized per distinct
alias string and plain `[a-z0-9_]+` handles take a fast path (their
only possible key is the alias itself).

Fresh-process end-to-end (lexical-only store, one call per query):
vex 171 ms complete, people stage 69.6 ms (the cold call), 509 chars
/ 3 sources; owner 59 ms, people 9.9 ms, 1,444 chars / 8 sources
(byte-identical delivery to before2's best rep); self 286 ms,
people 1.3 ms, 5,270 chars / 15 sources.

## A6 — prompt ↔ tool-schema guidance duplication (commit ace1fda)

Fixed context resent on EVERY tool round was 15,467 chars (full
log+wiki+web prompt 8,264 + all tool schemas 7,203). Rules were
stated two or three times: the -# footer rule in SYSTEM,
LOG_GUIDANCE and WEB_GUIDANCE; citation rules in LOG and WEB
guidance; tool-choice guidance inside the search/collect tool
descriptions as well as the prompt.

Canonical homes now: footer/citation rules in SYSTEM plus one
compact line each in LOG/WEB guidance (the -# line in LOG_GUIDANCE
is pinned by test and kept); tool choice (collect = broad research,
search = specific lookup) stated once in LOG_GUIDANCE; tool
descriptions keep only mechanics (return shape, limits, paging,
author resolution forms). LOG_GUIDANCE also drops its
persona/language restatement of SYSTEM and the duplicated
"respond to the message field".

Result of A6 alone: **15,467 → 14,795 chars (−672, −4.3%) resent
per round** (LOG_TOOLS descriptions −513: search 721→539, collect
753→479, read_server_wiki 124→67; LOG_GUIDANCE 2,423→2,339;
WEB_GUIDANCE 502→427). S1 then deliberately reinvested 149 of
those chars in LOG_GUIDANCE (2,339→2,488; see the guidance bullet
under S1 — the longer sentence exists to stop reflexive tool
rounds, each of which resends the whole ~15k block), so the final
fixed total is **14,944 chars (−523, −3.4% net)**. At the observed
~29–44k input tokens per multi-round turn the net saving is
roughly 130 tokens per round, ~260 per typical 2-round turn, plus
the rounds S1 avoids outright. Retrieval code untouched by A6;
the diagnose queries' delivered context is unchanged (see
fresh-process probe above).

## S1 — fewer tool rounds via better first-pass evidence (commit 3951cbe)

- **Server Wiki event pre-fetch (new `server_wiki_event` stage).** LOG_GUIDANCE
  told the model to try read_server_wiki first for people AND events,
  but pre-retrieval only covered person pages — event questions
  always cost an extra round. When a question resolves to ≠ 1 person
  and names an event (a 「」/『』 quoted phrase, or a distinctive
  Latin-script run: ≥5 chars, or ≥3 with an uppercase/digit — short
  lowercase runs like "vex"/"bot" never qualify), pre_retrieve now
  delivers up to 4 Server Wiki event entries itself, published under the
  existing "server_wiki" chunk kind, on the wiki pool. Event titles are
  descriptive Japanese sentences, so verbatim title matching is
  hopeless; the Server Wiki lookup's title-substring match on the named
  event is what makes this work. Verified on the production DB:
  "Paper Challenge 2って何だっけ？" → 1 server_wiki chunk (「Paper
  Challenge 2の協力依頼」); "「技術者倫理」の議論ってどうなった？" →
  4 server_wiki chunks; "CTFの募集ってどうなった？" → 4 server_wiki chunks. Zero
  cost when no candidate exists (all 3 diagnose queries have none).
- **Author evidence 2 → 5.** The single-person author search in the
  dense stage now takes 5 hits instead of 2 — the same search the
  model otherwise runs as a follow-up search_server_log round;
  delivery is still gated by the 6,000-char selection budget. The
  dense-call pattern pin in
  test_first_person_topic_keeps_other_authors_evidence is updated
  2 → 5; its content assertions are unchanged.
- **Guidance aligned with reality.** LOG_GUIDANCE now says the
  resolved-person context (Server Wiki entries, overview sample, author
  messages) is already pre-fetched, and read_server_wiki is for event
  pages or people the context did not cover — instead of demanding
  the tool first for everything.

Interface note (per instructions): pre_retrieve gains one stage
(`server_wiki_event`, timings-visible in diagnostics) and the author search
limit changes; fusion, schemas and the diagnose surface are
unchanged. Phase B was not touched.

## Harness comparison caveat (after2 / after2b)

The full harness was re-run twice after the changes (labels
`after2`, `after2b`). Both runs are load-dominated and NOT usable as
an end-to-end before/after: the box was saturated by sibling work
(snapshot warmup 434.7 s vs 225.5 s in before2; load average ~1.6 on
1.5 cores; the VM also restarted mid-session). Timed-out stages keep
their threads (and stage slots) running past the 850 ms deadline, so
reps running back-to-back starve and skip — in after2b the vex rep0
people stage shows 856 ms (starved cold fill) and later reps skip
people/dense/lexical entirely. The old code showed the same failure
shape in before2 (vex people stage 795–894 ms on EVERY rep, final
rep 0 chunks). The load-independent measurements above (component
probe, direct resolve_people probe, fresh-process pre_retrieve) are
the basis for the A5/S1 conclusions; in-harness, the self query's
people stage reads 0.7–5.4 ms with the new code. Fixed sizes in the
harness reconcile exactly with the direct measurement: system
(LOG+WIKI) 7,756 → 7,821 (LOG_GUIDANCE net +65 after S1's
reinvestment, above), tool schemas 7,203 → 6,690, full fixed block
15,467 → 14,944.

## Tests

Full suite after all four commits: **1061 passed / 13 skipped**
(baseline 1057 + 4 new: 2 A5 snapshot tests, event-candidate
extraction, Server Wiki event pre-fetch integration), ruff clean. One
timing-sensitive test (test_slow_server_wiki_does_not_block_overview_or_
member_wiki) flaked once under load during development and passes
consistently in isolation and in full-file runs.

## Deployment

All changes are library-level (`server_log_context.py`,
`server_log_tools.py`, `prompts.py`). **A bot restart is required to
deploy**; Server Wiki / embed worker / relay are unaffected and can keep
running. No production process was restarted during this work.

## Left for later

- The residual fixed context (~14.8k chars) is now mostly unique
  mechanics/persona text; further cuts would trade answer quality,
  so A6 is considered closed unless prompts are redesigned.
- Input tokens' remaining driver is the tool rounds themselves plus
  history re-sends; S1 reduces rounds for named-event and
  person-evidence questions, but generic topic questions still rely
  on the model's judgement to stop after round 0.
- Under box saturation the pre-retrieval stage slots stay held by
  timed-out threads and later turns skip stages (observed in the
  harness, pre-existing design). Worth a future look (e.g. slot
  lease expiry), but it is a concurrency-behaviour change and was
  not bundled into this round.

## v1.9.0 — hybrid three-path retrieval (2026-11-24)

Fusion extended from [FTS, teacher dense] to [FTS, teacher dense, ft dense]
with the production rrf() (k=60, equal weights); the _fusion head (lexical
anchor + teacher top-1) is unchanged, as are all pool parameters. The ft
query is encoded by the ft model's own ONNX encoder
(configured via SERVER_LOG_HYBRID_ONNX_DIR; output parity with
the training-side encoder verified at cosine 1.0). Offline gate (600-query eval): hit@10 0.0233→0.0350,
MRR@10 0.0073→0.0093. Aux index mirrors the primary snapshot/scope
semantics; any ft-side failure (missing model, missing tag, encoder
error) falls back to the exact two-path behavior. The embed worker now
also runs ft passes so new messages gain ft vectors. Rollback: remove
SERVER_LOG_HYBRID_MODEL from deploy.env and restart.

Latency (this VM, snapshot-path store.search, limit=10, medians of the
settled reps): query A 129→~190 ms, query B 221→292 ms, query C 116→200 ms
(+65–80 ms per dense search: one ft encode + one matmul). Projected
pre_retrieve totals stay well under the 850 ms diagnose budget.
Snapshot build incl. aux sync measured 675 s in a throttled cell
(background refresh; searches keep using the previous snapshot).
