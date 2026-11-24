import ctypes
import hashlib
import json
import logging
import re
import sqlite3
import unicodedata
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from threading import Lock, Thread
from time import monotonic, sleep

from nezubot_discrod.server_log import _date, embedding_text, text_hash
from nezubot_discrod.server_log_timing import stage

_BLAS_LOCK = Lock()
logger = logging.getLogger(__name__)


class DenseHits(list):
    def __init__(self, values, *, partial, coverage):
        super().__init__(values)
        self.partial = partial
        self.coverage = coverage


_MEANINGFUL_CHARS = re.compile(r"[0-9a-z\u3040-\u30ff\u3400-\u9fff]")


def embeddable_text(text):
    """Whether a message carries semantics worth learning as a dense vector.

    After NFKC normalization, fewer than two meaningful characters
    (alphanumerics, kana or kanji) means emoji-only, symbol-only,
    single-character or blank noise. Such messages stay out of the
    embedding queue and the pending count; the stored log row is untouched.
    """
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return len(_MEANINGFUL_CHARS.findall(normalized)) >= 2


def _blas_limiter(np):
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits
    except ImportError:
        pass
    try:
        library = ctypes.CDLL(np._core._multiarray_umath.__file__)
    except (AttributeError, OSError):
        raise ImportError(
            "Install threadpoolctl to control this NumPy BLAS implementation"
        ) from None
    for prefix in ("scipy_openblas", "openblas"):
        for suffix in ("64_", "_64_", ""):
            get = getattr(library, f"{prefix}_get_num_threads{suffix}", None)
            set_threads = getattr(library, f"{prefix}_set_num_threads{suffix}", None)
            if get is None or set_threads is None:
                continue
            get.argtypes, get.restype = [], ctypes.c_int
            set_threads.argtypes, set_threads.restype = [ctypes.c_int], None

            @contextmanager
            def limit(*, limits, user_api, get=get, set_threads=set_threads):
                if user_api != "blas":
                    raise ValueError("Only BLAS thread limits are supported")
                previous = get()
                set_threads(limits)
                try:
                    yield
                finally:
                    set_threads(previous)

            return limit
    raise ImportError("Install threadpoolctl to control this NumPy BLAS implementation")


class DenseIndex:
    def __init__(self, store, embedder=None):
        import numpy as np

        self.np = np
        self.threadpool_limits = _blas_limiter(np)
        self.store = store
        if embedder is None:
            if getattr(store, "embedding_onnx_dir", None):
                from nezubot_discrod.local_embedder import LocalE5Embedder

                embedder = LocalE5Embedder(
                    store.embedding_onnx_dir, max_length=store.max_length
                )
            else:
                from fastembed import TextEmbedding

                embedder = TextEmbedding(store.model, threads=8, providers=["CPUExecutionProvider"])
                if "e5" in store.model.lower():
                    embedder.model.tokenizer.enable_truncation(max_length=store.max_length)
        self.embedder = embedder
        self.lock = Lock()
        self._query_gate = Lock()
        self._query_waiters = 0
        self.cache_lock = Lock()
        self.query_cache_lock = Lock()
        self.refresh_lock = Lock()
        self.refresh_thread = None
        self.last_refresh = 0
        self.snapshot = None
        self.snapshot_mode = False
        self.sequence = None
        self.model = store.model
        self.static_model = None
        self.rebuilds = 0
        self.ids = {}
        self.matrix = None
        self.size = 0
        self.reserve_hint = 0
        self.channels = np.empty(0, dtype=np.int32)
        self.authors = np.empty(0, dtype=np.int32)
        self.channel_names = np.empty(0, dtype=np.int32)
        self.dates = np.empty(0, dtype="S32")
        self.active = np.empty(0, dtype=bool)
        self.channel_codes = {}
        self.author_codes = {}
        self.channel_name_codes = {}
        self.message_ids = []
        self.sort_keys = []
        self.versions = []
        self.queries = OrderedDict()
        self.scope_coverage = OrderedDict()
        self.query_cache_size = 128
        self.pending = sqlite3.connect(":memory:", check_same_thread=False)
        self.pending.executescript("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY, channel_id TEXT, channel_name TEXT, author_id TEXT,
                author_name TEXT, created_at TEXT, deleted INTEGER DEFAULT 0
            );
            CREATE INDEX pending_channel_date ON messages(channel_id, created_at);
            CREATE INDEX pending_author_id ON messages(channel_id, author_id, created_at);
            CREATE INDEX pending_author_name ON messages(channel_id, author_name, created_at);
            CREATE TABLE deleted_messages (id TEXT PRIMARY KEY);
        """)

    def encode(self, texts, *, query=False):
        prefix = ""
        if "e5" in self.store.model.lower():
            prefix = "query: " if query else "passage: "
        if query:
            # Register as a waiting query so passage (document) encoding
            # yields the embedder between batches instead of re-acquiring
            # the lock ahead of an interactive query (audit F3).
            with self._query_gate:
                self._query_waiters += 1
        elif self._query_waiters:
            while True:
                with self._query_gate:
                    if not self._query_waiters:
                        break
                sleep(0.005)
        try:
            with stage("query_lock_wait" if query else "passage_lock_wait"):
                self.lock.acquire()
        except BaseException:
            if query:
                with self._query_gate:
                    self._query_waiters -= 1
            raise
        try:
            vectors = self.np.asarray(
                list(self.embedder.embed([prefix + text for text in texts])),
                dtype=self.np.float32,
            )
        finally:
            self.lock.release()
            if query:
                with self._query_gate:
                    self._query_waiters -= 1
        if (
            vectors.ndim != 2
            or len(vectors) != len(texts)
            or not vectors.shape[1]
            or not self.np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid embedding output")
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero embedding output")
        return vectors / norms

    def _embedding_order(self, rows):
        texts = [embedding_text(row) for row in rows]
        tokenizer = getattr(getattr(self.embedder, "model", None), "tokenizer", None)
        if tokenizer is None:
            lengths = list(map(len, texts))
        else:
            prefix = "passage: " if "e5" in self.store.model.lower() else ""
            with self.lock:
                lengths = [sum(value.attention_mask) for value in
                           tokenizer.encode_batch([prefix + text for text in texts])]
        return [rows[i] for i in sorted(range(len(rows)), key=lengths.__getitem__)]

    def embed_pending(self, batch_size, progress, stop):
        total = 0
        limit = max(1024, batch_size)
        deferred = True
        while deferred and not stop.is_set():
            deferred = False
            cursor = None
            while not stop.is_set():
                with self.store.connect() as db:
                    query = """
                        SELECT m.* FROM messages m LEFT JOIN message_vectors v
                        ON v.message_id = m.id AND v.model = ?
                        WHERE m.deleted = 0
                        AND NOT EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id)
                        AND m.is_bot = 0 AND length(trim(m.text)) >= 2
                        AND v.message_id IS NULL
                    """
                    params = [self.store.model]
                    if cursor is not None:
                        # Noise rows are never embedded, so page past every
                        # scanned row instead of refetching the same head.
                        # Pending rows are embedded newest first so recent
                        # conversations become searchable earliest.
                        query += ("AND (m.created_at < ? OR "
                                  "(m.created_at = ? AND m.id < ?)) ")
                        params += [cursor[0], cursor[0], cursor[1]]
                    query += "ORDER BY m.created_at DESC, m.id DESC LIMIT ?"
                    fetched = db.execute(query, (*params, limit)).fetchall()
                if not fetched:
                    break
                cursor = fetched[-1]["created_at"], fetched[-1]["id"]
                rows = [row for row in fetched if embeddable_text(row["text"])]
                if not rows:
                    if len(fetched) < limit:
                        break
                    continue
                rows = self._embedding_order(rows)
                for start in range(0, len(rows), batch_size):
                    if stop.is_set():
                        break
                    batch = rows[start:start + batch_size]
                    vectors = self.encode([embedding_text(row) for row in batch])
                    with self.store.connect() as db:
                        db.execute("BEGIN IMMEDIATE")
                        written = 0
                        for row, vector in zip(batch, vectors, strict=True):
                            current = db.execute(
                                "SELECT m.*, v.text_hash AS vector_hash, "
                                "EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id) "
                                "AS tombstoned FROM messages m LEFT JOIN message_vectors v "
                                "ON v.message_id = m.id AND v.model = ? WHERE m.id = ?",
                                (self.store.model, row["id"]),
                            ).fetchone()
                            if (
                                current is None or current["deleted"] or current["tombstoned"]
                                or current["is_bot"]
                            ):
                                continue
                            current_hash = text_hash(current)
                            if current_hash != text_hash(row):
                                # Edited while encoding; rescan afterwards
                                # so the new text is embedded by this call.
                                deferred = True
                                continue
                            if current["vector_hash"] == current_hash:
                                continue
                            db.execute(
                                "INSERT OR REPLACE INTO message_vectors VALUES (?, ?, ?, ?, ?)",
                                (
                                    row["id"], self.store.model, current_hash, len(vector),
                                    vector.astype("<f2").tobytes(),
                                ),
                            )
                            written += 1
                            total += 1
                        if written:
                            db.execute(
                                "UPDATE log_state SET generation = generation + 1 WHERE id = 1"
                            )
                    if progress is not None:
                        progress(total)
                    if self.snapshot_mode:
                        self.request_refresh()
                if len(fetched) < limit:
                    break
        return total

    def _reserve(self, count, dimensions):
        if self.matrix is not None and self.matrix.shape[1] != dimensions:
            raise ValueError("Inconsistent vector dimensions")
        if self.matrix is not None and len(self.matrix) >= count:
            return
        capacity = max(1024, count, self.reserve_hint, int(self.size * 1.5))
        self.reserve_hint = 0
        matrix = self.np.empty((capacity, dimensions), dtype=self.np.float32)
        channels = self.np.empty(capacity, dtype=self.np.int32)
        authors = self.np.empty(capacity, dtype=self.np.int32)
        channel_names = self.np.empty(capacity, dtype=self.np.int32)
        dates = self.np.empty(capacity, dtype="S32")
        active = self.np.zeros(capacity, dtype=bool)
        if self.matrix is not None:
            matrix[:self.size] = self.matrix[:self.size]
            channels[:self.size] = self.channels[:self.size]
            authors[:self.size] = self.authors[:self.size]
            channel_names[:self.size] = self.channel_names[:self.size]
            dates[:self.size] = self.dates[:self.size]
            active[:self.size] = self.active[:self.size]
        self.matrix, self.channels, self.authors, self.active = matrix, channels, authors, active
        self.channel_names, self.dates = channel_names, dates

    def _load(self, rows):
        for row in rows:
            message_id = row["id"]
            index = self.ids.get(message_id)
            if index is not None:
                self.active[index] = False
            self.pending.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            if (
                row["deleted"] or row["tombstoned"] or row["is_bot"]
                or len(row["text"].strip()) < 2
            ):
                continue
            if row["vector"] is None or row["vector_hash"] != text_hash(row):
                if embeddable_text(row["text"]):
                    self.pending.execute(
                        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, 0)",
                        tuple(row[key] for key in (
                            "id", "channel_id", "channel_name", "author_id",
                            "author_name", "created_at"
                        )),
                    )
                continue
            vector = self.np.frombuffer(row["vector"], dtype="<f2").astype(self.np.float32)
            norm = self.np.linalg.norm(vector)
            if (
                len(vector) != row["dimensions"] or not len(vector)
                or not self.np.isfinite(vector).all() or norm == 0
            ):
                raise ValueError("Invalid stored vector")
            if self.snapshot_mode:
                # Published matrix rows are immutable; edits append a new version.
                index = None
            self._reserve(self.size + (index is None), len(vector))
            if index is None:
                index = self.size
                self.size += 1
                self.ids[message_id] = index
                self.message_ids.append(message_id)
                self.sort_keys.append(None)
                self.versions.append(None)
            self.matrix[index] = vector / norm
            self.versions[index] = (row["vector_hash"], row["dimensions"],
                                    hashlib.sha256(row["vector"]).digest())
            self.channels[index] = self.channel_codes.setdefault(
                row["channel_id"], len(self.channel_codes)
            )
            self.authors[index] = self.author_codes.setdefault(
                row["author_id"], len(self.author_codes)
            )
            self.channel_names[index] = self.channel_name_codes.setdefault(
                row["channel_name"], len(self.channel_name_codes)
            )
            self.dates[index] = row["created_at"].encode("ascii")
            self.sort_keys[index] = (row["channel_id"], row["created_at"], message_id)
            self.active[index] = True

    def _select_params(self, static_model):
        return (self.model, static_model) if static_model else (self.model,)

    def _sync(self, db):
        sequence = db.execute("SELECT sequence FROM dense_state WHERE id = 1").fetchone()[0]
        if self.sequence is not None and sequence < self.sequence:
            return False
        static_model = getattr(self.store, "static_model", None)
        static_changed = self.static_model != static_model
        if sequence != self.sequence or self.model != self.store.model or static_changed:
            self.scope_coverage.clear()
        if static_model:
            # Static-distillation cascade: the primary model's vector wins;
            # a static vector fills in only where the primary has none yet.
            select = """
                SELECT m.*, COALESCE(v.text_hash, vs.text_hash) AS vector_hash,
                    COALESCE(v.vector, vs.vector) AS vector,
                    COALESCE(v.dimensions, vs.dimensions) AS dimensions,
                    EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id) AS tombstoned
                FROM messages m
                LEFT JOIN message_vectors v ON v.message_id = m.id AND v.model = ?
                LEFT JOIN message_vectors vs ON vs.message_id = m.id AND vs.model = ?
            """
        else:
            select = """
                SELECT m.*, v.text_hash AS vector_hash, v.vector, v.dimensions,
                    EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id) AS tombstoned
                FROM messages m LEFT JOIN message_vectors v ON v.message_id = m.id AND v.model = ?
            """
        if (self.sequence is None or self.model != self.store.model or static_changed
                or self.snapshot_mode and self.size > max(1024, len(self.ids) * 2)):
            self.ids.clear()
            self.message_ids.clear()
            self.sort_keys.clear()
            self.versions.clear()
            self.channel_codes.clear()
            self.author_codes.clear()
            self.channel_name_codes.clear()
            self.size = 0
            self.matrix = None
            if static_model:
                self.reserve_hint = db.execute(
                    "SELECT count(*) FROM message_vectors WHERE model IN (?, ?)",
                    (self.store.model, static_model),
                ).fetchone()[0] + 1024
            else:
                self.reserve_hint = db.execute(
                    "SELECT count(*) FROM message_vectors WHERE model = ?", (self.store.model,)
                ).fetchone()[0] + 1024
            self.pending.execute("DELETE FROM messages")
            with self.query_cache_lock:
                self.queries.clear()
            self.model = self.store.model
            self.static_model = static_model
            self.rebuilds += 1
            rows = db.execute(select + " ORDER BY m.channel_id, m.created_at, m.id",
                              self._select_params(static_model))
            while batch := rows.fetchmany(512):
                self._load(batch)
        elif sequence != self.sequence:
            changed = [row[0] for row in db.execute(
                "SELECT message_id FROM dense_changes WHERE version > ?", (self.sequence,)
            )]
            for start in range(0, len(changed), 500):
                batch = changed[start:start + 500]
                for message_id in batch:
                    if (index := self.ids.get(message_id)) is not None:
                        self.active[index] = False
                    self.pending.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                self._load(db.execute(
                    select + f" WHERE m.id IN ({','.join('?' for _ in batch)})",
                    (*self._select_params(static_model), *batch),
                ))
        self.pending.commit()
        self.sequence = sequence
        return True

    def _query_vector(self, query):
        key = self.store.model, self.store.max_length, query
        with self.query_cache_lock:
            vector = self.queries.get(key)
        if vector is None:
            vector = self.encode([query], query=True)[0]
        with self.query_cache_lock:
            self.queries[key] = vector
            self.queries.move_to_end(key)
            while len(self.queries) > self.query_cache_size:
                self.queries.popitem(last=False)
        return vector

    def refresh_snapshot(self):
        self.snapshot_mode = True
        with self.cache_lock, self.store.connect() as db:
            db.execute("BEGIN")
            sequence = db.execute("SELECT sequence FROM dense_state WHERE id = 1").fetchone()[0]
            if (self.snapshot is not None and sequence == self.snapshot.sequence
                    and self.snapshot.model == self.store.model):
                return
            try:
                if not self._sync(db):
                    return
            except ValueError:
                self.sequence = None
                raise
            snapshot = copy(self)
            snapshot.snapshot = None
            snapshot.cache_lock = Lock()
            snapshot.active = self.active.copy()
            snapshot.channel_codes = self.channel_codes.copy()
            snapshot.author_codes = self.author_codes.copy()
            snapshot.channel_name_codes = self.channel_name_codes.copy()
            snapshot.message_ids = self.message_ids[:]
            snapshot.sort_keys = self.sort_keys[:]
            snapshot.versions = self.versions[:]
            snapshot.scope_coverage = OrderedDict(self.scope_coverage)
            snapshot.pending = sqlite3.connect(":memory:", check_same_thread=False)
            self.pending.backup(snapshot.pending)
            self.snapshot = snapshot

    def request_refresh(self):
        self.snapshot_mode = True
        if not self.refresh_lock.acquire(blocking=False):
            return
        if monotonic() - self.last_refresh < 1:
            self.refresh_lock.release()
            return
        self.last_refresh = monotonic()

        def refresh():
            try:
                self.refresh_snapshot()
            except Exception:
                logger.warning("Dense snapshot refresh unavailable", exc_info=True)
            finally:
                self.refresh_lock.release()

        self.refresh_thread = Thread(target=refresh, name="dense-snapshot-refresh", daemon=True)
        self.refresh_thread.start()

    def search_snapshot(self, *args, **kwargs):
        snapshot = self.snapshot
        self.request_refresh()
        if snapshot is None or snapshot.model != self.store.model:
            return None
        return snapshot.search(*args, **kwargs, synchronize=False)

    def _coverage(self, permitted, where, args, *, channel, after, before):
        key = where, tuple(args)
        if key not in self.scope_coverage:
            pending = self.pending.execute(
                f"SELECT count(*) FROM messages m WHERE {where}", args
            ).fetchone()[0]
            coverage = 1.0
            if pending:
                scoped = permitted.copy()
                if channel is not None:
                    code = self.channel_codes.get(str(channel), -1)
                    name = self.channel_name_codes.get(str(channel).removeprefix("#"), -1)
                    scoped &= ((self.channels[:self.size] == code)
                               | (self.channel_names[:self.size] == name))
                for operator, value in [(self.np.greater, after), (self.np.less, before)]:
                    if value is not None:
                        scoped &= operator(self.dates[:self.size], _date(value).encode("ascii"))
                embedded = int(scoped.sum())
                coverage = embedded / (embedded + pending)
            self.scope_coverage[key] = coverage
        self.scope_coverage.move_to_end(key)
        coverage = self.scope_coverage[key]
        while len(self.scope_coverage) > self.query_cache_size:
            self.scope_coverage.popitem(last=False)
        return coverage

    def search(
        self, db, query, channel_ids, where, args, limit, *, author_ids=None,
        channel=None, after=None, before=None, synchronize=True,
    ):
        with stage("cache_lock_wait"):
            self.cache_lock.acquire()
        try:
            try:
                if synchronize:
                    with stage("cache_sync"):
                        if not self._sync(db):
                            return None
            except ValueError:
                self.sequence = None
                raise
            if self.matrix is None:
                return None
            codes = [self.channel_codes[str(c)] for c in channel_ids
                     if str(c) in self.channel_codes]
            permitted = self.np.isin(self.channels[:self.size], codes) & self.active[:self.size]
            if author_ids is not None:
                authors = [self.author_codes[a] for a in author_ids if a in self.author_codes]
                permitted &= self.np.isin(self.authors[:self.size], authors)
            count = int(permitted.sum())
            if not count:
                return None
            coverage = self._coverage(
                permitted, where, args, channel=channel, after=after, before=before
            )
            if not synchronize:
                sequence = db.execute("SELECT sequence FROM dense_state WHERE id = 1").fetchone()[0]
                if sequence != self.sequence:
                    coverage = None
            with stage("query_embedding"):
                vector = self._query_vector(query)
            with _BLAS_LOCK, self.threadpool_limits(limits=1, user_api="blas"):
                scores = self.matrix[:self.size] @ vector
            scores[~permitted] = -self.np.inf
            fetch = min(count, max(200, limit * 4))
            while True:
                top = self.np.argpartition(scores, -fetch)[-fetch:]
                cutoff = scores[top].min()
                top = self.np.flatnonzero(scores >= cutoff)
                top = sorted(top, key=lambda i: (-float(scores[i]), self.sort_keys[i]))[:fetch]
                ids = [self.message_ids[i] for i in top]
                if synchronize:
                    valid = {row[0] for row in db.execute(
                        f"SELECT m.id FROM messages m WHERE {where} AND m.is_bot = 0 "
                        "AND m.id IN (SELECT value FROM json_each(?))", [*args, json.dumps(ids)]
                    )}
                else:
                    versions = {self.message_ids[i]: self.versions[i] for i in top}
                    valid = set()
                    for row in db.execute(
                        "SELECT m.*, v.text_hash AS vector_hash, v.dimensions, v.vector "
                        "FROM messages m JOIN message_vectors v ON v.message_id = m.id "
                        f"AND v.model = ? WHERE {where} AND m.is_bot = 0 "
                        "AND m.id IN (SELECT value FROM json_each(?))",
                        [self.model, *args, json.dumps(ids)],
                    ):
                        version = (row["vector_hash"], row["dimensions"],
                                   hashlib.sha256(row["vector"]).digest())
                        if version == versions[row["id"]] and text_hash(row) == row["vector_hash"]:
                            valid.add(row["id"])
                result = [message_id for message_id in ids if message_id in valid]
                if len(result) >= limit or fetch == count:
                    return DenseHits(result[:limit], partial=coverage is None or coverage < 1,
                                     coverage=coverage)
                fetch = min(count, fetch * 2)
        finally:
            self.cache_lock.release()
