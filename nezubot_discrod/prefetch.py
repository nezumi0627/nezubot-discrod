"""Ahead-of-question evidence prefetch for active conversations.

In a channel where a conversation is running, the next question is
very often about the same topic or the same person the conversation
state already settled on. This module fetches that evidence *before*
the question arrives — during the bot's idle time, never while a
reply is being generated, and never under load — and caches it for a
short TTL. When a question does arrive and a fresh entry matches its
subject, pre_retrieve treats the cached hits as already-searched
evidence: they are delivered as their own chunk, count toward the
sufficiency verdict, and stand in for the dense job when they cover
the settled person. A stale or missing entry simply falls back to
the normal live search.

The cache is keyed by the exact audience scope (the channel set the
evidence was fetched under) plus the subject, so evidence never
crosses a permission boundary: a different audience is a different
entry.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

PREFETCH_TTL_SECONDS = 120.0
PREFETCH_MAX_ENTRIES = 32
PREFETCH_MIN_INTERVAL_SECONDS = 45.0
PREFETCH_MAX_LOAD = 3.0
PREFETCH_HITS = 6


def load_ok(loadavg=None):
    """Whether the VM is quiet enough for speculative work. The bot
    shares ~1.5 cores; prefetching into a busy box would slow the
    very answers it is meant to speed up."""
    try:
        current = os.getloadavg()[0] if loadavg is None else loadavg
    except (OSError, AttributeError):
        return False
    return current <= PREFETCH_MAX_LOAD


def person_subject(person_id):
    return ("person", str(person_id))


def topic_subject(topic):
    return ("topic", " ".join(str(topic or "").split()))


class PrefetchCache:
    """Short-TTL store of prefetched archive hits, keyed by
    (audience scope, subject). Entries carry their fetch time so the
    scheduler can honour a per-key minimum interval even after an
    entry expires."""

    def __init__(self, ttl=PREFETCH_TTL_SECONDS, max_entries=PREFETCH_MAX_ENTRIES):
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries = {}
        self._fetched_at = {}

    @staticmethod
    def _key(scope_channels, subject):
        scope = tuple(sorted({str(channel) for channel in scope_channels}))
        return (scope, (subject[0], str(subject[1])))

    def get(self, scope_channels, subject, *, now=None):
        """The fresh entry for a subject, or None. An entry is a dict
        with hits (list of archive hits), query and fetched_at."""
        current = time.time() if now is None else now
        entry = self._entries.get(self._key(scope_channels, subject))
        if entry is None or current - entry["fetched_at"] > self.ttl:
            return None
        return entry

    def put(self, scope_channels, subject, hits, query, *, now=None):
        current = time.time() if now is None else now
        key = self._key(scope_channels, subject)
        self._fetched_at[key] = current
        if not hits:
            self._entries.pop(key, None)
            return None
        entry = {"hits": list(hits), "query": query, "fetched_at": current}
        self._entries[key] = entry
        while len(self._entries) > self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k]["fetched_at"])
            self._entries.pop(oldest, None)
        return entry

    def due(self, scope_channels, subject, *, now=None):
        """Whether a (re)fetch for this key may start: no fresh entry
        and the minimum interval since the last attempt has passed."""
        current = time.time() if now is None else now
        if self.get(scope_channels, subject, now=current) is not None:
            return False
        last = self._fetched_at.get(self._key(scope_channels, subject))
        return last is None or current - last >= PREFETCH_MIN_INTERVAL_SECONDS

    def invalidate_scope(self, scope_channels):
        scope = tuple(sorted({str(channel) for channel in scope_channels}))
        for key in [key for key in self._entries if key[0] == scope]:
            self._entries.pop(key, None)
