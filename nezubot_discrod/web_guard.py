"""Per-turn outbound payload guard for the web tools.

This replaces the old per-turn taint rule ("once private data and web
content were both read in a turn, all further web access is disabled").
That rule made ordinary research impossible: the automatic archive
pre-retrieval at the start of almost every turn counted as a private
read, so after a single successful search the model could no longer
open any of the results.

The protection goal is unchanged — private server content (log
excerpts, wiki text, other members' messages) must not leak out
through web requests the model crafts — but enforcement now inspects
the outbound payload itself instead of disabling the turn:

- Everything the turn treats as private is registered with the guard
  (the injected pre-retrieval excerpts / wiki context / recent-message
  context from the turn payload, plus every log/wiki tool result).
- Before a web tool runs, its outbound payload (search query, article
  title, fetch URL) is checked against that material. A payload that
  copies a distinctive private fragment — a verbatim span of ~35+
  normalized characters, a long distinctive token (URL, hash, ID-like
  string), or, for queries, a Discord message ID — is blocked, and the
  model is told to rephrase in its own words.
- Fragments that also appear in the user's own messages are exempt:
  the user asking about their own words is not a leak.

Clean payloads (the normal case: the model searches for a topic in
its own words and opens URLs that search results returned) pass
regardless of what the turn has read, so multi-step research —
search, open a listing, open several articles, fetch the next page,
compare sources — works in one turn.
"""

import hashlib
import json
import re

# Verbatim-span detection: normalized (lowercased, whitespace-collapsed)
# character windows. Registration slides by STEP, checking slides by 1,
# so any verbatim span of WINDOW + STEP - 1 characters or more is
# guaranteed to contain a registered window.
WINDOW = 32
STEP = 4
# Whitespace-delimited tokens at least this long are distinctive enough
# to treat as fingerprints (URLs, hashes, long identifiers).
TOKEN_MIN = 16
# Discord snowflake IDs (message / channel / user IDs).
SNOWFLAKE = re.compile(r"(?<!\d)\d{15,20}(?!\d)")

BLOCKED_MESSAGE = (
    "Tool error: this web request was blocked because its {kind} appears to "
    "copy text from private server content read in this turn (server logs, "
    "wiki or other members' messages). Private content must not be sent to "
    "web services. Rephrase the request in your own words — a short topic "
    "summary, never a quote, an excerpt or a message ID — and try again."
)


def _normalize(text):
    return " ".join(str(text).split()).lower()


def _windows(normalized, step):
    if len(normalized) < WINDOW:
        return set()
    return {
        hashlib.sha1(normalized[i:i + WINDOW].encode()).hexdigest()
        for i in range(0, len(normalized) - WINDOW + 1, step)
    }


def _tokens(normalized):
    return {
        token.strip("()[]{}<>\"'`,;")
        for token in normalized.split()
        if len(token.strip("()[]{}<>\"'`,;")) >= TOKEN_MIN
    }


def _ids(text):
    return set(SNOWFLAKE.findall(str(text)))


class OutboundGuard:
    """Tracks one turn's private material and vets web payloads."""

    def __init__(self):
        self._private_windows = set()
        self._private_tokens = set()
        self._private_ids = set()
        self._user_windows = set()
        self._user_tokens = set()
        self._user_ids = set()

    def note_user_text(self, text):
        """Register the user's own words; they exempt matching fragments."""
        if not text:
            return
        normalized = _normalize(text)
        self._user_windows |= _windows(normalized, 1)
        self._user_tokens |= _tokens(normalized)
        self._user_ids |= _ids(text)

    def note_private(self, text):
        """Register private content the turn has read (any str-able)."""
        if text is None:
            return
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return
        normalized = _normalize(text)
        self._private_windows |= _windows(normalized, STEP)
        self._private_tokens |= _tokens(normalized)
        self._private_ids |= _ids(text)

    def note_private_value(self, value):
        """Register every string inside a nested payload value."""
        if isinstance(value, str):
            self.note_private(value)
        elif isinstance(value, dict):
            for item in value.values():
                self.note_private_value(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.note_private_value(item)

    def seed_from_user_payload(self, text):
        """Seed the guard from one sent user message.

        Structured turns carry a JSON payload whose "message" field is
        the user's own text and whose other fields hold injected
        private context (retrieved excerpts, wiki context, recent
        messages, conversation transcript). Bare turns are plain user
        text. Assistant messages are never seeded: they were already
        posted in the channel.
        """
        if not text:
            return
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, list):
            # A wrapped message list (the persistent-thread input):
            # seed from each message's content in turn.
            for item in payload:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    self.seed_from_user_payload(item["content"])
            return
        if not isinstance(payload, dict):
            self.note_user_text(text)
            return
        for key, value in payload.items():
            if key == "message":
                if isinstance(value, str):
                    self.note_user_text(value)
            else:
                self.note_private_value(value)

    def check(self, payload, *, kind):
        """Return a block reason for an outbound payload, or None.

        kind is "query" (search query / article title) or "url"
        (fetch target). Message IDs are only checked on queries: a
        fetch URL legitimately contains structural IDs (e.g. a Discord
        message link the user posted), while an ID inside a search
        query can only have been copied from private content.
        """
        if not payload or not isinstance(payload, str):
            return None
        normalized = _normalize(payload)
        windows = _windows(normalized, 1) - self._user_windows
        if windows & self._private_windows:
            return "verbatim fragment of private content"
        tokens = _tokens(normalized) - self._user_tokens
        if tokens & self._private_tokens:
            return "distinctive token copied from private content"
        if kind == "query":
            ids = _ids(payload) - self._user_ids
            if ids & self._private_ids:
                return "message ID copied from private content"
        return None

    def check_call(self, name, args):
        """Vet one web tool call; return the tool-error text or None."""
        if not isinstance(args, dict):
            return None
        checks = []
        if name in {"web_search", "wikipedia_search"}:
            checks.append(("query", args.get("query"), "query"))
        elif name == "wikipedia_read":
            checks.append(("query", args.get("title"), "article title"))
        elif name in {"read_web_page", "browse_page"}:
            checks.append(("url", args.get("url"), "URL"))
        for kind, payload, label in checks:
            if self.check(payload, kind=kind):
                return BLOCKED_MESSAGE.format(kind=label)
        return None
