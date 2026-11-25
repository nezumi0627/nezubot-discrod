"""Contextual query rewriting and context-linked recall queries.

Short Discord follow-ups — 「これ直せないの？」「俺とかはどう？」
「さっきのやつは？」 — are not self-contained search queries: the
demonstrative points at the reply target, the comparison points at
the previous turn's subject, the topic lives in the conversation
state. Feeding the raw words to FTS/dense retrieves almost nothing
(the measured zero-source problem). This module rewrites such
questions into self-contained search queries from the conversation
state and the reply chain, with deterministic rules only — an extra
LLM call per question would cost more latency than the search
itself. The original question is always kept alongside: people
resolution and event matching run on it, and the rewrite only
replaces the text the archive rankers see.

build_context_query assembles the second recall path (the
Sakana/vex pattern): a dense query built from the live conversation
itself — the reply target and the latest things other people said —
so past messages the discussion is circling around surface even when
the question never names them.
"""

import re

_DEMONSTRATIVE = re.compile(r"これ|それ|あれ|この|その|あの|こいつ|そいつ|あいつ|さっきの")
_FOLLOWUP_START = re.compile(
    r"^(?:それ以外|じゃあ|では|逆に|さっきの|あれは|他には|ついでに|んで|で、|それで|あと)"
)
_SELF_COMPARE = re.compile(
    r"(?:俺|おれ|自分|僕|ぼく|うち|わたし|私)(?:とか|も|はどう|ってどう|は？|は\?)"
)
MAX_REWRITE_CHARS = 500
MAX_CONTEXT_CHARS = 600


def _clip(text, limit):
    body = " ".join(str(text or "").split())
    if len(body) > limit:
        return body[: limit - 1] + "…"
    return body


def needs_rewrite(text):
    """Whether a question depends on context to mean anything to a
    search engine: it carries a demonstrative, opens with a follow-up
    connective, compares the speaker to a running subject, or is a
    very short question hanging on a conversation."""
    body = str(text or "").strip()
    if not body:
        return False
    if _DEMONSTRATIVE.search(body) or _FOLLOWUP_START.search(body):
        return True
    if _SELF_COMPARE.search(body):
        return True
    return len(body) <= 8 and body.endswith(("？", "?", "の", "か"))


def _subject_label(state):
    subject = (state or {}).get("active_subject") or {}
    if subject.get("kind") == "person":
        return subject.get("label") or ""
    return ""


def _reply_target(conversation):
    target = (conversation or {}).get("reply_target") or {}
    if target.get("content"):
        return target
    chain = (conversation or {}).get("reply_chain") or []
    return chain[0] if chain else {}


def rewrite_query(text, *, state=None, conversation=None, speaker_name=None):
    """A self-contained search query for a context-dependent question,
    or None when nothing useful can be added (the caller then searches
    with the original text)."""
    body = str(text or "").strip()
    if not body or not needs_rewrite(body):
        return None
    parts = []
    target = _reply_target(conversation)
    subject_label = _subject_label(state)
    topic = _clip((state or {}).get("topic"), 120)
    if _SELF_COMPARE.search(body) and speaker_name:
        # 「俺とかはどう？」 asks for the same treatment of the speaker
        # as of the running subject/topic: search the speaker against
        # the topic (or the subject) instead of the bare pronoun.
        parts.append(speaker_name)
        parts.append(topic or subject_label)
    elif target.get("content") and _DEMONSTRATIVE.search(body):
        # The demonstrative refers to the replied-to message; its own
        # words are the search signal.
        parts.append(_clip(target["content"], 200))
    elif subject_label and subject_label not in body:
        parts.append(subject_label)
    elif topic and _DEMONSTRATIVE.search(body):
        parts.append(topic)
    parts.append(body)
    rewritten = " ".join(part for part in parts if part).strip()
    if not rewritten or rewritten == body:
        return None
    return rewritten[:MAX_REWRITE_CHARS]


def _transcript_line(line):
    """One transcript line as (speaker name, bare words): the
    "[HH:MM] name: " dressing and the reply annotation stripped."""
    text = re.sub(r"^\[\d{2}:\d{2}\] ", "", line)
    name = ""
    if ": " in text:
        name, text = text.split(": ", 1)
    text = re.sub(r"（↩ .*? への返信）$", "", text).strip()
    return name.strip(), text


def build_context_query(conversation, state=None, speaker_name=None):
    """A dense-search query built from the live conversation: the
    reply target's words, the latest lines the *speaker themselves*
    contributed, and the running topic. None when there is no
    conversation to speak of.

    The transcript tail used to be whoever spoke last — usually
    somebody else — so the context route kept retrieving around the
    previous speaker's words and dragging the answer toward the
    latest topic (the B-3 recency pull). Other people's lines are
    only used when the speaker has said nothing in the window yet
    (a first interjection into a running conversation)."""
    parts = []
    target = _reply_target(conversation)
    if target.get("content"):
        parts.append(_clip(target["content"], 300))
    transcript = str((conversation or {}).get("transcript") or "")
    if transcript:
        lines = [
            line for line in transcript.split("\n")
            if line.strip() and "←" not in line
        ]
        parsed = [_transcript_line(line) for line in lines]
        own = [
            text for name, text in parsed
            if text and speaker_name and name == speaker_name
        ]
        pool = own[-2:] if own else [
            text for _name, text in parsed if text
        ][-1:]
        for text in pool:
            parts.append(_clip(text, 150))
    topic = _clip((state or {}).get("topic"), 120)
    if topic and all(topic not in part for part in parts):
        parts.append(topic)
    query = " ".join(parts).strip()
    if len(query) < 10:
        return None
    return query[:MAX_CONTEXT_CHARS]
