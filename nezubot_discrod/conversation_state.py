"""Structured short-term memory of one channel's live conversation.

The transcript the model receives is a flat slice of the latest
messages; in a multi-party Discord channel several topics interleave
there, and short follow-ups (「これ直せないの？」「俺とかはどう？」)
only make sense against what the conversation has already settled.
This module keeps that settled structure per channel: the topic under
discussion, who is taking part, the active subject (a person or the
message being discussed), facts established so far — each with the
message id that backs it, marked confirmed or still just a claim under
discussion — solutions already attempted, questions still open, and
the last turn's intent.

Updates are deterministic rules over the live conversation context
(reply chain, mentions, grounding) and the retrieval outcome; no
extra model call is made. State is persisted in the server log
database in its own table so it survives restarts, expires after
STATE_TTL_SECONDS of silence, and resets its topic fields when the
conversation visibly switches topic. Everything is best-effort: a
state failure never blocks an answer.
"""

import json
import logging
import re
import sqlite3
import time

logger = logging.getLogger(__name__)

STATE_TTL_SECONDS = 1800.0
MAX_PARTICIPANTS = 8
MAX_FACTS = 4
MAX_ATTEMPTED = 4
MAX_UNRESOLVED = 3
TOPIC_CLIP = 80
FACT_CLIP = 160

_DDL = """
CREATE TABLE IF NOT EXISTS conversation_states (
    scope TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL
)
"""

_SOLUTION_REQUEST = re.compile(
    r"解決|直せ|直し|なおせ|治せ|できないの|出来ないの|どうすれば|なんとか|対処"
)
_FOLLOWUP = re.compile(
    r"^(?:それ以外|じゃあ|では|逆に|さっきの|あれは|他には|ついでに|んで|で、|それで|あと)"
)
_PERSON_QUESTION = re.compile(r"誰|どんな人|について|ってどんな|とは")
_QUESTION_FORM = re.compile(r"[？?]\s*$|(?:の|だ|です|ます)か[？?]\s*$|ないの[？?]\s*$")


def _clip(text, limit):
    body = " ".join(str(text or "").split())
    if len(body) > limit:
        return body[: limit - 1] + "…"
    return body


def classify_intent(text, *, has_reply=False):
    """A coarse intent for the latest message, used to seed follow-up
    handling: a solution request about the reply target reads very
    differently from a fresh question."""
    body = str(text or "").strip()
    if _SOLUTION_REQUEST.search(body):
        return "request_solution"
    if _FOLLOWUP.search(body):
        return "followup"
    if _PERSON_QUESTION.search(body):
        return "person_question"
    if _QUESTION_FORM.search(body) or body.endswith(("？", "?")):
        return "question"
    return "statement"


def _participants(anchor, conversation):
    people = []
    seen = set()

    def add(person_id, name):
        if person_id is None:
            return
        key = str(person_id)
        if key in seen:
            return
        seen.add(key)
        people.append({"id": key, "name": name or key})

    if anchor is not None:
        add(anchor.get("author_id"), anchor.get("author_name"))
    for item in (conversation or {}).get("reply_chain") or []:
        add(item.get("author_id"), item.get("author"))
    for item in (conversation or {}).get("mentions") or []:
        add(item.get("id"), item.get("display_name"))
    grounded = (conversation or {}).get("grounded_person") or {}
    add(grounded.get("id"), grounded.get("display_name"))
    return people[:MAX_PARTICIPANTS]


def _subject(conversation, resolved_people, names):
    grounded = (conversation or {}).get("grounded_person") or {}
    if grounded.get("id"):
        return {
            "kind": "person",
            "id": str(grounded["id"]),
            "label": grounded.get("display_name") or str(grounded["id"]),
            "author_id": None,
        }
    if resolved_people and len(resolved_people) == 1:
        person = str(resolved_people[0])
        return {
            "kind": "person",
            "id": person,
            "label": names.get(person) or person,
            "author_id": None,
        }
    chain = (conversation or {}).get("reply_chain") or []
    if chain:
        head = chain[0]
        return {
            "kind": "message",
            "id": str(head.get("message_id") or ""),
            "label": _clip(head.get("content"), 60),
            "author_id": str(head["author_id"]) if head.get("author_id") else None,
        }
    return None


def _topic_switched(previous, question, conversation, subject):
    """A fresh self-contained question with no reply and no reference
    to the running subject starts a new topic; the old topic fields
    would otherwise poison rewrites and the assembled context."""
    if not previous:
        return False
    if (conversation or {}).get("reply_chain"):
        return False
    body = str(question or "")
    if len(body) < 12 or _FOLLOWUP.search(body):
        return False
    old = previous.get("active_subject") or {}
    if old.get("kind") == "person" and old.get("label") and old["label"] in body:
        return False
    if subject is not None and old.get("id") and subject.get("id") == old.get("id"):
        return False
    return True


def build_state_update(previous, *, now, anchor, question, conversation,
                       resolved_people=(), answer_summary=None):
    """Fold one answered turn into the conversation state.

    previous is the stored state dict or None. anchor is
    {"id", "author_id", "author_name", "content"} for the message being
    answered. Returns the new state dict (not yet persisted)."""
    conversation = conversation or {}
    participants = _participants(anchor, conversation)
    names = {person["id"]: person["name"] for person in participants}
    for item in conversation.get("mentions") or []:
        if item.get("id"):
            names.setdefault(str(item["id"]), item.get("display_name") or str(item["id"]))
    subject = _subject(conversation, resolved_people, names)
    switched = _topic_switched(previous, question, conversation, subject)
    if subject is None and previous and not switched:
        subject = previous.get("active_subject")
    chain = conversation.get("reply_chain") or []

    if chain:
        topic = _clip(chain[0].get("content"), TOPIC_CLIP)
    elif subject is not None and subject.get("kind") == "message":
        topic = subject.get("label") or ""
    elif previous and not switched and previous.get("topic"):
        topic = previous["topic"]
    else:
        topic = _clip(question, TOPIC_CLIP)

    facts = []
    if chain:
        head = chain[0]
        facts.append({
            "text": _clip(head.get("content"), FACT_CLIP),
            "message_id": str(head.get("message_id") or ""),
            "confirmed": False,
            "kind": "claim_under_discussion",
        })
    if previous and not switched:
        for fact in previous.get("established_facts") or []:
            if fact.get("kind") == "claim_under_discussion":
                continue
            if all(fact.get("message_id") != item.get("message_id") for item in facts):
                facts.append(fact)
    facts = facts[:MAX_FACTS]

    attempted = list((previous or {}).get("attempted_solutions") or []) \
        if previous and not switched else []
    if answer_summary:
        entry = {"text": _clip(answer_summary, FACT_CLIP),
                 "message_id": str((anchor or {}).get("id") or "")}
        attempted = [entry] + [
            item for item in attempted if item.get("text") != entry["text"]
        ]
    attempted = attempted[:MAX_ATTEMPTED]

    intent = classify_intent(question, has_reply=bool(chain))
    unresolved = list((previous or {}).get("unresolved_questions") or []) \
        if previous and not switched else []
    if intent != "statement" and anchor is not None:
        entry = {
            "text": _clip(question, FACT_CLIP),
            "message_id": str(anchor.get("id") or ""),
            "author_id": str(anchor.get("author_id") or ""),
        }
        unresolved = [entry] + [
            item for item in unresolved
            if item.get("message_id") != entry["message_id"]
        ]
    unresolved = unresolved[:MAX_UNRESOLVED]

    return {
        "topic": topic,
        "participants": participants,
        "active_subject": subject,
        "established_facts": facts,
        "attempted_solutions": attempted,
        "unresolved_questions": unresolved,
        "last_intent": intent,
        "turn_count": int((previous or {}).get("turn_count") or 0) + 1,
        "last_message_id": str((anchor or {}).get("id") or ""),
        "updated_at": now,
    }


def state_context_view(state):
    """The compact structured summary handed to the model as the
    conversation_state field: facts keep their source message ids and
    their confirmed/claim status, so the model can weigh them."""
    if not state:
        return None
    view = {}
    facts = state.get("established_facts") or []
    if facts:
        view["established_facts"] = facts
    attempted = state.get("attempted_solutions") or []
    if attempted:
        view["attempted_solutions"] = [item.get("text") for item in attempted]
    unresolved = state.get("unresolved_questions") or []
    if unresolved:
        view["unresolved_questions"] = [item.get("text") for item in unresolved]
    if state.get("last_intent"):
        view["last_intent"] = state["last_intent"]
    participants = state.get("participants") or []
    if participants:
        view["participants"] = [person.get("name") for person in participants]
    return view or None


def active_topic_view(state, conversation=None):
    """The active_topic field: the running topic plus the subject it
    hangs on. Falls back to this turn's grounding when no state has
    accumulated yet."""
    if state:
        view = {}
        if state.get("topic"):
            view["topic"] = state["topic"]
        if state.get("active_subject"):
            view["subject"] = state["active_subject"]
        if view:
            return view
    grounded = (conversation or {}).get("grounded_person") or {}
    if grounded.get("id"):
        return {
            "subject": {
                "kind": "person",
                "id": str(grounded["id"]),
                "label": grounded.get("display_name") or str(grounded["id"]),
            }
        }
    return None


class ConversationStateStore:
    """Per-channel conversation state persisted in the server log DB."""

    def __init__(self, server_log):
        self.server_log = server_log

    def _ensure(self, db):
        db.execute(_DDL)

    def load(self, scope, *, now=None):
        """The stored state for a channel scope, or None when absent
        or older than the TTL (a silent channel's topic is over)."""
        try:
            with self.server_log.connect() as db:
                self._ensure(db)
                row = db.execute(
                    "SELECT state, updated_at FROM conversation_states WHERE scope = ?",
                    (str(scope),),
                ).fetchone()
        except (OSError, sqlite3.Error):
            logger.warning("Conversation state load failed", exc_info=True)
            return None
        if row is None:
            return None
        current = time.time() if now is None else now
        if current - float(row[1]) > STATE_TTL_SECONDS:
            return None
        try:
            state = json.loads(row[0])
        except ValueError:
            return None
        return state if isinstance(state, dict) else None

    def save(self, scope, state):
        try:
            with self.server_log.connect() as db:
                self._ensure(db)
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT INTO conversation_states VALUES (?, ?, ?) "
                    "ON CONFLICT(scope) DO UPDATE SET state = excluded.state, "
                    "updated_at = excluded.updated_at",
                    (str(scope), json.dumps(state, ensure_ascii=False),
                     float(state.get("updated_at") or time.time())),
                )
        except (OSError, sqlite3.Error):
            logger.warning("Conversation state save failed", exc_info=True)

    def clear(self, scope):
        try:
            with self.server_log.connect() as db:
                self._ensure(db)
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "DELETE FROM conversation_states WHERE scope = ?", (str(scope),)
                )
        except (OSError, sqlite3.Error):
            logger.warning("Conversation state clear failed", exc_info=True)

    def update(self, scope, *, anchor, question, conversation,
               resolved_people=(), answer_summary=None, now=None):
        current = time.time() if now is None else now
        previous = self.load(scope, now=current)
        state = build_state_update(
            previous, now=current, anchor=anchor, question=question,
            conversation=conversation, resolved_people=resolved_people,
            answer_summary=answer_summary,
        )
        self.save(scope, state)
        return state
