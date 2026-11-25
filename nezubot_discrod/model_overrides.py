"""Per-user answer-model overrides, chosen with /model.

The default answer model comes from deploy config (AI_MODEL). A
member can pick a different model for their own turns only; the
choice is stored in the server log DB and resolved at the start of
each of that member's turns as "user override, else default".
Nobody else's turns — and the configured default itself — are
affected, and choosing "default" deletes the row outright.

Only two models are selectable (the owner's call: keep the list to
the two that are actually used). ``gpt-6-luna`` runs on the usual
Codex path; ``swe-2-1`` is not a Codex model at all — it routes the
member's turns to the local Devin bridge instead (see
nezubot_discrod.devin_answer), where the model answers in one shot from
the prepared context, without tools. Because swe-2-1 never appears
in the Codex catalog, the /model handler exempts it from the live
catalog check it applies to Codex slugs, so a Codex model that
disappears upstream is refused politely instead of failing the
member's next turn. A stored value that is no longer selectable is
ignored at resolve time for the same reason.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# The Devin backend slug. Selecting it switches the member's answer
# backend from Codex to the Devin bridge; it is not a Codex catalog
# model and must never be sent to the Codex client.
DEVIN_MODEL = "swe-2-1"

# Selectable answer models, in /model choice order.
MODEL_CHOICES = (
    "gpt-6-luna",
    DEVIN_MODEL,
)


def is_devin_model(model):
    return model == DEVIN_MODEL

# The /model choice value that clears the override.
RESET_CHOICE = "default"

OVERRIDES_DDL = """
CREATE TABLE IF NOT EXISTS user_model_overrides (
    user_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def ensure_model_override_schema(db):
    db.execute(OVERRIDES_DDL)


def _now():
    return datetime.now(UTC).isoformat()


def get_model_override(store, user_id):
    """The member's selectable override slug, or None.

    A stored slug that is no longer in MODEL_CHOICES (removed from
    the catalog since it was chosen) resolves as no override, so the
    member silently falls back to the default instead of failing.
    """
    user_id = str(user_id)
    with store.connect() as db:
        ensure_model_override_schema(db)
        row = db.execute(
            "SELECT model FROM user_model_overrides WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    model = row[0]
    return model if model in MODEL_CHOICES else None


def set_model_override(store, user_id, model):
    if model not in MODEL_CHOICES:
        raise ValueError(f"Model is not selectable: {model}")
    user_id = str(user_id)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_model_override_schema(db)
        db.execute(
            "INSERT INTO user_model_overrides (user_id, model, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET"
            " model = excluded.model, updated_at = excluded.updated_at",
            (user_id, model, _now()),
        )


def clear_model_override(store, user_id):
    """Delete the member's override row. Returns True if one existed."""
    user_id = str(user_id)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        ensure_model_override_schema(db)
        cursor = db.execute(
            "DELETE FROM user_model_overrides WHERE user_id = ?", (user_id,)
        )
        return cursor.rowcount > 0


def resolve_model(store, user_id, default):
    """Effective answer model for one member's turn: their override
    when one is stored, otherwise the configured default."""
    if user_id is None:
        return default
    return get_model_override(store, user_id) or default
