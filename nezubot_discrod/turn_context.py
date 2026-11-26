"""Per-turn correlation id.

One Assistant turn fans out into separately logged stages — the
automatic retrieval diagnostics (server_log_context), the provider
call (codex) and the Assistant's own completion line — and the
2026-11-26 measurement pass found they could only be paired by
file order, because nothing tied the three log systems together.

A turn id is minted when a turn starts (Assistant.reply) and
carried on a contextvar, following the pattern of
server_log_timing.CURRENT: tasks created inside the turn (the
pre-retrieval task) inherit a copy of the context, and calls
awaited down the chain (the Codex client, under its lock) still
see the value. Callers outside a turn (rate-limit probes, the
curator) simply read None.
"""

import uuid
from contextvars import ContextVar

CURRENT_TURN_ID = ContextVar("nezubot_turn_id", default=None)


def new_turn_id():
    return uuid.uuid4().hex[:12]


def start_turn():
    """Mint this task's turn id, bind it, and return it."""
    turn_id = new_turn_id()
    CURRENT_TURN_ID.set(turn_id)
    return turn_id


def current_turn_id():
    return CURRENT_TURN_ID.get()
