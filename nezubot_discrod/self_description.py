"""The bot's spoken description of its own search setup, generated
from the same configuration the search code reads.

prompts.py used to hardcode this text, and it silently went stale:
it still described the pre-v1.26 setup ("multilingual E5 by default")
after the ruri switch and the v1.29.0 restructure. Building it here —
from the deployment environment the search modules read (the
SERVER_LOG_* values in deploy.env) and the retrieval constants
defined in server_log.py — means a config change plus a restart is
all it takes for the description to follow; there is no second copy
to keep in sync.

The current version reaches the prompt through the existing
format_latest_update() injection in prompts.py (updates.py's latest
entry), so it is not repeated here.

Only values with a conservative model-tag shape are interpolated, so
a malformed deployment value degrades to generic wording instead of
corrupting the prompt. Nothing interpolated here is a secret: model
tags and retrieval parameters are not credentials.
"""

import os
import re

from nezubot_discrod.server_log import (
    DEFAULT_MODEL,
    POOL_FLOOR,
    REPLY_GRAPH_MAX_NEIGHBORS,
    REPLY_GRAPH_SEED_LIMIT,
    RRF_K,
)

_MODEL_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")


def _model_tag(raw):
    """The configured model tag, or None when unset or not tag-shaped."""
    if raw and _MODEL_TAG.match(raw.strip()):
        return raw.strip()
    return None


def _main_model():
    # Mirrors ServerLog.__init__: SERVER_LOG_EMBEDDING_MODEL wins,
    # DEFAULT_MODEL is the fallback.
    return _model_tag(os.environ.get("SERVER_LOG_EMBEDDING_MODEL")) or DEFAULT_MODEL


def _aux_model():
    # Mirrors ServerLog.__init__: unset means no auxiliary dense route.
    return _model_tag(os.environ.get("SERVER_LOG_HYBRID_MODEL"))


def _aux_gate_mode():
    # Mirrors pre_retrieve in server_log_context: only the exact
    # value "grounding" closes the gate once a person resolves;
    # anything else (including unset) means aux always fires.
    if os.environ.get("SERVER_LOG_AUX_GATE", "always").strip().lower() == "grounding":
        return "grounding"
    return "always"


def _reply_graph_enabled():
    # Mirrors pre_retrieve in server_log_context: only the exact
    # value "off" disables reply-graph expansion.
    return os.environ.get("SERVER_LOG_REPLY_GRAPH", "on").strip().lower() != "off"


def build_search_description() -> str:
    """The parenthetical describing the deployed server-log search.

    Slots into SYSTEM's capabilities sentence where the hardcoded
    "(SQLite FTS5 plus dense embeddings, ...)" text used to be.
    """
    text = f"SQLite FTS5 plus dense embeddings — main model {_main_model()}"
    aux = _aux_model()
    if aux:
        text += f", auxiliary model {aux}"
        if aux.endswith("-ctx"):
            # The code's own convention (DenseIndex.contextual in
            # server_log_dense): a -ctx tag means the documents were
            # embedded together with the message each one replies to.
            text += ", embedded with the message each one replies to"
        if _aux_gate_mode() == "always":
            text += ", fused into every search"
        else:
            text += ", fused in unless a person is resolved"
    text += (
        f" — combined by reciprocal rank fusion (k={RRF_K}) "
        f"over a candidate pool of {POOL_FLOOR}, "
        "with automatic pre-retrieval for questions about people and past events"
    )
    if _reply_graph_enabled():
        text += (
            ", plus reply-graph expansion adding reply parents and "
            f"children of the top {REPLY_GRAPH_SEED_LIMIT} hits, "
            f"up to {REPLY_GRAPH_MAX_NEIGHBORS} extra messages"
        )
    return text
