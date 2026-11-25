import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.bot import show_model
from nezubot_discrod.codex import CodexClient
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.feature_loader import BUILTIN_COMMANDS
from nezubot_discrod.model_overrides import (
    DEVIN_MODEL,
    MODEL_CHOICES,
    RESET_CHOICE,
    clear_model_override,
    get_model_override,
    is_devin_model,
    resolve_model,
    set_model_override,
)
from nezubot_discrod.server_log import ServerLog

GUILD_ID = 123
DEFAULT = "gpt-6-luna"
OVERRIDE = DEVIN_MODEL  # "swe-2-1"


@pytest.fixture()
def store(tmp_path):
    return ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")


# --- store level -------------------------------------------------------


def test_resolve_without_override_returns_default(store):
    assert get_model_override(store, 111) is None
    assert resolve_model(store, 111, DEFAULT) == DEFAULT
    assert resolve_model(store, None, DEFAULT) == DEFAULT


def test_set_then_resolve_returns_override(store):
    set_model_override(store, 111, OVERRIDE)
    assert get_model_override(store, 111) == OVERRIDE
    assert resolve_model(store, 111, DEFAULT) == OVERRIDE
    # The row is persisted with a timestamp.
    with store.connect() as db:
        row = db.execute(
            "SELECT model, updated_at FROM user_model_overrides"
            " WHERE user_id = '111'"
        ).fetchone()
    assert row[0] == OVERRIDE and row[1]


def test_override_is_per_user(store):
    set_model_override(store, 111, OVERRIDE)
    set_model_override(store, 222, DEFAULT)
    assert get_model_override(store, 111) == OVERRIDE
    assert get_model_override(store, 222) == DEFAULT
    assert resolve_model(store, 111, DEFAULT) == OVERRIDE
    # A member with no row is untouched by either override.
    assert get_model_override(store, 333) is None
    assert resolve_model(store, 333, DEFAULT) == DEFAULT


def test_set_replaces_previous_override(store):
    set_model_override(store, 111, OVERRIDE)
    set_model_override(store, 111, DEFAULT)
    assert get_model_override(store, 111) == DEFAULT


def test_clear_restores_default(store):
    set_model_override(store, 111, OVERRIDE)
    assert clear_model_override(store, 111) is True
    assert get_model_override(store, 111) is None
    assert resolve_model(store, 111, DEFAULT) == DEFAULT
    assert clear_model_override(store, 111) is False


def test_set_rejects_unselectable_model(store):
    with pytest.raises(ValueError):
        set_model_override(store, 111, "gpt-reserve")
    with pytest.raises(ValueError):
        set_model_override(store, 111, "codex-auto-review")
    # Selectable under v1.14.0, dropped when the list was cut to two.
    with pytest.raises(ValueError):
        set_model_override(store, 111, "gpt-6.2-sol")
    with pytest.raises(ValueError):
        set_model_override(store, 111, "gpt-5.6-terra")
    assert get_model_override(store, 111) is None


def test_stale_stored_slug_resolves_as_default(store):
    # A slug that was selectable when chosen (gpt-6.2-sol under
    # v1.14.0) but has since been dropped must not break the
    # member's turns: it resolves as no override.
    with store.connect() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS user_model_overrides ("
            " user_id TEXT PRIMARY KEY, model TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO user_model_overrides VALUES ('111', 'gpt-6.2-sol', 't')"
        )
    assert get_model_override(store, 111) is None
    assert resolve_model(store, 111, DEFAULT) == DEFAULT


def test_choices_are_luna_and_devin_only():
    assert MODEL_CHOICES == ("gpt-6-luna", "swe-2-1")
    assert DEVIN_MODEL == "swe-2-1"
    assert is_devin_model("swe-2-1")
    assert not is_devin_model("gpt-6-luna")
    assert "gpt-reserve" not in MODEL_CHOICES
    assert "codex-auto-review" not in MODEL_CHOICES


# --- Codex catalog accessor --------------------------------------------


def test_catalog_slugs_none_until_loaded_then_slugs():
    client = CodexClient(SimpleNamespace())
    assert client.catalog_slugs() is None
    client._catalog_models = [{"slug": "gpt-6-luna"}, {"slug": "gpt-6.2-sol"}]
    assert client.catalog_slugs() == {"gpt-6-luna", "gpt-6.2-sol"}


# --- /model handler ------------------------------------------------------


def make_interaction(user_id=111, guild_id=GUILD_ID):
    return SimpleNamespace(
        guild_id=guild_id,
        user=SimpleNamespace(id=user_id),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def make_model_bot(store, *, slugs=None, with_accessor=True):
    ai_client = SimpleNamespace()
    if with_accessor:
        ai_client.catalog_slugs = lambda: slugs
    return SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID, model=DEFAULT),
        server_log=store,
        ai_client=ai_client,
    )


def sent_text(interaction):
    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    assert kwargs["ephemeral"] is True
    return args[0]


def test_show_model_set_is_ephemeral_and_shows_before_after(store):
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, OVERRIDE))
    text = sent_text(interaction)
    assert f"{DEFAULT} → {OVERRIDE}" in text
    assert resolve_model(store, 111, DEFAULT) == OVERRIDE


def test_show_model_second_switch_reports_current_override(store):
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    asyncio.run(show_model(make_interaction(), bot, OVERRIDE))
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, DEFAULT))
    assert f"{OVERRIDE} → {DEFAULT}" in sent_text(interaction)


def test_show_model_reset_clears_override(store):
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    asyncio.run(show_model(make_interaction(), bot, OVERRIDE))
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, RESET_CHOICE))
    text = sent_text(interaction)
    assert "デフォルトに戻した" in text
    assert OVERRIDE in text and DEFAULT in text
    assert get_model_override(store, 111) is None


def test_show_model_reset_without_override_is_gentle(store):
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, RESET_CHOICE))
    assert "もともとデフォルト" in sent_text(interaction)


def test_show_model_rejects_unlisted_choice(store):
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, "gpt-6.2-sol"))
    assert "選べない" in sent_text(interaction)
    assert get_model_override(store, 111) is None


def test_show_model_rejects_choice_missing_from_live_catalog(store):
    # The live Codex catalog is loaded but no longer carries
    # gpt-6-luna: choosing it is refused politely, nothing stored.
    bot = make_model_bot(store, slugs={"gpt-5.6-luna"})
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, DEFAULT))
    text = sent_text(interaction)
    assert "カタログに無い" in text
    assert get_model_override(store, 111) is None


def test_show_model_accepts_devin_despite_live_catalog(store):
    # swe-2-1 is a Devin backend and never appears in the Codex
    # catalog; the catalog check must not refuse it, and the
    # confirmation notes that it answers without tools.
    bot = make_model_bot(store, slugs={DEFAULT})
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, OVERRIDE))
    text = sent_text(interaction)
    assert f"{DEFAULT} → {OVERRIDE}" in text
    assert "ツール" in text
    assert resolve_model(store, 111, DEFAULT) == OVERRIDE


def test_show_model_accepts_when_catalog_not_loaded_yet(store):
    bot = make_model_bot(store, slugs=None)
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, OVERRIDE))
    assert f"{DEFAULT} → {OVERRIDE}" in sent_text(interaction)
    assert resolve_model(store, 111, DEFAULT) == OVERRIDE


def test_show_model_accepts_without_catalog_accessor(store):
    bot = make_model_bot(store, with_accessor=False)
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, OVERRIDE))
    assert resolve_model(store, 111, DEFAULT) == OVERRIDE


def test_show_model_wrong_guild_refuses_and_writes_nothing(store):
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    interaction = make_interaction(guild_id=999)
    asyncio.run(show_model(interaction, bot, OVERRIDE))
    assert "指定サーバー" in sent_text(interaction)
    assert get_model_override(store, 111) is None


def test_show_model_db_failure_is_ephemeral_notice(store, monkeypatch):
    import nezubot_discrod.bot as bot_module

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk gone")

    monkeypatch.setattr(bot_module, "resolve_model", broken)
    bot = make_model_bot(store, slugs=set(MODEL_CHOICES))
    interaction = make_interaction()
    asyncio.run(show_model(interaction, bot, OVERRIDE))
    assert "失敗した" in sent_text(interaction)


def test_model_name_is_reserved_for_builtins():
    assert "model" in BUILTIN_COMMANDS


# --- Assistant wiring ----------------------------------------------------


def recording_client():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=None,
            sources=(),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None)
                )
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=create))
        )
    )
    return client, calls


def test_assistant_codex_path_uses_override_and_payload_names_it(store):
    # The configured default here differs from the override so the
    # resolution is observable end to end.
    set_model_override(store, 111, DEFAULT)
    captured = []

    async def wiki_completion(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            usage=None,
            sources=(),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None)
                )
            ],
        )

    client, _calls = recording_client()
    ai = Assistant(
        client, "gpt-5.6-luna", SandboxSettings(),
        server_log=store, wiki_completion=wiki_completion,
    )
    asyncio.run(ai.reply(
        1, "hello", member_id=111,
        speaker_names={
            "username": "someone",
            "display_name": "Someone",
            "server_nickname": None,
        },
    ))
    assert captured[0]["model"] == DEFAULT
    payload = json.loads(captured[0]["messages"][-1]["content"])
    assert payload["assistant"]["model"] == DEFAULT
    # Another member stays on the configured default.
    asyncio.run(ai.reply(
        1, "hello", member_id=222,
        speaker_names={
            "username": "other",
            "display_name": "Other",
            "server_nickname": None,
        },
    ))
    assert captured[1]["model"] == "gpt-5.6-luna"
