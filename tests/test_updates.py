from datetime import date

from nezubot_discrod.prompts import SYSTEM
from nezubot_discrod.updates import (
    UPDATES,
    format_latest_update,
    format_update_history,
    latest_update,
    latest_version,
)


def test_history_is_newest_first_with_unique_versions():
    versions = [entry.version for entry in UPDATES]
    assert len(versions) == len(set(versions))
    assert latest_version() == versions[0]
    assert latest_update() is UPDATES[0]
    for entry in UPDATES:
        date.fromisoformat(entry.date)
        assert entry.changes


def test_latest_entry_covers_todays_shipped_changes():
    history = format_update_history()
    for fact in (
        "n! プレフィックスコマンド",
        "ティーザー",
        "gpt-6-luna",
        "教師ベクトル",
        "人物解決",
    ):
        assert fact in history


def test_system_prompt_carries_identity_and_latest_update():
    assert "You are nezubot-discrod, the Discord bot" in SYSTEM
    assert "learned from its logs" in SYSTEM
    assert "何が変わった？" in SYSTEM
    assert format_latest_update() in SYSTEM
