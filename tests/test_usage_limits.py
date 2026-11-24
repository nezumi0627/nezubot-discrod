from nezubot_discrod.bot import HELP_TEXT
from nezubot_discrod.usage_limits import build_limit_embed, format_reset, parse_rate_limits

FIVE_HOUR_RESET = 1764054780
WEEKLY_RESET = 1764480000


def snapshot(**overrides):
    base = {
        "limitId": "codex",
        "planType": "plus",
        "primary": {"usedPercent": 23, "windowDurationMins": 300, "resetsAt": FIVE_HOUR_RESET},
        "secondary": {"usedPercent": 41, "windowDurationMins": 10080, "resetsAt": WEEKLY_RESET},
        "credits": None,
    }
    base.update(overrides)
    return base


RESPONSE = {
    "rateLimits": snapshot(),
    "rateLimitsByLimitId": {"codex": snapshot()},
}


def test_parse_extracts_five_hour_and_weekly():
    usage = parse_rate_limits(RESPONSE)
    assert usage["five_hour"]["used_percent"] == 23
    assert usage["five_hour"]["resets_at"] == FIVE_HOUR_RESET
    assert usage["weekly"]["used_percent"] == 41
    assert usage["weekly"]["resets_at"] == WEEKLY_RESET
    assert usage["plan_type"] == "plus"


def test_parse_falls_back_to_single_bucket_view():
    usage = parse_rate_limits({"rateLimits": snapshot()})
    assert usage["five_hour"]["used_percent"] == 23
    assert usage["weekly"]["used_percent"] == 41


def test_parse_classifies_by_duration_when_windows_are_swapped():
    swapped = snapshot(
        primary={"usedPercent": 41, "windowDurationMins": 10080, "resetsAt": WEEKLY_RESET},
        secondary={"usedPercent": 23, "windowDurationMins": 300, "resetsAt": FIVE_HOUR_RESET},
    )
    usage = parse_rate_limits({"rateLimits": swapped})
    assert usage["five_hour"]["used_percent"] == 23
    assert usage["weekly"]["used_percent"] == 41


def test_parse_without_durations_uses_positions():
    bare = snapshot(
        primary={"usedPercent": 10, "resetsAt": None},
        secondary={"usedPercent": 20, "resetsAt": None},
    )
    usage = parse_rate_limits({"rateLimits": bare})
    assert usage["five_hour"]["used_percent"] == 10
    assert usage["weekly"]["used_percent"] == 20


def test_parse_handles_a_single_window():
    usage = parse_rate_limits({"rateLimits": snapshot(secondary=None)})
    assert usage["five_hour"]["used_percent"] == 23
    assert usage["weekly"] is None


def test_parse_rejects_unusable_payloads():
    assert parse_rate_limits(None) is None
    assert parse_rate_limits({}) is None
    assert parse_rate_limits({"rateLimits": {}}) is None
    assert parse_rate_limits({"rateLimits": snapshot(primary=None, secondary=None)}) is None
    broken = snapshot(primary={"windowDurationMins": 300}, secondary={"usedPercent": "41"})
    assert parse_rate_limits({"rateLimits": broken}) is None


def test_limit_embed_shows_remaining_percents_and_resets():
    embed = build_limit_embed(parse_rate_limits(RESPONSE))
    fields = {field.name: field.value for field in embed.fields}
    # used 23% / 41% is rendered as the remaining headroom, not the usage.
    assert "残り 77%" in fields["🕐 5時間枠"]
    assert "使用済み" not in fields["🕐 5時間枠"]
    assert format_reset(FIVE_HOUR_RESET) in fields["🕐 5時間枠"]
    assert "残り 59%" in fields["📅 週間枠"]
    assert "使用済み" not in fields["📅 週間枠"]
    assert format_reset(WEEKLY_RESET) in fields["📅 週間枠"]
    assert "plus" in embed.description


def test_help_lists_limit():
    assert "/limit" in HELP_TEXT
