"""Parsing and rendering for the /limit subscription-usage command.

The raw payload is the Codex app-server's GetAccountRateLimitsResponse
(account/rateLimits/read): a snapshot whose primary window is the
5-hour limit and whose secondary window is the weekly limit, each with
a usedPercent, a windowDurationMins and a resetsAt unix timestamp.
Windows are classified by duration when the backend reports it, with
the primary/secondary positions as the fallback. Anything unparsable
yields None so the caller shows an honest failure instead of a guess.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
FIVE_HOUR_MAX_MINS = 360
WEEKLY_MIN_MINS = 1440


def _window(raw):
    if not isinstance(raw, dict):
        return None
    percent = raw.get("usedPercent")
    if not isinstance(percent, int) or isinstance(percent, bool):
        return None
    resets = raw.get("resetsAt")
    duration = raw.get("windowDurationMins")
    return {
        "used_percent": percent,
        "resets_at": resets if isinstance(resets, int) else None,
        "window_mins": duration if isinstance(duration, int) else None,
    }


def _snapshot(response):
    by_id = response.get("rateLimitsByLimitId")
    if isinstance(by_id, dict) and by_id:
        snapshot = by_id.get("codex")
        if isinstance(snapshot, dict):
            return snapshot
        for candidate in by_id.values():
            if isinstance(candidate, dict):
                return candidate
    snapshot = response.get("rateLimits")
    return snapshot if isinstance(snapshot, dict) else None


def parse_rate_limits(response):
    """Extract the 5-hour and weekly usage windows, or None."""
    if not isinstance(response, dict):
        return None
    snapshot = _snapshot(response)
    if snapshot is None:
        return None
    primary = _window(snapshot.get("primary"))
    secondary = _window(snapshot.get("secondary"))
    five_hour = weekly = None
    for window in (primary, secondary):
        if window is None or window["window_mins"] is None:
            continue
        if window["window_mins"] <= FIVE_HOUR_MAX_MINS:
            five_hour = five_hour or window
        elif window["window_mins"] >= WEEKLY_MIN_MINS:
            weekly = weekly or window
    if five_hour is None and primary is not weekly:
        five_hour = primary
    if weekly is None and secondary is not five_hour:
        weekly = secondary
    if five_hour is None and weekly is None:
        return None
    plan = snapshot.get("planType")
    return {
        "five_hour": five_hour,
        "weekly": weekly,
        "plan_type": plan if isinstance(plan, str) else None,
    }


def format_reset(resets_at):
    """A resetsAt unix timestamp as a Japan-time 'MM/DD HH:MM' string."""
    if resets_at is None:
        return None
    return datetime.fromtimestamp(resets_at, JST).strftime("%m/%d %H:%M")


def build_limit_embed(usage):
    """Render parsed usage windows as the /limit embed."""
    import discord

    description = "ChatGPT / Codex サブスクリプションの利用枠"
    if usage.get("plan_type"):
        description += f"（プラン: {usage['plan_type']}）"
    embed = discord.Embed(
        title="nezubot-discrod 利用枠",
        description=description,
        color=discord.Color.blurple(),
    )
    for label, key in (("🕐 5時間枠", "five_hour"), ("📅 週間枠", "weekly")):
        window = usage.get(key)
        if window is None:
            continue
        # The user reads the headroom, not the consumption: show what
        # is left in the window.
        value = f"残り {100 - window['used_percent']}%"
        reset = format_reset(window["resets_at"])
        if reset:
            value += f"\nリセット: {reset}（日本時間）"
        embed.add_field(name=label, value=value, inline=True)
    return embed
