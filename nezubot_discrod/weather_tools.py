"""Structured weather and clock tools (Open-Meteo; no API key).

Deliberately separate from the web tools. The endpoints are fixed
public APIs, the arguments are a place name or an IANA timezone, and
the results are structured observations rather than arbitrary page
text, so these tools do not take part in the web tools' per-turn
taint accounting (a weather lookup cannot be turned into a fetch of
an attacker-chosen URL, and its output cannot instruct further
fetches). Fetching reuses the WebTools connection pool, so the same
egress proxy, SSRF vetting and timeout discipline apply.
"""

import json
import re
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nezubot_discrod.web_tools import WebError, WebTools

JST = ZoneInfo("Asia/Tokyo")
_WEEKDAYS = "月火水木金土日"
_TIMEZONE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*(/[A-Za-z0-9_+\-.]+){0,2}")


def _date_view(current):
    return (
        f"{current.year}年{current.month}月{current.day}日"
        f"（{_WEEKDAYS[current.weekday()]}）"
    )


def jst_now_view(now=None):
    """The current moment as the model should see it: Japan time with
    the weekday spelled out, so 'today' never drifts to a UTC date or
    a guessed weekday."""
    current = (now or datetime.now(JST)).astimezone(JST)
    return {
        "timezone": "Asia/Tokyo",
        "iso": current.isoformat(timespec="seconds"),
        "date": _date_view(current),
        "time": current.strftime("%H:%M"),
    }


def current_time_view(name="Asia/Tokyo"):
    if (
        not isinstance(name, str)
        or not _TIMEZONE_PATTERN.fullmatch(name)
        or ".." in name
    ):
        raise WebError("Invalid timezone")
    try:
        zone = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise WebError(f"Unknown timezone: {name}") from error
    current = datetime.now(zone)
    return {
        "timezone": name,
        "iso": current.isoformat(timespec="seconds"),
        "date": _date_view(current),
        "time": current.strftime("%H:%M"),
        "utc_offset": current.strftime("%z"),
    }


# WMO weather interpretation codes, as used by Open-Meteo.
WMO_LABELS = {
    0: "快晴",
    1: "晴れ",
    2: "晴れ時々曇り",
    3: "曇り",
    45: "霧",
    48: "霧",
    51: "弱い霧雨",
    53: "霧雨",
    55: "霧雨",
    56: "着氷性の霧雨",
    57: "着氷性の霧雨",
    61: "弱い雨",
    63: "雨",
    65: "強い雨",
    66: "着氷性の雨",
    67: "着氷性の雨",
    71: "弱い雪",
    73: "雪",
    75: "強い雪",
    77: "雪",
    80: "弱いにわか雨",
    81: "にわか雨",
    82: "強いにわか雨",
    85: "にわか雪",
    86: "にわか雪",
    95: "雷雨",
    96: "雹を伴う雷雨",
    99: "雹を伴う雷雨",
}


def weather_label(code):
    if type(code) is int and code in WMO_LABELS:
        return WMO_LABELS[code]
    return f"不明（コード{code}）"


# GeoNames feature codes: capitals and administrative seats outrank
# plain populated places when population alone cannot separate them.
_FEATURE_RANK = {"PPLC": 0, "PPLA": 1, "PPLA2": 2, "PPLA3": 3, "PPLA4": 4, "PPLA5": 5}


def _is_place(item):
    code = str(item.get("feature_code") or "")
    if not code:
        return True
    return code.startswith("PPL") or code.startswith("ADM") or code in _FEATURE_RANK


def _pick(candidates, place):
    if not candidates:
        return None

    def rank(item):
        code = str(item.get("feature_code") or "")
        name = str(item.get("name", ""))
        exact = (
            0 if name in (place, place + "市", place + "都")
            else 1 if name.startswith(place) else 2
        )
        return (
            -(item.get("population") or 0),
            _FEATURE_RANK.get(code, 6 if code.startswith("PPL") else 9),
            exact,
        )

    return min(candidates, key=rank)


def _is_weak(item):
    """A pick that is probably a minor locality: no population on
    record and not a capital or administrative seat."""
    code = str(item.get("feature_code") or "")
    return not item.get("population") and code not in _FEATURE_RANK


def _needs_suffix_variants(place):
    if place.endswith(("市", "区", "町", "村", "都", "道", "府", "県")):
        return False
    return any(
        "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in place
    )


def _tool(name, description, required, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


WEATHER_TOOLS = [
    _tool(
        "get_weather",
        "Get the current weather for a place from Open-Meteo: the current "
        "temperature, condition, humidity and wind observed in the place's "
        "own local time, plus today's high and low. Use this for weather "
        "or temperature questions instead of web search snippets.",
        ["place"],
        {"place": {"type": "string", "minLength": 1, "maxLength": 100}},
    ),
    _tool(
        "current_time",
        "Get the current date and time in an IANA timezone (for example "
        "America/New_York or Europe/London). Defaults to Asia/Tokyo.",
        [],
        {"timezone": {"type": "string", "minLength": 1, "maxLength": 64}},
    ),
]
WEATHER_PARAMETERS = {
    tool["function"]["name"]: tool["function"]["parameters"] for tool in WEATHER_TOOLS
}


class WeatherTools:
    def __init__(self, web=None):
        self.web = web if web is not None else WebTools()
        self._geocode_cache = {}

    def call(self, name, args):
        schema = WEATHER_PARAMETERS.get(name)
        if (
            schema is None
            or not isinstance(args, dict)
            or set(args) - schema["properties"].keys()
            or not set(schema["required"]).issubset(args)
        ):
            raise ValueError("Invalid weather tool arguments")
        for key, value in args.items():
            spec = schema["properties"][key]
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > spec["maxLength"]
            ):
                raise ValueError("Invalid weather tool arguments")
        try:
            result = self.web.run(self.execute(name, args))
            return json.dumps(result, ensure_ascii=False)
        except (WebError, json.JSONDecodeError) as error:
            return f"Web tool error: {error}"
        except TimeoutError:
            return "Web tool error: timed out"

    async def execute(self, name, args):
        if name == "current_time":
            return current_time_view(args.get("timezone", "Asia/Tokyo"))
        return await self._weather(args["place"].strip())

    async def _search_places(self, name):
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {"name": name, "count": 10, "language": "ja", "format": "json"}
        )
        _, _, text = await self.web.fetch(url)
        data = json.loads(text)
        results = data.get("results") if isinstance(data, dict) else None
        return [
            item for item in results or []
            if isinstance(item, dict)
            and isinstance(item.get("latitude"), (int, float))
            and isinstance(item.get("longitude"), (int, float))
            and _is_place(item)
        ]

    async def _geocode(self, place):
        key = place.lower()
        if key in self._geocode_cache:
            return self._geocode_cache[key]
        candidates = await self._search_places(place)
        best = _pick(candidates, place)
        if (best is None or _is_weak(best)) and _needs_suffix_variants(place):
            # Japanese city names often resolve only with their
            # administrative suffix: 福岡 alone returns eight hamlets
            # (福岡市 is never among them), 東京/大阪/京都 return
            # nothing at all, while 福岡市/東京都 hit the real city.
            for suffix in ("市", "都"):
                candidates += await self._search_places(place + suffix)
            best = _pick(candidates, place)
        if best is None:
            raise WebError(f"Place not found: {place}")
        label = ", ".join(
            part for part in (
                best.get("name"), best.get("admin1"), best.get("country"),
            ) if part
        )
        entry = {
            "latitude": best["latitude"],
            "longitude": best["longitude"],
            "label": label,
        }
        if len(self._geocode_cache) > 256:
            self._geocode_cache.clear()
        self._geocode_cache[key] = entry
        return entry

    async def _weather(self, place):
        geo = await self._geocode(place)
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 1,
        })
        _, _, text = await self.web.fetch(url)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise WebError("Invalid weather response")
        current = data.get("current") or {}
        daily = data.get("daily") or {}
        if not isinstance(current.get("temperature_2m"), (int, float)):
            raise WebError("Invalid weather response")

        def first(values):
            return values[0] if isinstance(values, list) and values else None

        return {
            "place": geo["label"],
            "timezone": data.get("timezone"),
            # Open-Meteo's current.time is the observation time in the
            # place's local time (ISO, no offset suffix): report it as
            # such so the model never has to convert a Z timestamp.
            "observed_local": current.get("time"),
            "current": {
                "temperature_c": current["temperature_2m"],
                "condition": weather_label(current.get("weather_code")),
                "humidity_pct": current.get("relative_humidity_2m"),
                "wind_kmh": current.get("wind_speed_10m"),
            },
            "today": {
                "high_c": first(daily.get("temperature_2m_max")),
                "low_c": first(daily.get("temperature_2m_min")),
                "condition": weather_label(first(daily.get("weather_code"))),
            },
        }
