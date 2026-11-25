import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import pytest
from test_assistant import client_for, message, tool
from test_codex import FakeProcess, conversation, finished
from test_codex import tool as codex_tool

from nezubot_discrod.assistant import Assistant, instant_lookup_requested
from nezubot_discrod.codex import AppServer
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.prompts import SYSTEM, WEATHER_GUIDANCE, conversation_prompt
from nezubot_discrod.server_log_tools import LogResult
from nezubot_discrod.weather_tools import (
    WEATHER_TOOLS,
    WeatherTools,
    current_time_view,
    jst_now_view,
    weather_label,
)
from nezubot_discrod.web_tools import WebError

GEOCODE_FUKUOKA = {
    "results": [
        # A small locality named 福岡 comes first in the raw response;
        # the city must win on population among exact name matches.
        {"name": "福岡", "latitude": 35.87773, "longitude": 139.54678,
         "population": 1200, "admin1": "埼玉県", "country": "日本"},
        {"name": "福岡", "latitude": 33.59018, "longitude": 130.40172,
         "population": 1600000, "admin1": "福岡県", "country": "日本"},
        {"name": "福岡市", "latitude": 33.59018, "longitude": 130.40172,
         "population": 1600000, "admin1": "福岡県", "country": "日本"},
    ]
}

FORECAST_FUKUOKA = {
    "timezone": "Asia/Tokyo",
    "current": {
        "time": "2026-11-25T09:45", "temperature_2m": 10.2,
        "relative_humidity_2m": 55, "weather_code": 1, "wind_speed_10m": 8.3,
    },
    "daily": {
        "weather_code": [2], "temperature_2m_max": [14.0],
        "temperature_2m_min": [6.5],
    },
}


class StubWeb:
    """WebTools-shaped stub: fetch returns canned JSON by URL marker
    (matched against the percent-decoded URL)."""

    def __init__(self, payloads):
        self.payloads = payloads
        self.fetched = []

    def run(self, coro):
        return asyncio.run(coro)

    async def fetch(self, url, **kwargs):
        self.fetched.append(url)
        decoded = unquote(url)
        for marker, payload in self.payloads:
            if marker in decoded:
                return (url, "application/json", json.dumps(payload))
        raise AssertionError(f"unexpected fetch: {url}")


def weather_for(payloads):
    return WeatherTools(StubWeb(payloads))


def test_jst_now_view_shape():
    view = jst_now_view(datetime(2026, 11, 25, 9, 52, 3, tzinfo=ZoneInfo("UTC")))
    assert view["timezone"] == "Asia/Tokyo"
    assert view["iso"] == "2026-11-25T18:52:03+09:00"
    assert view["date"] == "2026年11月25日（水）"
    assert view["time"] == "18:52"


def test_current_time_defaults_to_japan_and_converts():
    japan = current_time_view()
    assert japan["timezone"] == "Asia/Tokyo"
    assert japan["utc_offset"] == "+0900"
    new_york = current_time_view("America/New_York")
    assert new_york["timezone"] == "America/New_York"
    assert new_york["utc_offset"] in {"-0400", "-0500"}
    assert new_york["iso"].endswith(new_york["utc_offset"][:3] + ":" + new_york["utc_offset"][3:])


@pytest.mark.parametrize("name", ["", "../etc/passwd", "Not/AZone", "Asia Tokyo", "/UTC"])
def test_current_time_rejects_bad_timezones(name):
    with pytest.raises(WebError):
        current_time_view(name)


def test_weather_labels():
    assert weather_label(0) == "快晴"
    assert weather_label(63) == "雨"
    assert weather_label(95) == "雷雨"
    assert "コード" in weather_label(12345)
    assert "コード" in weather_label(None)


def test_get_weather_picks_the_city_and_reports_local_time():
    tools = weather_for([
        ("geocoding-api", GEOCODE_FUKUOKA), ("api.open-meteo", FORECAST_FUKUOKA),
    ])
    result = json.loads(tools.call("get_weather", {"place": "福岡"}))
    assert result["place"] == "福岡, 福岡県, 日本"
    assert result["timezone"] == "Asia/Tokyo"
    assert result["observed_local"] == "2026-11-25T09:45"
    assert result["current"]["temperature_c"] == 10.2
    assert result["current"]["condition"] == "晴れ"
    assert result["today"] == {"high_c": 14.0, "low_c": 6.5, "condition": "晴れ時々曇り"}
    forecast_url = tools.web.fetched[-1]
    assert "latitude=33.59018" in forecast_url
    assert "timezone=auto" in forecast_url


# The real Open-Meteo responses for a bare 福岡: eight hamlets, the
# city never appears; only the 福岡市 query returns it.
GEOCODE_FUKUOKA_BARE = {
    "results": [
        {"name": "福岡", "latitude": 35.87773, "longitude": 139.54678,
         "admin1": "埼玉県", "country": "日本", "feature_code": "PPL"},
        {"name": "福岡", "latitude": 40.27857, "longitude": 141.32318,
         "admin1": "岩手県", "country": "日本", "feature_code": "PPL"},
        {"name": "福岡", "latitude": 32.61923, "longitude": 130.75891,
         "admin1": "熊本県", "country": "日本", "feature_code": "PPLL"},
    ]
}
GEOCODE_FUKUOKA_CITY = {
    "results": [
        {"name": "福岡市", "latitude": 33.6, "longitude": 130.41667,
         "population": 1612392, "admin1": "福岡県", "country": "日本",
         "feature_code": "PPLA"},
    ]
}


def test_get_weather_falls_back_to_city_suffix_for_japanese_names():
    tools = weather_for([
        ("name=福岡&", GEOCODE_FUKUOKA_BARE),
        ("name=福岡市&", GEOCODE_FUKUOKA_CITY),
        ("name=福岡都&", {"results": []}),
        ("api.open-meteo", FORECAST_FUKUOKA),
    ])
    result = json.loads(tools.call("get_weather", {"place": "福岡"}))
    assert result["place"] == "福岡市, 福岡県, 日本"
    assert "latitude=33.6" in tools.web.fetched[-1]


def test_get_weather_unknown_place_is_a_tool_error():
    tools = weather_for([("geocoding-api", {"results": []})])
    assert "Web tool error" in tools.call("get_weather", {"place": "存在しない場所"})


def test_geocode_is_cached():
    tools = weather_for([
        ("geocoding-api", GEOCODE_FUKUOKA), ("api.open-meteo", FORECAST_FUKUOKA),
    ])
    tools.call("get_weather", {"place": "福岡"})
    tools.call("get_weather", {"place": "福岡"})
    geocodes = [url for url in tools.web.fetched if "geocoding-api" in url]
    assert len(geocodes) == 1


def test_current_time_tool_via_call():
    tools = WeatherTools(StubWeb([]))
    result = json.loads(tools.call("current_time", {"timezone": "Europe/London"}))
    assert result["timezone"] == "Europe/London"
    assert "Web tool error" in tools.call("current_time", {"timezone": "Nowhere/Fake"})


@pytest.mark.parametrize("name,args", [
    ("get_weather", {}),
    ("get_weather", {"place": ""}),
    ("get_weather", {"place": "x" * 101}),
    ("get_weather", {"place": "福岡", "extra": 1}),
    ("current_time", {"timezone": 5}),
    ("unknown_tool", {}),
])
def test_weather_schema_validation(name, args):
    with pytest.raises(ValueError):
        WeatherTools(StubWeb([])).call(name, args)


@pytest.mark.parametrize("content", [
    "福岡の気温おしえて", "今日の天気は？", "明日の降水確率", "今何時？",
    "今日は何日だっけ", "湿度が高いね",
])
def test_instant_lookup_intent(content):
    assert instant_lookup_requested(content)


@pytest.mark.parametrize("content", [
    "俺について簡単にまとめて", "昨日のログ見せて", "SQLiteの使い方", "",
])
def test_non_instant_lookup_intent(content):
    assert not instant_lookup_requested(content)


def test_instant_lookup_skips_pre_retrieval(monkeypatch):
    retrieve = AsyncMock(return_value=LogResult(""))
    monkeypatch.setattr("nezubot_discrod.assistant.pre_retrieve", retrieve)
    store = Mock()
    store.recent.return_value = []
    completion = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
    ))
    assistant = Assistant(
        Mock(), "test", SandboxSettings(), server_log=store,
        wiki_completion=completion, web=Mock(), weather=Mock(),
    )
    answer = asyncio.run(assistant.reply(
        10, "福岡の気温おしえて", member_id=20, guild_id=100, channel_ids=[10],
    ))
    assert answer == "done"
    retrieve.assert_not_awaited()
    sent = completion.call_args.kwargs
    assert sent["weather"] is assistant.weather
    payload = json.loads(sent["messages"][-1]["content"])
    assert payload["now"]["timezone"] == "Asia/Tokyo"
    assert WEATHER_GUIDANCE in sent["messages"][0]["content"]


def test_weather_guidance_only_with_weather_tools():
    assert WEATHER_GUIDANCE not in conversation_prompt(web=True)
    assert WEATHER_GUIDANCE in conversation_prompt(web=True, weather=True)


def test_system_prompt_defaults_to_japan_time():
    assert "now field" in SYSTEM
    assert "Japan time by default" in SYSTEM
    assert "convert them to the relevant local time" in SYSTEM


def test_legacy_provider_dispatches_weather_tool():
    weather = Mock()
    weather.call.return_value = '{"current":{"temperature_c":10.2}}'
    client, sent = client_for(
        message(calls=[tool('{"place":"福岡"}', "get_weather")]), message("10度だよ"),
    )
    ai = Assistant(client, "test", SandboxSettings(), web=Mock(), weather=weather)
    assert asyncio.run(ai.reply(1, "福岡の気温は？")) == "10度だよ"
    weather.call.assert_called_once_with("get_weather", {"place": "福岡"})
    assert "get_weather" in [t["function"]["name"] for t in sent[0]["tools"]]


def test_codex_turn_registers_and_runs_weather_tools():
    web = Mock()
    web.call.return_value = '{"results":[{"title":"news"}]}'
    weather = Mock()
    weather.call.return_value = '{"current":{"temperature_c":10.2}}'
    process = FakeProcess(conversation(
        codex_tool({"query": "福岡 ニュース"}, "web_search"),
        codex_tool({"place": "福岡"}, "get_weather"),
        codex_tool({"timezone": "America/New_York"}, "current_time"),
        *finished(),
    ))
    # private_read=True: even in a private turn, after web content was
    # read, the weather/clock tools stay available (they are not web
    # content and cannot fetch attacker-chosen URLs).
    asyncio.run(AppServer(process, SandboxSettings()).complete(
        "test", [], "/empty", web=web, weather=weather, private_read=True,
    ))
    web.call.assert_called_once_with("web_search", {"query": "福岡 ニュース"})
    assert weather.call.call_count == 2
    results = [p["result"] for p in process.sent if p.get("id") == 99]
    assert all(result["success"] for result in results)
    start = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
    assert {t["function"]["name"] for t in WEATHER_TOOLS} <= {
        t["name"] for t in start["dynamicTools"]
    }
    assert start["baseInstructions"] == conversation_prompt(web=True, weather=True)
