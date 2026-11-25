import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nezubot_discrod.web_browser import BrowserSearch, _EgressRelay, parse_bing
from nezubot_discrod.web_tools import BROWSER_RESULT_NOTE, WebError, WebTools

PROXY_ENV_VARS = (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch):
    # Same reasoning as test_web_tools: engine vetting consults the
    # proxy environment, and the tests must observe the no-proxy path.
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _ck_url(real_url):
    encoded = base64.urlsafe_b64encode(real_url.encode()).decode().rstrip("=")
    return f"https://www.bing.com/ck/a?!&&p=xyz&u=a1{encoded}&ntb=1"


BING_HTML = (
    '<html><body><ol id="b_results">'
    '<li class="b_algo">'
    f'<h2><a href="{_ck_url("https://example.org/story")}">Bing result title</a></h2>'
    '<div class="b_caption"><p class="b_lineclamp2">Bing snippet text.</p></div>'
    "</li>"
    '<li class="b_algo">'
    '<h2><a href="https://direct.example.net/page">Direct result</a></h2>'
    '<div class="b_caption"><p>Direct snippet.</p></div>'
    "</li>"
    '<li class="b_algo">'
    '<h2><a href="http://192.168.0.1/admin">Private result</a></h2>'
    '<div class="b_caption"><p>Must be dropped by vetting.</p></div>'
    "</li>"
    "</ol></body></html>"
)

BING_NEWS_HTML = (
    '<html><body><ol id="b_results">'
    '<li class="b_algo">'
    '<span class="news_dt">2026/11/25</span>'
    '<h2><a href="https://news.example.org/a">News title</a></h2>'
    '<div class="b_caption"><p>News snippet.</p></div>'
    "</li>"
    "</ol></body></html>"
)


def test_parse_bing_unwraps_redirects_and_vets_results():
    results = parse_bing(BING_HTML, 5)
    assert results == [
        {
            "title": "Bing result title",
            "url": "https://example.org/story",
            "snippet": "Bing snippet text.",
        },
        {
            "title": "Direct result",
            "url": "https://direct.example.net/page",
            "snippet": "Direct snippet.",
        },
    ]
    assert len(parse_bing(BING_HTML, 1)) == 1
    assert parse_bing("<html><body>no results</body></html>", 5) == []


def test_parse_bing_attaches_news_date():
    results = parse_bing(BING_NEWS_HTML, 5)
    assert results[0]["url"] == "https://news.example.org/a"
    assert results[0]["date"] == "2026/11/25"


async def _relay_request(relay, request):
    reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
    writer.write(request)
    await writer.drain()
    try:
        response = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        return response.decode("iso-8859-1").split("\r\n")[0]
    finally:
        writer.close()


def test_relay_vets_connect_targets():
    async def run():
        # Upstream is unreachable on purpose: vetted-out targets must
        # be refused before any upstream connection is attempted.
        relay = _EgressRelay(upstream="http://127.0.0.1:1")
        assert await relay.start() is not None
        try:
            assert await _relay_request(
                relay, b"CONNECT 192.168.0.1:443 HTTP/1.1\r\nHost: x\r\n\r\n"
            ) == "HTTP/1.1 403 Forbidden"
            assert await _relay_request(
                relay, b"CONNECT example.org:80 HTTP/1.1\r\nHost: x\r\n\r\n"
            ) == "HTTP/1.1 403 Forbidden"
            assert await _relay_request(
                relay, b"POST http://example.org/ HTTP/1.1\r\nHost: x\r\n\r\n"
            ) == "HTTP/1.1 405 Method Not Allowed"
        finally:
            await relay.stop()

    asyncio.run(run())


def test_relay_reports_unreachable_upstream_as_502():
    async def run():
        relay = _EgressRelay(upstream="http://127.0.0.1:1")
        assert await relay.start() is not None
        try:
            assert await _relay_request(
                relay, b"CONNECT example.org:443 HTTP/1.1\r\nHost: x\r\n\r\n"
            ) == "HTTP/1.1 502 Bad Gateway"
        finally:
            await relay.stop()

    asyncio.run(run())


def test_relay_stays_down_without_a_proxy():
    # The autouse fixture cleared the proxy environment, so no
    # upstream is configured and the relay must not start.
    async def run():
        relay = _EgressRelay()
        assert await relay.start() is None
        assert relay.port is None

    asyncio.run(run())


DDG_HTML = (
    '<html><body><a class="result__a" href="https://example.org/page">'
    "Example result</a>"
    '<a class="result__snippet">A snippet</a></body></html>'
)
EMPTY_PAGE = "<html><body>blocked, no results</body></html>"


def _tier_with_render(render):
    tier = BrowserSearch()
    tier._render = render
    return tier


def test_browser_search_falls_back_across_engines():
    seen = []

    async def render(url):
        seen.append(url)
        if "lite.duckduckgo" in url:
            return EMPTY_PAGE
        if "html.duckduckgo" in url:
            return DDG_HTML
        return None

    hits = asyncio.run(_tier_with_render(render).search("example"))
    assert hits[0]["title"] == "Example result"
    assert len(seen) == 2  # lite first, html second, bing never reached


def test_browser_search_reaches_bing_as_last_engine():
    async def render(url):
        if "bing.com" in url:
            return BING_HTML
        return EMPTY_PAGE

    hits = asyncio.run(_tier_with_render(render).search("example"))
    assert hits[0]["url"] == "https://example.org/story"


def test_browser_search_returns_empty_when_all_engines_fail():
    async def render(url):
        return None

    assert asyncio.run(_tier_with_render(render).search("example")) == []


def test_browser_search_never_raises():
    tier = BrowserSearch()
    with patch.object(tier, "_search", AsyncMock(side_effect=RuntimeError("boom"))):
        assert asyncio.run(tier.search("example")) == []


def _browser_fake(hits):
    return SimpleNamespace(search=AsyncMock(return_value=hits))


BROWSER_HIT = {
    "title": "Browser result",
    "url": "https://example.org/browser",
    "snippet": "Found by the browser.",
}


def test_escalation_when_http_returns_nothing():
    web = WebTools(browser_search=_browser_fake([BROWSER_HIT]))

    async def empty(url, **kwargs):
        return (url, "text/html", EMPTY_PAGE)

    with patch("nezubot_discrod.web_tools.fetch_text", side_effect=empty):
        result = json.loads(web.call("web_search", {"query": "example"}))
    assert result["results"] == [BROWSER_HIT]
    assert result["note"] == BROWSER_RESULT_NOTE
    web._browser_search.search.assert_awaited_once_with("example", 5, 1)


def test_escalation_rescues_a_blocked_http_search():
    web = WebTools(browser_search=_browser_fake([BROWSER_HIT]))
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(side_effect=WebError("HTTP 403")),
    ):
        result = json.loads(web.call("web_search", {"query": "example"}))
    assert result["results"] == [BROWSER_HIT]


def test_http_error_surfaces_when_browser_also_fails():
    web = WebTools(browser_search=_browser_fake([]))
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(side_effect=WebError("HTTP 403")),
    ):
        assert "Web tool error: HTTP 403" in web.call("web_search", {"query": "example"})


def test_no_escalation_when_http_answers():
    web = WebTools(browser_search=_browser_fake([BROWSER_HIT]))
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(return_value=("https://example.org", "text/html", DDG_HTML)),
    ):
        result = json.loads(web.call("web_search", {"query": "example"}))
    assert result["results"][0]["title"] == "Example result"
    web._browser_search.search.assert_not_awaited()


def test_browser_tier_kill_switch(monkeypatch):
    monkeypatch.delenv("WEB_TOOLS_ENABLED", raising=False)
    monkeypatch.delenv("WEB_SEARCH_BROWSER", raising=False)
    web = WebTools.from_env()
    assert web is not None and web._browser_search is not None
    monkeypatch.setenv("WEB_SEARCH_BROWSER", "off")
    web = WebTools.from_env()
    assert web is not None and web._browser_search is None
    # Plain construction never wires the tier: unit tests and helper
    # contexts stay HTTP-only and hermetic.
    assert WebTools()._browser_search is None
