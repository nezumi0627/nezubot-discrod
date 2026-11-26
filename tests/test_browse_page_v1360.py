"""v1.36.0: browse_page — the read-only browser tool.

Pins: forbidden targets are refused before any browser launches;
every request a browsed page makes passes the SSRF route guard; the
tool is registered beside the other web tools and fails soft when
the browser tier is absent; the outbound guard vets its URL like
read_web_page's; WEB_GUIDANCE routes JS pages to it while keeping
GitHub metadata authoritative.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nezubot_discrod.prompts import WEB_GUIDANCE
from nezubot_discrod.web_browser import (
    BrowserSearch,
    _guard_browse_route,
    _guard_browse_socket,
)
from nezubot_discrod.web_guard import OutboundGuard
from nezubot_discrod.web_tools import WEB_PARAMETERS, WebError, WebTools, page_links

PROXY_ENV_VARS = (
    "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY",
)


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch):
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8799/health",       # the devin bridge
    "http://127.0.0.1:8798/",             # the ws relay's neighborhood
    "http://localhost:8799/",
    "http://[::1]/",
    "https://192.168.1.1/admin",
    "http://10.0.0.5/",
    "http://169.254.169.254/latest/meta-data/",
    "http://172.16.0.9/",
    "file:///etc/passwd",
    "https://user:pass@example.com/",
])
def test_browse_refuses_internal_targets_before_launching(url, monkeypatch):
    tier = BrowserSearch()

    async def forbidden_launch():
        raise AssertionError("browser must not launch for a refused URL")

    monkeypatch.setattr(tier, "_ensure_browser", forbidden_launch)
    with pytest.raises(WebError):
        asyncio.run(tier.browse(url))


class FakeRoute:
    def __init__(self, url, resource_type="document"):
        self.request = SimpleNamespace(url=url, resource_type=resource_type)
        self.action = None

    async def abort(self):
        self.action = "abort"

    async def continue_(self):
        self.action = "continue"


@pytest.mark.parametrize("url,expected", [
    ("https://example.com/page", "continue"),
    ("https://sub.example.co.jp/a?b=c", "continue"),
    ("http://127.0.0.1:8799/", "abort"),
    ("http://localhost/", "abort"),
    ("https://192.168.0.1/", "abort"),
    ("http://169.254.169.254/", "abort"),
    ("file:///etc/passwd", "abort"),
    ("chrome-extension://abc/page.html", "abort"),
    ("data:text/html,<p>hi</p>", "continue"),
    ("blob:https://example.com/9f2c", "continue"),
])
def test_browse_route_guard_vets_every_request(url, expected):
    route = FakeRoute(url)
    asyncio.run(_guard_browse_route(route))
    assert route.action == expected


def test_browse_route_guard_blocks_heavy_resources():
    route = FakeRoute("https://example.com/big.png", resource_type="image")
    asyncio.run(_guard_browse_route(route))
    assert route.action == "abort"


class FakeSocketRoute:
    def __init__(self, url):
        self.url = url
        self.action = None

    async def close(self):
        self.action = "close"

    async def connect_to_server(self):
        self.action = "connect"


@pytest.mark.parametrize("url,expected", [
    ("wss://example.com/socket", "connect"),
    ("ws://127.0.0.1:8799/socket", "close"),
    ("wss://10.1.2.3/socket", "close"),
])
def test_browse_socket_guard_vets_websockets(url, expected):
    route = FakeSocketRoute(url)
    asyncio.run(_guard_browse_socket(route))
    assert route.action == expected


def test_browse_renders_through_a_fresh_context(monkeypatch):
    tier = BrowserSearch()
    contexts = []

    class FakePage:
        url = "https://example.com/final"

        async def goto(self, url, **kwargs):
            self.went = url

        async def wait_for_timeout(self, ms):
            pass

        async def title(self):
            return "Rendered title"

        async def content(self):
            return "<html><body><p>rendered</p></body></html>"

    class FakeContext:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            contexts.append(self)

        async def route(self, pattern, handler):
            pass

        async def route_web_socket(self, pattern, handler):
            pass

        async def new_page(self):
            return FakePage()

        async def close(self):
            self.closed = True

    class FakeBrowser:
        def is_connected(self):
            return True

        async def new_context(self, **kwargs):
            return FakeContext(**kwargs)

    async def fake_ensure():
        return FakeBrowser()

    monkeypatch.setattr(tier, "_ensure_browser", fake_ensure)
    result = asyncio.run(tier.browse("https://example.com/start"))
    assert result["url"] == "https://example.com/final"
    assert result["title"] == "Rendered title"
    assert "rendered" in result["html"]
    # Read-only, stateless: downloads off, and the context was closed.
    assert contexts[0].kwargs["accept_downloads"] is False
    assert contexts[0].closed is True


RENDERED_HTML = (
    "<html><head><title>App</title></head><body>"
    "<p>Rendered body text</p>"
    '<a href="/next">Next page</a>'
    '<a href="https://other.example.org/x">Other</a>'
    '<a href="http://127.0.0.1:8799/">internal</a>'
    "</body></html>"
)


class FakeBrowserTier:
    async def browse(self, url):
        return {"url": url, "title": "App", "html": RENDERED_HTML}


def test_browse_page_tool_returns_text_and_vetted_links():
    tools = WebTools(browser_search=FakeBrowserTier())
    import json

    result = json.loads(tools.call("browse_page", {"url": "https://example.com/app"}))
    assert result["text"].startswith("Rendered body text")
    assert result["title"] == "App"
    urls = [link["url"] for link in result["links"]]
    assert urls == ["https://example.com/next", "https://other.example.org/x"]


def test_browse_page_tool_without_browser_tier_fails_soft():
    tools = WebTools()
    result = tools.call("browse_page", {"url": "https://example.com/"})
    assert result.startswith("Web tool error:")


def test_browse_page_is_registered():
    assert "browse_page" in WEB_PARAMETERS
    assert WEB_PARAMETERS["browse_page"]["required"] == ["url"]


def test_page_links_skips_duplicates_and_javascript():
    html = (
        '<a href="/a">A</a><a href="/a">A again</a>'
        '<a href="javascript:void(0)">js</a><a href="#frag">frag</a>'
    )
    links = page_links(html, "https://example.com/base")
    assert [link["url"] for link in links] == ["https://example.com/a"]


def test_outbound_guard_vets_browse_page_urls():
    guard = OutboundGuard()
    guard.note_private(
        "2026-11-24 21:03 タヌキ: 来週の月曜日にサーバーの移行作業を行う予定で、"
        "障害が出たら即座にロールバックすることになりました。"
    )
    fragment = (
        "来週の月曜日にサーバーの移行作業を行う予定で、"
        "障害が出たら即座にロールバックすることになりました"
    )
    assert guard.check_call("browse_page", {"url": "https://x.example/?q=" + fragment})
    assert guard.check_call("browse_page", {"url": "https://example.org/a"}) is None


def test_web_guidance_routes_js_pages_to_browse_page():
    assert "browse_page" in WEB_GUIDANCE
    assert "GitHub情報（確定情報）" in WEB_GUIDANCE
