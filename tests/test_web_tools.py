import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from test_assistant import client_for, message, tool
from test_codex import FakeProcess, conversation, finished
from test_codex import tool as codex_tool

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import AppServer, CodexSecurityError
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.prompts import SELF_MOD_PROMPT, WEB_GUIDANCE, conversation_prompt
from nezubot_discrod.web_tools import (
    MAX_BYTES,
    SEARCH_ATTEMPT_TIMEOUT,
    SEARCH_TIMEOUT,
    WEB_TOOLS,
    PublicResolver,
    WebError,
    WebTools,
    _fetch_redirects,
    fetch_text,
    parse_ddg,
    proxy_for_url,
    public_ip,
    readable,
    vet_proxy_url,
    vet_url,
)

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
    # The host running the tests may export egress proxy variables; the
    # direct-path tests must observe the no-proxy behavior regardless.
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.1.2",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "192.0.2.1",
        "100.64.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "fe80::1",
        "fc00::1",
        "fec0::1",
        "ff02::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "::ffff:8.8.8.8",
        "fe80::1%eth0",
        "64:ff9b::a9fe:a9fe",
        "2002:7f00:1::",
        "not-an-ip",
    ],
)
def test_reject_non_public_ips(ip):
    with pytest.raises(WebError):
        public_ip(ip)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.org",
        "http://localhost:8080",
        "http://localhost./",
        "http://a.localhost/",
        "http://169.254.169.254/",
        "http://[::ffff:127.0.0.1]/",
        "http://127.1/",
        "http://2130706433/",
        "http://0177.0.0.1/",
        "http://１２７。０。０。１/",
        "http://user:password@example.org/",
        "http://example.org:0",
        "http://example.org:bad",
        "http://example.org/\r\nHost:evil",
        "https://",
        "https://[::1",
        "//example.org/path",
    ],
)
def test_url_vetting(url):
    with pytest.raises(WebError):
        vet_url(url)


def test_public_addresses_and_urls():
    assert public_ip("8.8.8.8") == "8.8.8.8"
    assert public_ip("2606:4700:4700::1111") == "2606:4700:4700::1111"
    assert vet_url("https://example.org/path?q=1") == "https://example.org/path?q=1"


def test_resolver_pins_public_endpoints_and_rejects_mixed_dns():
    async def run():
        loop = asyncio.get_running_loop()
        public = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))
        private = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))
        resolver = PublicResolver()
        with patch.object(loop, "getaddrinfo", AsyncMock(side_effect=[[public], [private]])) as dns:
            connector = aiohttp.TCPConnector(
                resolver=resolver, use_dns_cache=False, force_close=True
            )
            try:
                hosts = await connector._resolve_host("example.org", 443)
                assert [host["host"] for host in hosts] == ["8.8.8.8"]
                assert hosts[0]["flags"] == socket.AI_NUMERICHOST
                assert hosts[0]["hostname"] == "example.org"
                assert dns.await_count == 1
                with pytest.raises(WebError):
                    await connector._resolve_host("example.org", 443)
            finally:
                await connector.close()
        with patch.object(loop, "getaddrinfo", AsyncMock(return_value=[public, private])):
            with pytest.raises(WebError):
                await resolver.resolve("mixed.example", 443)
        await resolver.close()

    asyncio.run(run())


class Response:
    def __init__(self, text=b"hello", *, status=200, headers=None, length=None):
        self.status = status
        self.headers = {"Content-Type": "text/plain", **(headers or {})}
        self.content_length = length
        self.charset = "utf-8"
        self.body = text
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_chunked(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i : i + size]


def test_redirects_recheck_targets_and_limit_hops():
    for target in [
        "http://169.254.169.254/",
        "http://localhost:8080",
        "file:///etc/passwd",
        "http://[::1]/",
        "http://user:pass@example.com/",
    ]:
        session = SimpleNamespace(
            get=Mock(return_value=Response(status=302, headers={"Location": target}))
        )
        with pytest.raises(WebError):
            asyncio.run(_fetch_redirects(session, "https://example.org/"))
        assert session.get.call_count == 1
        session.get.assert_called_with("https://example.org/", allow_redirects=False)
    session = SimpleNamespace(
        get=Mock(
            side_effect=[
                Response(status=301, headers={"Location": "/next"}),
                Response(b"done"),
            ]
        )
    )
    assert asyncio.run(_fetch_redirects(session, "https://example.org/")) == (
        "https://example.org/next",
        "text/plain",
        "done",
    )
    loop = SimpleNamespace(get=Mock(return_value=Response(status=302, headers={"Location": "/"})))
    with pytest.raises(WebError, match="redirect"):
        asyncio.run(_fetch_redirects(loop, "https://example.org/"))
    assert loop.get.call_count == 6


def test_search_post_redirect_policy_and_vetting():
    for status, posts in [(302, 1), (303, 1), (307, 2), (308, 2)]:
        redirect = Response(status=status, headers={"Location": "/next"})
        session = SimpleNamespace(post=Mock(side_effect=[redirect, Response(b"done")]),
                                  get=Mock(return_value=Response(b"done")))
        asyncio.run(_fetch_redirects(session, "https://example.org/", data={"q": "python"},
                                     headers={"User-Agent": "Mozilla/5.0"}))
        assert session.post.call_count == posts
        assert session.get.call_count == (2 - posts)
    session = SimpleNamespace(post=Mock(return_value=Response(
        status=307, headers={"Location": "http://169.254.169.254/"})))
    with pytest.raises(WebError):
        asyncio.run(_fetch_redirects(session, "https://example.org/", data={"q": "python"}))
    assert session.post.call_count == 1


@pytest.mark.parametrize(
    "response,match",
    [
        (Response(length=MAX_BYTES + 1), "2 MiB"),
        (Response(b"a" * (MAX_BYTES + 1)), "2 MiB"),
        (Response(headers={"Content-Type": "application/octet-stream"}), "content type"),
        (Response(headers={"Content-Encoding": "gzip"}), "Compressed"),
        (Response(status=404), "HTTP 404"),
        (Response(status=302), "redirect"),
    ],
)
def test_size_type_and_status_caps(response, match):
    with pytest.raises(WebError, match=match):
        asyncio.run(
            _fetch_redirects(
                SimpleNamespace(get=Mock(return_value=response)), "https://example.org/"
            )
        )


def test_fetch_timeout_proxy_isolation_and_error_conversion():
    class Session:
        def __init__(self, **kwargs):
            self.connector = kwargs["connector"]
            assert isinstance(self.connector._resolver, PublicResolver)
            assert not kwargs["trust_env"] and not kwargs["auto_decompress"]
            assert kwargs["headers"]["Accept-Encoding"] == "identity"
            assert kwargs["headers"]["User-Agent"] == (
                "nezubot-discrod/1.0"
            )
            assert isinstance(kwargs["cookie_jar"], aiohttp.DummyCookieJar)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.connector.close()

        def get(self, *args, **kwargs):
            raise TimeoutError

    timeout = asyncio.timeout
    with (
        patch("nezubot_discrod.web_tools.aiohttp.ClientSession", Session),
        patch("nezubot_discrod.web_tools.asyncio.timeout", wraps=timeout) as budget,
    ):
        with pytest.raises(WebError, match="TimeoutError"):
            asyncio.run(fetch_text("https://example.org"))
    budget.assert_called_once_with(10)


def test_fetch_text_timeout_parameter():
    class Session:
        def __init__(self, **kwargs):
            self.connector = kwargs["connector"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.connector.close()

        def get(self, *args, **kwargs):
            return Response(b"ok")

    timeout = asyncio.timeout
    with (
        patch("nezubot_discrod.web_tools.aiohttp.ClientSession", Session),
        patch("nezubot_discrod.web_tools.asyncio.timeout", wraps=timeout) as budget,
    ):
        asyncio.run(fetch_text("https://example.org", timeout=25))
    budget.assert_called_once_with(25)


def test_search_uses_extended_timeout():
    web = WebTools()
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(return_value=("https://example.org", "text/html", "<html></html>")),
    ) as fetch:
        web.call("web_search", {"query": "example"})
    # The overall search budget stays SEARCH_TIMEOUT; the first attempt
    # only gets a slice of it so a stalled attempt can be retried.
    assert fetch.call_args.kwargs["timeout"] == SEARCH_ATTEMPT_TIMEOUT
    assert SEARCH_ATTEMPT_TIMEOUT < SEARCH_TIMEOUT == 30


DDG_HTML = (
    '<html><body><a class="result__a" href="https://example.org/page">'
    "Example result</a>"
    '<a class="result__snippet">A snippet</a></body></html>'
)


def test_search_retries_once_after_a_timeout():
    web = WebTools()
    attempts = []

    async def flaky(url, **kwargs):
        attempts.append(kwargs["timeout"])
        if len(attempts) == 1:
            raise WebError("Web fetch failed (TimeoutError)")
        return ("https://example.org", "text/html", DDG_HTML)

    with patch("nezubot_discrod.web_tools.fetch_text", side_effect=flaky):
        result = json.loads(web.call("web_search", {"query": "example"}))
    assert attempts == [SEARCH_ATTEMPT_TIMEOUT, SEARCH_TIMEOUT - SEARCH_ATTEMPT_TIMEOUT]
    assert result["results"][0]["title"] == "Example result"


def test_search_does_not_retry_other_errors():
    web = WebTools()
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(side_effect=WebError("HTTP 403")),
    ) as fetch:
        assert "Web tool error" in web.call("web_search", {"query": "example"})
    fetch.assert_called_once()


def test_search_cache_avoids_refetch():
    web = WebTools()
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(return_value=("https://example.org", "text/html", DDG_HTML)),
    ) as fetch:
        first = web.call("web_search", {"query": "example"})
        second = web.call("web_search", {"query": "example"})
        web.call("web_search", {"query": "other"})
    assert first == second
    assert fetch.call_count == 2


LITE_HTML = (
    "<html><body><table>"
    '<tr><td><a class="result-link" href="https://example.org/lite">Lite result</a></td></tr>'
    '<tr><td class="result-snippet">Lite snippet</td></tr>'
    "</table></body></html>"
)


def test_search_falls_back_to_lite_endpoint_when_html_is_empty():
    web = WebTools()

    async def by_url(url, **kwargs):
        if "lite.duckduckgo" in url:
            return (url, "text/html", LITE_HTML)
        return (url, "text/html", "<html><body>blocked</body></html>")

    with patch("nezubot_discrod.web_tools.fetch_text", side_effect=by_url):
        result = json.loads(web.call("web_search", {"query": "example"}))
    assert result["results"][0]["title"] == "Lite result"
    assert result["results"][0]["snippet"] == "Lite snippet"


def test_pooled_session_is_reused_across_calls():
    web = WebTools()
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(return_value=("https://example.org", "text/html", "ok")),
    ) as fetch:
        web.call("read_web_page", {"url": "https://example.org/"})
        web.call("read_web_page", {"url": "https://example.org/other"})
    sessions = {id(call.kwargs["session"]) for call in fetch.call_args_list}
    assert len(sessions) == 1


def test_proxy_for_url_selection(monkeypatch):
    assert proxy_for_url("https://example.org/") is None
    assert proxy_for_url("http://example.org/") is None
    monkeypatch.setenv("https_proxy", "http://proxy.local:3128")
    assert proxy_for_url("https://example.org/") == "http://proxy.local:3128"
    assert proxy_for_url("http://example.org/") is None
    monkeypatch.setenv("ALL_PROXY", "http://all.local:8080")
    assert proxy_for_url("http://example.org/") == "http://all.local:8080"
    assert proxy_for_url("https://example.org/") == "http://proxy.local:3128"
    monkeypatch.delenv("https_proxy")
    monkeypatch.setenv("HTTPS_PROXY", "http://upper.local:3128")
    assert proxy_for_url("https://example.org/") == "http://upper.local:3128"
    monkeypatch.setenv("https_proxy", "socks5://proxy.local:1080")
    assert proxy_for_url("https://example.org/") == "http://upper.local:3128"
    monkeypatch.setenv("https_proxy", "not a url")
    assert proxy_for_url("https://example.org/") == "http://upper.local:3128"


def test_proxy_for_url_no_proxy_bypass(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.local:3128")
    monkeypatch.setenv("http_proxy", "http://proxy.local:3128")
    monkeypatch.setenv("no_proxy", "example.org, .internal.example, bypass.local:8443")
    assert proxy_for_url("https://example.org/") is None
    assert proxy_for_url("https://sub.example.org/") is None
    assert proxy_for_url("https://www.internal.example/") is None
    assert proxy_for_url("http://bypass.local/") is None
    assert proxy_for_url("https://other.example/") == "http://proxy.local:3128"
    monkeypatch.setenv("NO_PROXY", "*")
    assert proxy_for_url("https://other.example/") is None


@pytest.mark.parametrize(
    "url",
    [
        "http://intranet/",
        "http://intranet/wiki",
        "https://wiki.internal/",
        "http://printer.local/",
        "http://nas.lan/",
        "http://0x7f.0.0.1/",
    ],
)
def test_vet_proxy_url_rejects_internal_names(url):
    with pytest.raises(WebError):
        vet_proxy_url(url)


def test_vet_proxy_url_allows_public_targets():
    vet_proxy_url("https://example.org/path")
    vet_proxy_url("https://html.duckduckgo.com/html/")
    vet_proxy_url("https://[2606:4700:4700::1111]/")
    vet_proxy_url("https://1.2.3.example/")


def test_fetch_routes_through_env_proxy(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.local:3128")
    seen = {}

    class Session:
        def __init__(self, **kwargs):
            self.connector = kwargs["connector"]
            seen["resolver"] = self.connector._resolver
            assert not kwargs["trust_env"] and not kwargs["auto_decompress"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.connector.close()

        def get(self, url, **kwargs):
            seen["get"] = (url, kwargs)
            return Response(b"proxied")

    with patch("nezubot_discrod.web_tools.aiohttp.ClientSession", Session):
        assert asyncio.run(fetch_text("https://example.org/")) == (
            "https://example.org/",
            "text/plain",
            "proxied",
        )
    assert not isinstance(seen["resolver"], PublicResolver)
    assert seen["get"][1]["proxy"] == "http://proxy.local:3128"
    assert seen["get"][1]["allow_redirects"] is False


def test_proxy_redirects_stay_proxied_and_recheck_targets():
    proxy = "http://proxy.local:3128"
    for target in ["http://intranet/", "https://wiki.internal/", "http://169.254.169.254/"]:
        session = SimpleNamespace(
            get=Mock(return_value=Response(status=302, headers={"Location": target}))
        )
        with pytest.raises(WebError):
            asyncio.run(_fetch_redirects(session, "https://example.org/", proxy=proxy))
        assert session.get.call_count == 1
        session.get.assert_called_with(
            "https://example.org/", allow_redirects=False, proxy=proxy
        )
    session = SimpleNamespace(
        get=Mock(
            side_effect=[
                Response(status=301, headers={"Location": "/next"}),
                Response(b"done"),
            ]
        )
    )
    assert asyncio.run(_fetch_redirects(session, "https://example.org/", proxy=proxy)) == (
        "https://example.org/next",
        "text/plain",
        "done",
    )
    assert all(
        call.kwargs["proxy"] == proxy for call in session.get.call_args_list
    )


def test_web_guidance_requires_reporting_tool_errors_verbatim():
    assert "quote the error as given" in WEB_GUIDANCE
    assert "never invent a cause" in WEB_GUIDANCE


def test_ddg_and_html_fixtures():
    results = parse_ddg((FIXTURES / "ddg.html").read_text(), 8)
    assert results == [
        {
            "title": "Example guide",
            "url": "https://example.org/guide",
            "snippet": "A useful example & introduction.",
        },
        {"title": "Second", "url": "https://example.com/second", "snippet": "Another result."},
    ]
    assert len(parse_ddg((FIXTURES / "ddg.html").read_text(), 1)) == 1
    assert parse_ddg("<html>No results</html>", 8) == []
    assert (
        readable("<head>hidden</head><p>Visible &amp; useful</p><script>evil()</script>")
        == "Visible & useful"
    )


def test_web_and_wikipedia_with_fixtures():
    web = WebTools()
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(
            return_value=(
                "https://example.org",
                "text/html",
                (FIXTURES / "ddg.html").read_text(),
            )
        ),
    ) as fetch:
        assert (
            len(json.loads(web.call("web_search", {"query": "example", "limit": 1}))["results"])
            == 1
        )
        assert fetch.call_args.args[0] == "https://html.duckduckgo.com/html/"
        assert fetch.call_args.kwargs["data"] == {"q": "example"}
        assert fetch.call_args.kwargs["headers"]["User-Agent"] == "Mozilla/5.0"
    for name, args, fixture in [
        ("wikipedia_search", {"query": "Python"}, "wikipedia-search.json"),
        ("wikipedia_read", {"title": "Python", "lang": "en"}, "wikipedia-read.json"),
    ]:
        with patch(
            "nezubot_discrod.web_tools.fetch_text",
            AsyncMock(
                return_value=(
                    "https://wikipedia.org",
                    "application/json",
                    (FIXTURES / fixture).read_text(),
                )
            ),
        ) as fetch:
            data = json.loads(web.call(name, args))
            assert ".wikipedia.org/w/api.php?" in fetch.call_args.args[0]
            if name == "wikipedia_search":
                assert data["results"][0]["snippet"] == "Python is a language."
                assert data["results"][0]["url"] == "https://ja.wikipedia.org/wiki/Python"
            else:
                assert data["text"].startswith("Python is a programming language.")
                assert data["url"] == "https://en.wikipedia.org/wiki/Python"
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(
            return_value=(
                "https://example.org",
                "text/html",
                "<p>" + "x" * 9000 + "</p>",
            )
        ),
    ):
        data = json.loads(web.call("read_web_page", {"url": "https://example.org"}))
        assert len(data["text"]) == 8000 and data["truncated"]


@pytest.mark.parametrize(
    "text", ['{"query":{"pages":[]}}', '{"query":{"pages":[{"missing":true,"title":"Gone"}]}}']
)
def test_missing_wikipedia(text):
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(return_value=("", "application/json", text)),
    ):
        assert json.loads(WebTools().call("wikipedia_read", {"title": "Gone"}))["missing"]


@pytest.mark.parametrize("text", ["invalid", "[]", '{"error":{}}', '{"query":{"pages":{}}}'])
def test_bad_wikipedia_response(text):
    with patch(
        "nezubot_discrod.web_tools.fetch_text",
        AsyncMock(return_value=("", "application/json", text)),
    ):
        assert "Web tool error" in WebTools().call("wikipedia_read", {"title": "Gone"})


INVALID = [
    ("web_search", {}),
    ("web_search", []),
    ("web_search", {"query": "x", "limit": 9}),
    ("web_search", {"query": "x", "limit": True}),
    ("web_search", {"query": " "}),
    ("web_search", {"query": 42}),
    ("web_search", {"query": "x", "url": "https://example.org"}),
    ("wikipedia_search", {"query": "x", "lang": "en.evil.com"}),
    ("wikipedia_read", {"title": "x", "lang": "../en"}),
    ("wikipedia_read", {"title": None}),
    ("read_web_page", {"url": "https://example.org", "headers": {}}),
]


@pytest.mark.parametrize("name,args", INVALID)
def test_schema_validation_in_both_providers(name, args):
    web = WebTools()
    with patch("nezubot_discrod.web_tools.fetch_text", AsyncMock()) as fetch:
        client, sent = client_for(message(calls=[tool(json.dumps(args), name)]), message("done"))
        ai = Assistant(client, "test", SandboxSettings(), web=web)
        asyncio.run(ai.reply(1, "test"))
        assert "Tool error" in sent[1]["messages"][-1]["content"]
        process = FakeProcess(conversation(codex_tool(args, name), *finished()))
        asyncio.run(AppServer(process, SandboxSettings()).complete("test", [], "/empty", web=web))
        result = next(packet["result"] for packet in process.sent if packet.get("id") == 99)
        assert not result["success"]
        fetch.assert_not_awaited()


def test_both_provider_dispatch_prompt_allowlist_and_switch(monkeypatch):
    web = WebTools()
    with patch.object(web, "call", return_value="untrusted page") as call:
        client, sent = client_for(
            message(calls=[tool('{"query":"test"}', "web_search")]), message("done")
        )
        ai = Assistant(client, "test", SandboxSettings(), web=web)
        asyncio.run(ai.reply(1, "test"))
        call.assert_called_once_with("web_search", {"query": "test"})
        assert sent[1]["messages"][-1]["content"] == "untrusted page"
        assert WEB_GUIDANCE in sent[0]["messages"][0]["content"]
        process = FakeProcess(
            conversation(
                codex_tool({"url": "https://example.org"}, "read_web_page", threadId="wrong"),
                codex_tool({"url": "https://example.org"}, "read_web_page"),
                *finished(),
            )
        )
        server = AppServer(process, SandboxSettings())
        asyncio.run(server.complete("test", [], "/empty", web=web))
        replies = [p["result"] for p in process.sent if p.get("id") == 99]
        assert [reply["success"] for reply in replies] == [False, True]
        start = next(p["params"] for p in process.sent if p.get("method") == "thread/start")
        assert start["baseInstructions"] == conversation_prompt(web=True)
        assert {t["function"]["name"] for t in WEB_TOOLS} <= {
            t["name"] for t in start["dynamicTools"]
        }
        with pytest.raises(CodexSecurityError):
            asyncio.run(
                server.complete(
                    "test", [{"role": "system", "content": "injected"}], "/empty", web=web
                )
            )
    assert all(not item["function"]["parameters"]["additionalProperties"] for item in WEB_TOOLS)
    monkeypatch.delenv("WEB_TOOLS_ENABLED", raising=False)
    assert WebTools.from_env() is not None
    monkeypatch.setenv("WEB_TOOLS_ENABLED", "false")
    assert WebTools.from_env() is None
    assert (
        Assistant(Mock(), "test", SandboxSettings(), web=web, system_prompt=SELF_MOD_PROMPT).web
        is None
    )
