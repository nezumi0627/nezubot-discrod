"""Real-browser fallback tier for web_search.

The HTTP tier in web_tools.py stays the primary path: it is cheap and
usually answers. It escalates here only when the HTTP endpoints return
no results (blocked, rate-limited or empty), because driving a real
browser costs seconds and hundreds of MB per launch.

The approach follows the Sakana agent's browser search (drive a real
Chrome, scrape the rendered result pages of DuckDuckGo Lite, then the
HTML endpoint, then Bing; unwrap redirect URLs; warn the model that
snippets are untrusted external text), with these changes for this
deployment:

- escalation-only: the browser never runs when the HTTP tier answered;
- one shared browser, launched on demand and closed after an idle
  timeout, instead of a resident process (RAM is scarce on this VM);
- heavy page resources (images, media, fonts, stylesheets) are
  aborted: only the result DOM is needed, and this VM has 1.5 cores;
- engine URLs and every result URL pass the same SSRF vetting as the
  HTTP tier (vet_url / vet_proxy_url / _finish_results).

Browser executable: Playwright's managed Chromium is preferred (it
matches the driver exactly); a system Chrome/Chromium is the
fallback, and WEB_SEARCH_BROWSER_EXECUTABLE overrides both. Two
egress-proxy quirks are routed around, not fought:

- when the proxy intercepts TLS with its own CA, system clients
  trust it via the OS store but Chromium does not read that store,
  so when WEB_SEARCH_BROWSER_CA_FILE points at the CA's PEM the
  browser is launched with a dedicated HOME whose .pki/nssdb has
  the CA imported (certutil, trusted-CA flags) before launch;
- connections from a Chromium process straight to an egress proxy
  can fail (ERR_TUNNEL_CONNECTION_FAILED) while the same CONNECT
  from Python succeeds, so the browser is pointed at a loopback
  CONNECT relay (below) and Python makes the upstream connection.

Everything here fails soft: Playwright missing, no Chromium binary,
a crashed browser or a failed navigation is logged and reported as
"no results", so web_search keeps working HTTP-only.
"""

import asyncio
import base64
import logging
import os
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlsplit

from nezubot_discrod.web_tools import (
    WebError,
    _finish_results,
    parse_ddg,
    parse_ddg_lite,
    proxy_for_url,
    vet_proxy_url,
    vet_url,
)

logger = logging.getLogger(__name__)

# Optional CA PEM for a TLS-intercepting egress proxy (see the module
# docstring). Empty/unset means the browser uses the default profile.
EGRESS_CA_FILE = os.getenv("WEB_SEARCH_BROWSER_CA_FILE") or ""
# System Chrome/Chromium executables tried when Playwright's managed
# build is absent (the managed build is always preferred).
SYSTEM_CHROMIUM_CANDIDATES = (
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)
# One shared browser, launched on demand and closed after this much
# idle time: the tier exists for rare escalations, not steady traffic.
BROWSER_IDLE_TIMEOUT = 300
# One budget for the whole tier, so a bad network day cannot eat the
# web_search call budget (CALL_TIMEOUT in web_tools).
BROWSER_SEARCH_TIMEOUT = 55
NAVIGATION_TIMEOUT_MS = 15000
# Let the result DOM settle briefly after domcontentloaded.
SETTLE_MS = 400
DEFAULT_LIMIT = 5
FALLBACK_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}
# Cap simultaneous relay tunnels: a results page opens a handful, and
# this VM does not have the cores for more.
RELAY_MAX_TUNNELS = 16
RELAY_CONNECT_TIMEOUT = 15


class BingResults(HTMLParser):
    """Extract results from a rendered Bing results page.

    Each result is an <li class="b_algo">: the first http(s) anchor is
    the result link (its text is the title), <p> text is the snippet,
    and news results carry a <span class="news_dt"> date. Links
    usually point at bing.com/ck/a redirectors whose `u` parameter
    holds the real URL (see _unwrap_bing).
    """

    _VOID = {"br", "img", "wbr", "hr", "input", "meta", "link"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.current = None
        self.depth = 0
        self.field = None
        self.field_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if self.current is None:
            if tag == "li" and "b_algo" in classes:
                self.current = {"title": "", "url": "", "snippet": "", "date": ""}
                self.results.append(self.current)
                self.depth = 1
            return
        if tag not in self._VOID:
            self.depth += 1
        if self.field is not None:
            if tag not in self._VOID:
                self.field_depth += 1
            return
        if tag == "a" and not self.current["url"]:
            href = attrs.get("href", "")
            if href.startswith(("http://", "https://")):
                self.current["url"] = href
                self.field, self.field_depth = "title", 1
        elif tag == "p":
            self.field, self.field_depth = "snippet", 1
        elif "news_dt" in classes:
            self.field, self.field_depth = "date", 1

    def handle_endtag(self, tag):
        if self.current is None:
            return
        if self.field is not None:
            self.field_depth -= 1
            if self.field_depth <= 0:
                self.field = None
        if tag not in self._VOID:
            self.depth -= 1
            if self.depth <= 0:
                self.current = None

    def handle_data(self, data):
        if self.field and self.current is not None:
            self.current[self.field] += data


def _unwrap_bing(url):
    parsed = urlsplit(url)
    if (parsed.hostname or "").endswith("bing.com") and parsed.path.startswith("/ck/"):
        encoded = parse_qs(parsed.query).get("u", [""])[0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            return base64.urlsafe_b64decode(padded).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return url
    return url


def parse_bing(html, limit):
    parser = BingResults()
    parser.feed(html)
    items = []
    for item in parser.results:
        item = {**item, "url": _unwrap_bing(item["url"])}
        # Bing's title anchors also carry a display-URL breadcrumb
        # ("Examplehttps://example.org › ..."); cut it off.
        title = item["title"]
        for scheme in ("https://", "http://"):
            cut = title.find(scheme, 1)
            if cut > 0:
                item = {**item, "title": title[:cut]}
                break
        items.append(item)
    results = _finish_results(items, limit)
    # _finish_results shapes title/url/snippet only; re-attach dates.
    dates = {
        item["url"]: " ".join(item["date"].split()) for item in items if item["date"]
    }
    for result in results:
        date = dates.get(result["url"])
        if date:
            result["date"] = date
    return results


# Engine order: the lite endpoint is the lightest page and a separate
# host from the HTML endpoint, then Bing as the independent fallback.
_ENGINES = (
    ("ddg-lite", "https://lite.duckduckgo.com/lite/?q=", parse_ddg_lite),
    ("ddg-html", "https://html.duckduckgo.com/html/?q=", parse_ddg),
    ("bing", "https://www.bing.com/search?q=", parse_bing),
)


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


class _EgressRelay:
    """Loopback CONNECT relay from Chromium to the egress proxy.

    Chromium's own connections to this cell's egress proxy fail while
    Python's succeed (see module docstring), so the browser is told
    its proxy is 127.0.0.1:<port> and this relay — running in the bot
    process — performs the upstream CONNECT with the proxy
    credentials sent proactively, the way aiohttp does. Targets are
    vetted with the same rules as proxied fetches (vet_url /
    vet_proxy_url): CONNECT only to port 443, plain-HTTP forwarding
    only for GET/HEAD. The proxy value and credentials are never
    logged.
    """

    def __init__(self, upstream=None):
        # upstream: egress proxy URL; None resolves via proxy_for_url
        # at start() time. An explicit value is a test seam.
        self._upstream_arg = upstream
        self._upstream = None
        self._server = None
        self.port = None
        self._writers = set()
        self._slots = asyncio.Semaphore(RELAY_MAX_TUNNELS)

    async def start(self):
        """Start listening; return the port, or None without a proxy."""
        if self._upstream_arg is not None:
            self._upstream = urlsplit(self._upstream_arg)
        else:
            proxy = proxy_for_url("https://lite.duckduckgo.com/")
            if proxy is None:
                return None
            self._upstream = urlsplit(proxy)
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self):
        if self._server is not None:
            self._server.close()
            self._server = None
        for writer in list(self._writers):
            try:
                writer.close()
            except Exception:
                pass
        self._writers.clear()
        self.port = None

    def _upstream_auth(self):
        if not self._upstream.username:
            return None
        creds = (
            f"{unquote(self._upstream.username)}:"
            f"{unquote(self._upstream.password or '')}"
        )
        return "Proxy-Authorization: Basic " + base64.b64encode(
            creds.encode()
        ).decode()

    async def _open_upstream(self):
        return await asyncio.wait_for(
            asyncio.open_connection(
                self._upstream.hostname, self._upstream.port or 3128
            ),
            RELAY_CONNECT_TIMEOUT,
        )

    async def _handle(self, client_reader, client_writer):
        self._writers.add(client_writer)
        try:
            async with self._slots:
                await self._serve(client_reader, client_writer)
        except Exception as error:
            logger.debug("relay connection failed: %s", type(error).__name__)
        finally:
            self._writers.discard(client_writer)
            try:
                client_writer.close()
            except Exception:
                pass

    async def _serve(self, client_reader, client_writer):
        raw = await asyncio.wait_for(
            client_reader.readuntil(b"\r\n\r\n"), RELAY_CONNECT_TIMEOUT
        )
        head = raw.decode("iso-8859-1")
        lines = head.split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        if method == "CONNECT":
            await self._serve_connect(client_reader, client_writer, target)
        elif method in {"GET", "HEAD"}:
            await self._serve_http(client_writer, lines)
        else:
            client_writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            await client_writer.drain()

    async def _serve_connect(self, client_reader, client_writer, target):
        host, _, port = target.rpartition(":")
        try:
            if port != "443":
                raise WebError("relay only tunnels port 443")
            vet_url(f"https://{host}/")
            vet_proxy_url(f"https://{host}/")
        except WebError:
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return
        try:
            up_reader, up_writer = await self._open_upstream()
        except Exception:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
            return
        self._writers.add(up_writer)
        try:
            headers = [f"CONNECT {target} HTTP/1.1", f"Host: {target}"]
            auth = self._upstream_auth()
            if auth:
                headers.append(auth)
            up_writer.write(("\r\n".join(headers) + "\r\n\r\n").encode())
            await up_writer.drain()
            response = await asyncio.wait_for(
                up_reader.readuntil(b"\r\n\r\n"), RELAY_CONNECT_TIMEOUT
            )
            status = response.decode("iso-8859-1").split("\r\n")[0]
            if " 200 " not in status:
                logger.debug("relay upstream refused CONNECT: %s", status)
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_writer.drain()
                return
            client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await client_writer.drain()
            await asyncio.gather(
                _pipe(client_reader, up_writer), _pipe(up_reader, client_writer)
            )
        finally:
            self._writers.discard(up_writer)
            try:
                up_writer.close()
            except Exception:
                pass

    async def _serve_http(self, client_writer, lines):
        url = lines[0].split(" ", 2)[1]
        try:
            canonical = vet_url(url)
            if urlsplit(canonical).scheme != "http":
                raise WebError("relay forwards plain HTTP only")
            vet_proxy_url(canonical)
        except WebError:
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return
        up_reader, up_writer = await self._open_upstream()
        self._writers.add(up_writer)
        try:
            forwarded = [lines[0]]
            for line in lines[1:]:
                name = line.split(":", 1)[0].strip().lower()
                if name in {"proxy-authorization", "proxy-connection", "connection"}:
                    continue
                if line:
                    forwarded.append(line)
            auth = self._upstream_auth()
            if auth:
                forwarded.append(auth)
            forwarded.append("Connection: close")
            up_writer.write(("\r\n".join(forwarded) + "\r\n\r\n").encode())
            await up_writer.drain()
            await _pipe(up_reader, client_writer)
        finally:
            self._writers.discard(up_writer)
            try:
                up_writer.close()
            except Exception:
                pass


async def _ensure_egress_ca_trusted(home):
    """Import the configured egress CA into <home>/.pki/nssdb (best effort).

    The egress proxy MITMs TLS with this CA; system clients trust it
    through the OS store, Chromium through the NSS DB under its HOME.
    Re-imports under a fixed nickname so CA rotation is picked up.
    """
    dbdir = os.path.join(home, ".pki", "nssdb")
    db = f"sql:{dbdir}"

    async def run(*args):
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    try:
        os.makedirs(dbdir, exist_ok=True)
        if not os.path.exists(os.path.join(dbdir, "cert9.db")):
            await run("certutil", "-N", "-d", db, "--empty-password")
        await run("certutil", "-D", "-n", "egress-proxy", "-d", db)
        await run(
            "certutil", "-A", "-n", "egress-proxy", "-t", "CT,C,c",
            "-i", EGRESS_CA_FILE, "-d", db,
        )
    except Exception as error:
        logger.warning("egress CA import failed: %s", type(error).__name__)


async def _browser_env():
    """Environment for the browser process, or None for the default.

    When an egress CA is configured, the browser gets a dedicated
    HOME under the bot's data dir so its NSS DB can carry the egress
    CA: the user's own ~/.pki/nssdb may be read-only, and its trust
    flags may not cover TLS server verification.
    """
    if not os.path.exists(EGRESS_CA_FILE):
        return None
    home = os.path.join(
        os.path.expanduser("~"), ".nezubot-discrod", "browser-home"
    )
    await _ensure_egress_ca_trusted(home)
    return {**os.environ, "HOME": home}


class BrowserSearch:
    """On-demand shared Chromium for search escalation.

    Searches are serialized through one lock (one page at a time on
    this small VM). The browser relaunches transparently after a
    crash. `search` never raises.
    """

    def __init__(self, *, idle_timeout=BROWSER_IDLE_TIMEOUT, executable=None):
        self._idle_timeout = idle_timeout
        if executable is None:
            executable = os.getenv("WEB_SEARCH_BROWSER_EXECUTABLE") or None
        self._executable = executable
        self._lock = asyncio.Lock()
        self._playwright = None
        self._browser = None
        self._relay = None
        self._idle_handle = None
        self._generation = 0

    async def search(self, query, limit=DEFAULT_LIMIT, page=1):
        try:
            async with asyncio.timeout(BROWSER_SEARCH_TIMEOUT):
                async with self._lock:
                    return await self._search(query, limit, page)
        except Exception as error:  # the tier must never break web_search
            logger.warning("browser search tier failed: %s", type(error).__name__)
            return []
        finally:
            self._schedule_idle_close()

    async def _search(self, query, limit, page=1):
        self._generation += 1
        offset = (page - 1) * 10
        for name, base, parser in _ENGINES:
            url = base + quote_plus(query)
            if offset:
                # DuckDuckGo takes a result offset; Bing a 1-based
                # first-result index.
                url += f"&first={offset + 1}" if name == "bing" else f"&s={offset}"
            try:
                vet_url(url)
                if proxy_for_url(url) is not None:
                    vet_proxy_url(url)
            except WebError:
                continue
            html = await self._render(url)
            if html is None:
                continue
            hits = parser(html, limit)
            if hits:
                logger.info("browser search via %s: %d hits", name, len(hits))
                return hits
        return []

    async def _render(self, url):
        """Render *url* in a fresh context; return its HTML or None."""
        try:
            browser = await self._ensure_browser()
            if browser is None:
                return None
            context = await browser.new_context(
                locale="ja-JP", user_agent=self._user_agent()
            )
            try:
                await context.route("**/*", _block_heavy_resources)
                page = await context.new_page()
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=NAVIGATION_TIMEOUT_MS,
                )
                await page.wait_for_timeout(SETTLE_MS)
                return await page.content()
            finally:
                await context.close()
        except Exception as error:
            logger.warning(
                "browser render failed for %s: %s: %s",
                urlsplit(url).hostname,
                type(error).__name__,
                str(error).splitlines()[0][:200] if str(error) else "",
            )
            if self._browser is not None and not self._browser.is_connected():
                await self._drop_browser()
            return None

    def _user_agent(self):
        version = getattr(self._browser, "version", "") or ""
        if version:
            return (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                f"(KHTML, like Gecko) Chrome/{version} Safari/537.36"
            )
        return FALLBACK_USER_AGENT

    async def _ensure_browser(self):
        if self._browser is not None:
            if self._browser.is_connected():
                return self._browser
            await self._drop_browser()
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning(
                "playwright is not installed; browser search tier unavailable"
            )
            return None
        try:
            self._playwright = await async_playwright().start()
            launch = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-first-run",
                    "--disable-extensions",
                ],
            }
            env = await _browser_env()
            if env is not None:
                launch["env"] = env
            executable = self._executable or self._resolve_executable()
            if executable:
                launch["executable_path"] = executable
            # Route the browser through the loopback relay whenever an
            # egress proxy is configured (see _EgressRelay). Without a
            # proxy the browser connects directly and no relay runs.
            self._relay = _EgressRelay()
            port = await self._relay.start()
            if port is not None:
                launch["proxy"] = {"server": f"http://127.0.0.1:{port}"}
            else:
                self._relay = None
            self._browser = await self._playwright.chromium.launch(**launch)
        except Exception as error:
            logger.warning(
                "chromium launch failed: %s: %s",
                type(error).__name__,
                str(error).splitlines()[0][:200] if str(error) else "",
            )
            await self._drop_browser()
            return None
        return self._browser

    def _resolve_executable(self):
        # A Playwright-managed Chromium matches the driver exactly;
        # prefer it when present. Otherwise fall back to a system
        # Chrome/Chromium. None lets Playwright raise its "executable
        # doesn't exist" error, which _ensure_browser already turns
        # into a soft failure.
        managed = self._playwright.chromium.executable_path
        if managed and os.path.exists(managed):
            return managed
        for candidate in SYSTEM_CHROMIUM_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        return None

    async def _drop_browser(self):
        browser, playwright, relay = self._browser, self._playwright, self._relay
        self._browser = None
        self._playwright = None
        self._relay = None
        if relay is not None:
            await relay.stop()
        for close in (
            (browser.close if browser is not None else None),
            (playwright.stop if playwright is not None else None),
        ):
            if close is None:
                continue
            try:
                await close()
            except Exception:
                pass

    def _schedule_idle_close(self):
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None
        if self._browser is None:
            return
        generation = self._generation
        loop = asyncio.get_running_loop()
        self._idle_handle = loop.call_later(
            self._idle_timeout, self._idle_close_due, generation
        )

    def _idle_close_due(self, generation):
        self._idle_handle = None
        # A search that started after this timer was set reschedules
        # its own close; do not kill the browser from under it.
        if generation != self._generation or self._browser is None:
            return
        asyncio.ensure_future(self._drop_browser())


async def _block_heavy_resources(route):
    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()
