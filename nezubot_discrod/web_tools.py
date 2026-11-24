import asyncio
import ipaddress
import json
import os
import re
import socket
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit

import aiohttp
from yarl import URL

from nezubot_discrod.persona import PERSONA

MAX_BYTES = 2 * 1024 * 1024
MAX_TEXT = 8000
USER_AGENT = f"{PERSONA.bot_name}/1.0" + (f" ({PERSONA.repo_url})" if PERSONA.repo_url else "")
SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0", "Accept": "text/html", "Accept-Language": "ja,en;q=0.8",
}


class WebError(ValueError):
    pass


def has_web_content(result):
    return not (isinstance(result, str) and result.startswith("Web tool error:"))


class HTMLCharset(HTMLParser):
    def __init__(self):
        super().__init__()
        self.encoding = None

    def handle_starttag(self, tag, attrs):
        if tag != "meta" or self.encoding is not None:
            return
        attrs = dict(attrs)
        if attrs.get("charset"):
            self.encoding = attrs["charset"].strip()
        elif (attrs.get("http-equiv") or "").lower() == "content-type":
            match = re.search(r"charset\s*=\s*[\"']?([a-zA-Z0-9._-]+)",
                              attrs.get("content") or "", re.IGNORECASE)
            if match:
                self.encoding = match[1]


def decode_text(body, media, charset):
    if not charset and body.startswith(b"\xef\xbb\xbf"):
        return body.decode("utf-8-sig", errors="replace")
    if not charset and media == "text/html":
        parser = HTMLCharset()
        parser.feed(bytes(body[:4096]).decode("ascii", errors="ignore"))
        charset = parser.encoding
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def public_ip(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise WebError("Invalid IP address") from error
    if (
        not address.is_global
        or address.is_private
        or address.is_reserved
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or "%" in value
        or (
            isinstance(address, ipaddress.IPv6Address)
            and (
                address.ipv4_mapped
                or address.sixtofour
                or address.teredo
                or address.is_site_local
                or address in ipaddress.ip_network("64:ff9b::/96")
                or address in ipaddress.ip_network("64:ff9b:1::/48")
            )
        )
    ):
        raise WebError("Non-public IP addresses are forbidden")
    return str(address)


def vet_url(url):
    if (
        not isinstance(url, str)
        or len(url) > 4096
        or any(ord(c) < 33 or ord(c) == 127 for c in url)
    ):
        raise WebError("Invalid URL")
    try:
        canonical = str(URL(url))
        parsed = urlsplit(canonical)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise WebError("Invalid URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port == 0
    ):
        raise WebError("Only HTTP(S) URLs without credentials are allowed")
    if host.rstrip(".").lower() == "localhost" or host.lower().rstrip(".").endswith(".localhost"):
        raise WebError("Localhost is forbidden")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if ":" in host or "%" in host or re.fullmatch(r"[0-9.]+", host):
            raise WebError("Invalid hostname") from None
    else:
        public_ip(host)
    return canonical


class PublicResolver(aiohttp.abc.AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host,
            port,
            family=family,
            type=socket.SOCK_STREAM,
        )
        if not addresses:
            raise WebError("Hostname has no addresses")
        # Pin all returned endpoints; aiohttp connects to these IPs with the original TLS hostname.
        return [
            {
                "hostname": host,
                "host": public_ip(info[4][0]),
                "port": port,
                "family": info[0],
                "proto": info[2],
                "flags": socket.AI_NUMERICHOST,
            }
            for info in addresses
        ]

    async def close(self):
        pass


async def fetch_text(url, *, data=None, headers=None):
    try:
        async with asyncio.timeout(10):
            connector = aiohttp.TCPConnector(
                resolver=PublicResolver(), use_dns_cache=False, force_close=True
            )
            async with aiohttp.ClientSession(
                connector=connector,
                trust_env=False,
                auto_decompress=False,
                cookie_jar=aiohttp.DummyCookieJar(),
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
            ) as session:
                return await _fetch_redirects(session, url, data=data, headers=headers)
    except (aiohttp.ClientError, OSError, TimeoutError, UnicodeError) as error:
        raise WebError(f"Web fetch failed ({type(error).__name__})") from error


async def _fetch_redirects(session, url, *, data=None, headers=None):
    for hop in range(6):
        url = vet_url(url)
        if data is not None:
            request = session.post(url, data=data, headers=headers, allow_redirects=False)
        else:
            request = session.get(url, **({"headers": headers} if headers else {}),
                                  allow_redirects=False)
        async with request as response:
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location or hop == 5:
                    raise WebError("Missing redirect target or too many redirects")
                url = urljoin(url, location)
                if response.status in {301, 302, 303}:
                    data = None
                continue
            if not 200 <= response.status < 300:
                raise WebError(f"HTTP {response.status}")
            media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media not in {"text/html", "text/plain", "application/json"} and not (
                media.startswith("application/") and media.endswith("+json")
            ):
                raise WebError("Unsupported content type")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise WebError("Compressed responses are not accepted")
            if response.content_length is not None and response.content_length > MAX_BYTES:
                raise WebError("Response exceeds 2 MiB")
            body = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise WebError("Response exceeds 2 MiB")
            text = decode_text(body, media, response.charset)
            return url, media, text
    raise WebError("Too many redirects")


class ReadableHTML(HTMLParser):
    BODY_TAGS = {"body", "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                 "article", "section", "main", "table", "ul", "ol"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if self.hidden == ["head"] and tag in self.BODY_TAGS:
            self.hidden.clear()
        if tag in {"script", "style", "noscript", "svg", "template", "head"}:
            self.hidden.append(tag)
        if not self.hidden:
            if tag in self.BODY_TAGS or tag == "tr":
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        if not self.hidden:
            if tag in self.BODY_TAGS or tag == "tr":
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        return "\n".join(
            " ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip()
        )


def readable(html):
    parser = ReadableHTML()
    parser.feed(html)
    for _ in range(20):
        if parser.cdata_elem not in {"script", "style"}:
            break
        if any(tag in parser.hidden for tag in {"template", "noscript", "svg"}):
            break
        boundary = re.search(r"<(?:" + "|".join(parser.BODY_TAGS) + r")\b", parser.rawdata,
                             re.IGNORECASE)
        if boundary is None:
            break
        rest = parser.rawdata[boundary.start():]
        parser.rawdata = ""
        parser.clear_cdata_mode()
        parser.hidden.clear()
        parser.feed(rest)
    return parser.text()


class DDGResults(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.current = None
        self.field = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self.current = {"title": "", "url": attrs.get("href", ""), "snippet": ""}
            self.results.append(self.current)
            self.field, self.depth = "title", 1
        elif "result__snippet" in classes and self.current is not None:
            self.field, self.depth = "snippet", 1
        elif self.field and tag not in {"br", "img", "wbr", "hr", "input", "meta", "link"}:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.field:
            self.depth -= 1
            if self.depth <= 0:
                self.field = None

    def handle_data(self, data):
        if self.field and self.current is not None:
            self.current[self.field] += data


def parse_ddg(html, limit):
    parser = DDGResults()
    parser.feed(html)
    results = []
    for item in parser.results:
        url = urljoin("https://html.duckduckgo.com/", item["url"])
        parsed = urlsplit(url)
        if parsed.hostname in {"duckduckgo.com", "html.duckduckgo.com"}:
            url = parse_qs(parsed.query).get("uddg", [url])[0]
        try:
            vet_url(url)
        except WebError:
            continue
        results.append(
            {
                "title": " ".join(item["title"].split())[:300],
                "url": url,
                "snippet": " ".join(item["snippet"].split())[:600],
            }
        )
        if len(results) == limit:
            break
    return results


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


TEXT = {"type": "string", "minLength": 1, "maxLength": 2000}
LANG = {"type": "string", "pattern": "^[a-z]{2,12}(?:-[a-z0-9]{1,12})*$", "maxLength": 32}
WEB_TOOLS = [
    _tool(
        "web_search",
        "Search the public web using DuckDuckGo HTML.",
        ["query"],
        {"query": TEXT, "limit": {"type": "integer", "minimum": 1, "maximum": 8}},
    ),
    _tool(
        "wikipedia_search", "Search Wikipedia articles.", ["query"], {"query": TEXT, "lang": LANG}
    ),
    _tool(
        "wikipedia_read",
        "Read a Wikipedia plaintext article extract.",
        ["title"],
        {"title": TEXT, "lang": LANG},
    ),
    _tool(
        "read_web_page",
        "Read untrusted public HTTP(S) page text, at most 8000 characters.",
        ["url"],
        {"url": {"type": "string", "minLength": 1, "maxLength": 4096}},
    ),
]
WEB_PARAMETERS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in WEB_TOOLS}


class WebTools:
    @classmethod
    def from_env(cls):
        return (
            cls()
            if os.getenv("WEB_TOOLS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
            else None
        )

    def call(self, name, args):
        schema = WEB_PARAMETERS.get(name)
        if (
            schema is None
            or not isinstance(args, dict)
            or set(args) - schema["properties"].keys()
            or not set(schema["required"]).issubset(args)
        ):
            raise ValueError("Invalid web tool arguments")
        for key, value in args.items():
            spec = schema["properties"][key]
            if spec["type"] == "integer":
                valid = type(value) is int and spec["minimum"] <= value <= spec["maximum"]
            else:
                valid = (
                    isinstance(value, str)
                    and bool(value.strip())
                    and len(value) <= spec["maxLength"]
                    and ("pattern" not in spec or re.fullmatch(spec["pattern"], value))
                )
            if not valid:
                raise ValueError("Invalid web tool arguments")
        try:
            result = asyncio.run(self.execute(name, args))
            return json.dumps(result, ensure_ascii=False)
        except (WebError, json.JSONDecodeError) as error:
            return f"Web tool error: {error}"

    async def execute(self, name, args):
        if name == "web_search":
            _, _, html = await fetch_text(
                "https://html.duckduckgo.com/html/", data={"q": args["query"]},
                headers=SEARCH_HEADERS,
            )
            hits = parse_ddg(html, args.get("limit", 5))
            return {
                "results": hits,
                "note": "" if hits else "No results, or search provider blocked the request.",
            }
        if name == "read_web_page":
            url, media, text = await fetch_text(args["url"])
            if media == "text/html":
                text = readable(text)
            return {"url": url, "text": text[:MAX_TEXT], "truncated": len(text) > MAX_TEXT}
        lang = args.get("lang", "ja")
        base = f"https://{lang}.wikipedia.org"
        params = {"action": "query", "format": "json", "formatversion": "2"}
        if name == "wikipedia_search":
            params.update(list="search", srsearch=args["query"], srlimit="8")
        else:
            params.update(prop="extracts", titles=args["title"], explaintext="1", redirects="1")
        _, _, text = await fetch_text(base + "/w/api.php?" + urlencode(params))
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("query", {}), dict):
            raise WebError("Invalid Wikipedia response")
        if "error" in data:
            raise WebError("Wikipedia API returned an error")
        if name == "wikipedia_search":
            hits = data.get("query", {}).get("search", [])
            if not isinstance(hits, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("title"), str)
                for item in hits
            ):
                raise WebError("Invalid Wikipedia search results")
            return {
                "results": [
                    {
                        "title": item["title"][:300],
                        "url": base + "/wiki/" + quote(item["title"].replace(" ", "_"), safe=""),
                        "snippet": readable(str(item.get("snippet", "")))[:600],
                    }
                    for item in hits[:8]
                ]
            }
        pages = data.get("query", {}).get("pages", [])
        if not isinstance(pages, list) or any(not isinstance(page, dict) for page in pages):
            raise WebError("Invalid Wikipedia pages")
        page = pages[0] if pages else {}
        title = str(page.get("title", args["title"]))[:2000]
        extract = str(page.get("extract", ""))
        return {
            "title": title,
            "url": base + "/wiki/" + quote(title.replace(" ", "_"), safe=""),
            "text": extract[:MAX_TEXT],
            "truncated": len(extract) > MAX_TEXT,
            "missing": bool(page.get("missing", not pages)),
        }
