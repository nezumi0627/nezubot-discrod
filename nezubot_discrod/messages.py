import json
import re
from urllib.parse import urlsplit


class Source(str):
    def __new__(cls, url, *, title=None, evidence=None):
        source = super().__new__(cls, url)
        source.title_text = title if isinstance(title, str) and title.strip() else None
        source.evidence = evidence if isinstance(evidence, str) else None
        return source


class Reply(str):
    def __new__(cls, text, *, usage=None, sources=()):
        reply = super().__new__(cls, text)
        reply.usage = usage
        reply.sources = tuple(sources)
        return reply


def token_usage(value):
    def field(name):
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    counts = field("prompt_tokens"), field("completion_tokens")
    return counts if all(type(count) is int and count >= 0 for count in counts) else None


def strip_small_text(text):
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("-#")
    ).strip()


INLINE_LINK = re.compile(
    r"\[([^\[\]\n]{1,80})\]\((?:<(https?://[^\s<>]+)>|(https?://[^\s<>()]+))\)"
)
CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
CITATION_LABEL = re.compile(r"[\u2460-\u2473\s,、]+|\d{1,2}(?:[\s,、]+\d{1,2})*")


def verify_inline_links(text, allowed):
    """Drop inline citation markers; keep other links only to URLs tools returned."""
    allowed = set(allowed)

    def replace(match):
        label, url = match[1], match[2] or match[3]
        if CITATION_LABEL.fullmatch(label):
            return ""
        return f"[{label}](<{url}>)" if url in allowed else label

    parts = CODE.split(text)
    return "".join(
        part if index % 2 else INLINE_LINK.sub(replace, part)
        for index, part in enumerate(parts)
    )


CONTENT_RUN = re.compile(r"[0-9A-Za-z\u30a0-\u30ff\u3400-\u9fff\uff10-\uff5a]{3,}")


def _content_grams(text):
    text = re.sub(r"https?://\S+", " ", text.casefold())
    return {run[i:i + 3] for run in CONTENT_RUN.findall(text) for i in range(len(run) - 2)}


def used_sources(answer, sources):
    """Keep sources the answer draws on; sources without evidence text are kept."""
    grams = _content_grams(answer)
    return [
        source for source in sources
        if getattr(source, "evidence", None) is None
        or grams & _content_grams(source.evidence)
    ]


def clean_sources(urls):
    sources = []
    for url in urls:
        if (not isinstance(url, str) or len(url.encode("utf-16-le")) // 2 > 400
                or re.search(r"[\s<>]", url)):
            continue
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                     and not parsed.username and not parsed.password)
        except ValueError:
            continue
        if valid:
            if url not in sources:
                sources.append(url)
            elif getattr(url, "title_text", None):
                index = sources.index(url)
                if not getattr(sources[index], "title_text", None):
                    sources[index] = url
    return sources


def tool_sources(name, result):
    if not isinstance(result, str):
        return []
    if name in {"search_server_log", "read_server_log", "collect_server_log"}:
        structured = getattr(result, "sources", None)
        if structured is not None:
            return clean_sources(structured)
        return clean_sources(re.findall(
            r"https://discord\.com/channels/\d+/\d+/\d+$", result, re.MULTILINE,
        ))
    if name not in {
        "web_search", "read_web_page", "wikipedia_search", "wikipedia_read", "read_server_wiki"
    }:
        return []
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    if name == "read_server_wiki":
        return clean_sources([
            source["url"] for page in data["pages"]
            for item in [*page["entries"], *page["aliases"]] for source in item["sources"]
        ])
    hits = data.get("results", [])
    if not isinstance(hits, list):
        hits = []
    return clean_sources([
        Source(hit["url"], title=hit.get("title")) for hit in [data, *hits]
        if isinstance(hit, dict) and isinstance(hit.get("url"), str)
    ])


def collect_tool_sources(sources, name, result):
    urls = tool_sources(name, result)
    if name in {"read_web_page", "wikipedia_read", "read_server_log"}:
        return clean_sources([*urls, *sources])
    return clean_sources([*sources, *urls])


TEASER_TRIGGER_CHARS = 200
TEASER_TARGET_CHARS = 100


def make_teaser(text, target=TEASER_TARGET_CHARS):
    """First ~target chars of a long answer for the channel, cut at a
    sentence or line boundary when one is nearby, plus the thread pointer."""
    head = text[: target + 30]
    cut = -1
    for mark in ("。", "！", "？", "\n"):
        cut = max(cut, head.rfind(mark, target // 2))
    if cut >= 0:
        return f"{head[:cut + 1].rstrip()}\n続きはコチラから⬇️"
    return f"{text[:target]}…\n続きはコチラから⬇️"


def split_reply(reply, model, limit=2000):
    model = " ".join(model.split())[:120]
    usage = getattr(reply, "usage", None)
    suffix = f"-# {model}" + (
        f" · 入力 {usage[0]:,} / 出力 {usage[1]:,} tokens" if usage is not None else ""
    )
    parts = split_message(strip_small_text(reply), limit)
    if parts and len((parts[-1] + "\n" + suffix).encode("utf-16-le")) // 2 <= limit:
        parts[-1] += "\n" + suffix
    else:
        parts.extend(split_message(suffix, limit))
    return parts


def split_message(text: str, limit: int = 2000) -> list[str]:
    if limit < 2:
        raise ValueError("limit must be at least 2")
    # Discord counts UTF-16 code units, not Python Unicode code points.
    parts, current, size = [], [], 0
    for char in text:
        width = len(char.encode("utf-16-le")) // 2
        if size + width > limit:
            parts.append("".join(current))
            current, size = [], 0
        current.append(char)
        size += width
    if current:
        parts.append("".join(current))
    return parts
