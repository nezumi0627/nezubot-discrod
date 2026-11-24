import json
import sqlite3

from nezubot_discrod.messages import Source
from nezubot_discrod.server_log import format_hits
from nezubot_discrod.server_wiki import ServerWiki


class LogResult(str):
    def __new__(cls, text, *, sources=()):
        result = super().__new__(cls, text)
        result.sources = tuple(sources)
        return result

    @classmethod
    def from_hits(cls, hits):
        hits = list(hits)
        sources = tuple(Source(
            hit.url, title=f"#{hit.channel_name} · {hit.author} · {hit.date[:10]}",
            evidence=hit.text,
        ) for hit in hits)
        return cls(format_hits(hits), sources=sources)


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


LOG_TOOLS = [
    _tool(
        "search_server_log",
        "Search the whole permission-visible server history, not just recent chat. Returns "
        "speaker name, author ID, timestamp and source URL; excerpts are at most 300 characters. "
        "The author filter accepts a username, global name, server nickname, ID or <@ID>; "
        "'me', '自分', 'おれ' resolve to the caller. 'nameが言ってたtopic', 'name topic' or "
        "'from:name topic' queries also filter by speaker; a name alone returns that speaker's "
        "latest messages. Omit query with author to list latest messages (default 10, max 20); "
        "page backwards with before=last timestamp.",
        [],
        {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000},
            **{
                key: {"type": "string", "minLength": 1, "maxLength": 100}
                for key in ("author", "channel", "after", "before")
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    ),
    _tool(
        "collect_server_log",
        "Collect broad evidence about a person or topic from the whole visible server history: "
        "total matching count, first/last dates, channel counts and a diverse cited sample "
        "across channels and eight time strata, not an inferred biography. Provide author "
        "(ID, mention, name or 'me') and/or query (lexical topic terms). Default 24 excerpts, "
        "max 60, each <=300 chars; page the sample with offset/next_offset (max 1000). "
        "Ambiguous names return IDs to disambiguate, never a merged profile.",
        [],
        {
            "query": {"type": "string", "minLength": 1, "maxLength": 2000},
            "author": {"type": "string", "minLength": 1, "maxLength": 100},
            "limit": {"type": "integer", "minimum": 1, "maximum": 60},
            "offset": {"type": "integer", "minimum": 0, "maximum": 1000},
        },
    ),
    _tool(
        "read_server_log",
        "Read surrounding messages in a visible channel, including each speaker's display name, "
        "author ID, timestamp and source URL.",
        ["message_id"],
        {
            "message_id": {"type": "string", "minLength": 1, "maxLength": 100},
            "direction": {"type": "string", "enum": ["around", "before", "after"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 30},
        },
    ),
    _tool(
        "read_server_wiki",
        "Read a cited Server Wiki person or event page from this channel's archive.",
        ["kind", "query"],
        {
            "kind": {"type": "string", "enum": ["person", "event"]},
            "query": {"type": "string", "minLength": 1, "maxLength": 100},
        },
    ),
]
LOG_PARAMETERS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in LOG_TOOLS}


class LogTools:
    def __init__(self, store, channel_ids, *, server_wiki_channel=None, caller_id=None):
        self.store = store
        self.channel_ids = tuple(str(channel) for channel in channel_ids)
        self.server_wiki_channels = (
            (str(server_wiki_channel),)
            if str(server_wiki_channel) in self.channel_ids
            else ()
        )
        self.caller_id = str(caller_id) if caller_id is not None else None

    def call(self, name, args):
        schema = LOG_PARAMETERS.get(name)
        if (
            schema is None
            or not isinstance(args, dict)
            or set(args) - schema["properties"].keys()
            or not set(schema["required"]).issubset(args)
        ):
            raise ValueError("Invalid server log tool arguments")
        for key, value in args.items():
            spec = schema["properties"][key]
            if spec["type"] == "integer":
                valid = type(value) is int and spec["minimum"] <= value <= spec["maximum"]
            else:
                valid = (
                    isinstance(value, str)
                    and bool(value.strip())
                    and len(value) <= spec.get("maxLength", 100)
                    and ("enum" not in spec or value in spec["enum"])
                )
            if not valid:
                raise ValueError("Invalid server log tool arguments")
        args = dict(args)
        if name in {"search_server_log", "collect_server_log"}:
            if not args.get("query") and not args.get("author"):
                raise ValueError("A topic or author is required")
            if args.get("author", "").casefold() in {
                "me", "myself", "自分", "おれ", "俺", "私", "僕",
            }:
                if self.caller_id is None:
                    raise ValueError("Caller identity is unavailable; provide a Discord ID")
                args["author"] = self.caller_id
            args.setdefault("query", "")
        try:
            if name == "read_server_wiki":
                if not self.server_wiki_channels:
                    return "Server Wiki wiki is unavailable outside a visible server channel."
                return json.dumps(
                    {"pages": ServerWiki(self.store).lookup(
                        args["kind"], args["query"], self.server_wiki_channels
                    )}, ensure_ascii=False,
                )
            if name == "collect_server_log":
                overview = self.store.collect(channel_ids=self.channel_ids, **args)
                hits = overview.pop("hits")
                result = LogResult.from_hits(hits)
                return LogResult(json.dumps(overview, ensure_ascii=False) + "\n" + result,
                                 sources=result.sources)
            method = self.store.search if name == "search_server_log" else self.store.read
            return LogResult.from_hits(method(channel_ids=self.channel_ids, **args))
        except (OSError, sqlite3.Error):
            return "Server log unavailable; try again later."
