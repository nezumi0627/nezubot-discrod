import json
import logging
import os
import re
import sqlite3
import stat
import unicodedata
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock

from nezubot_discrod.config import data_dir
from nezubot_discrod.policy import guard_host_write

logger = logging.getLogger(__name__)
MAX_PAGES = 32
MAX_REVISIONS = 32
MAX_CONTENT = 4000
FACT_PAGES = ("profile", "preferences", "relationships", "notes")
_FACT_LINE = re.compile(r"^- \*\*([a-z0-9][a-z0-9_-]{0,47})\*\*: (.+)$", re.MULTILINE)
_CREATE_LOCK = Lock()
PAGE = {"type": "string", "description": "Page name (a short slug, e.g. profile or projects)."}
BODY = {"type": "string", "maxLength": MAX_CONTENT, "description": "Markdown page text."}


def _function(name, description, required, **properties):
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


WIKI_TOOLS = [
    _function("wiki_list", "List the current speaker's wiki pages and revisions.", []),
    _function(
        "wiki_read", "Read a wiki page's latest or retained past revision.", ["page"],
        page=PAGE, revision={"type": "integer", "minimum": 1},
    ),
    _function(
        "wiki_search", "Find the current speaker's wiki pages, most relevant first.", ["query"],
        query={"type": "string", "maxLength": 100},
    ),
    _function(
        "wiki_remember", "Save one durable fact in a structured page, updating its stable key "
        "instead of appending duplicates.", ["page", "key", "text"],
        page={"type": "string", "enum": list(FACT_PAGES)},
        key={
            "type": "string", "maxLength": 48, "description": "Stable slug, e.g. favorite-editor.",
        },
        text={"type": "string", "maxLength": 240, "description": "One short, single-line fact."},
    ),
    _function(
        "wiki_append", "Append Markdown to a wiki page, creating it if needed.", ["page", "text"],
        page=PAGE, text=BODY,
    ),
    _function(
        "wiki_rewrite", "Replace a wiki page if its revision still matches.",
        ["page", "text", "revision"], page=PAGE, text=BODY,
        revision={"type": "integer", "minimum": 1},
    ),
]
TOOL_PARAMETERS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in WIKI_TOOLS}


class WikiError(ValueError):
    pass


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _query_terms(text):
    phrase = _normalized(text)
    spaced = re.sub(r"([\u3040-\u30ff\u3400-\u9fff]+)", r" \1 ", phrase)
    words = re.findall(r"[^\W_]+", spaced)
    for run in re.findall(r"[\u30a0-\u30ff\u3400-\u9fff]{2,}", phrase):
        words.extend(run[index:index + 2] for index in range(len(run) - 1))
    terms = tuple(dict.fromkeys(word for word in words if len(word) >= 2))[:24]
    return phrase, terms


def _relevance(text, phrase, terms):
    if not phrase:
        return 0
    text = _normalized(text)
    return 3 * (phrase in text) + sum(term in text for term in terms)


def _page_relevance(row, phrase, terms):
    page = _normalized(row["page"])
    return (
        12 * (page == phrase)
        + 4 * _relevance(page, phrase, terms)
        + _relevance(row["body"], phrase, terms)
    )


def _merge_fact(body, key, text):
    current = next((match for match in _FACT_LINE.finditer(body) if match[1] == key), None)
    if current is not None and _normalized(current[2]) == _normalized(text):
        return body
    line = f"- **{key}**: {text}"
    if current is not None:
        return body[:current.start()] + line + body[current.end():]
    return f"{body}\n\n{line}" if body else line


def _check_private_file(metadata):
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise PermissionError("Wiki SQLite files must be private regular files with no hard links")


class MemberWiki:
    def __init__(self, path: Path | None = None, *, checkout: str | Path | None = None):
        self.path = path if path is not None else data_dir() / "wiki.db"
        self.checkout = checkout if checkout is not None else Path(__file__).resolve().parents[1]

    def _guarded_path(self):
        root = self.path.parent
        guarded_root = guard_host_write(root, checkout=self.checkout, allowed_root=root)
        if guarded_root != root.parent.resolve() / root.name:
            raise PermissionError("Wiki data directory must not be a symlink")
        guarded_root.mkdir(mode=0o700, exist_ok=True)
        metadata = guarded_root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise PermissionError("Wiki data directory must be private and owned by the bot")
        for suffix in ("", "-journal", "-wal", "-shm"):
            name = self.path.name + suffix
            path = guard_host_write(root / name, checkout=self.checkout, allowed_root=root)
            if path != guarded_root / name:
                raise PermissionError("Wiki SQLite files must not be symlinks")
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            _check_private_file(metadata)
        return guarded_root / self.path.name

    @contextmanager
    def connect(self):
        path = self._guarded_path()
        with _CREATE_LOCK:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                _check_private_file(path.lstat())
            else:
                os.close(fd)
        with closing(sqlite3.connect(path, timeout=30)) as db, db:
            db.row_factory = sqlite3.Row
            # Same WAL tuning as the server-log archive (audit F14): WAL so
            # wiki writes do not block readers, NORMAL sync under WAL.
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = NORMAL")
            db.execute("PRAGMA busy_timeout = 30000")
            db.execute("PRAGMA temp_store = MEMORY")
            db.execute("""
                CREATE TABLE IF NOT EXISTS wiki_revision (
                    scope TEXT NOT NULL, member TEXT NOT NULL, page TEXT NOT NULL,
                    revision INTEGER NOT NULL, body TEXT NOT NULL,
                    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (scope, member, page, revision)
                )
            """)
            yield db

    def for_member(self, member_id: int, guild_id: int | None):
        return MemberPages(self, str(guild_id) if guild_id is not None else "dm", str(member_id))


class MemberPages:
    def __init__(self, wiki: MemberWiki, scope: str, member_id: str):
        self.wiki = wiki
        self.key = (scope, member_id)

    def call(self, name: str, args: dict) -> str:
        parameters = TOOL_PARAMETERS.get(name)
        if parameters is None or not isinstance(args, dict):
            raise WikiError("Unknown wiki tool or invalid arguments.")
        properties = parameters["properties"]
        if args.keys() - properties.keys() or not args.keys() >= set(parameters["required"]):
            raise WikiError("Invalid wiki tool arguments.")
        for key, value in args.items():
            if key == "revision":
                if type(value) is not int or not 1 <= value < 2**63:
                    raise WikiError("Invalid wiki revision.")
                continue
            limit = properties[key].get("maxLength", 48)
            if (
                not isinstance(value, str)
                or len(value) > limit
                or not value.strip()
                or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff]", value)
            ):
                raise WikiError(f"Invalid wiki {key}.")
        if "page" in args and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args["page"]):
            raise WikiError("Use lowercase letters, digits, hyphens or underscores in page slugs.")
        if name == "wiki_remember" and (
            args["page"] not in FACT_PAGES
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args["key"])
            or len(args["text"].splitlines()) != 1
            or args["text"].strip() != args["text"]
        ):
            raise WikiError("Use a structured page, a key slug and a single trimmed line of text.")

        if name in {"wiki_list", "wiki_search"}:
            rows = self.latest_pages()
            if name == "wiki_search":
                phrase, terms = _query_terms(args["query"])
                ranked = sorted(
                    ((_page_relevance(row, phrase, terms), row) for row in rows),
                    key=lambda hit: hit[0], reverse=True,
                )
                rows = [row for score, row in ranked if score]
            result = [{"page": row["page"], "revision": row["revision"]} for row in rows]
        elif name == "wiki_read":
            result = dict(self.read(**args))
        else:
            revision, changed = self.write(**args)
            result = {"page": args["page"], "revision": revision}
            if name == "wiki_remember":
                result["changed"] = changed
        return json.dumps(result, ensure_ascii=False)

    def latest_pages(self):
        with self.wiki.connect() as db:
            return db.execute("""
                SELECT page, revision, body FROM wiki_revision
                WHERE scope = ? AND member = ? AND (page, revision) IN (
                    SELECT page, MAX(revision) FROM wiki_revision
                    WHERE scope = ? AND member = ? GROUP BY page
                )
                ORDER BY saved_at DESC, page
            """, self.key * 2).fetchall()

    def context(self, query="") -> list[dict]:
        phrase, terms = _query_terms(query)
        rows = sorted(
            self.latest_pages(),
            key=lambda row: (
                -_page_relevance(row, phrase, terms),
                FACT_PAGES.index(row["page"]) if row["page"] in FACT_PAGES else len(FACT_PAGES),
            ),
        )
        result = []
        for row in rows[:4]:
            body = row["body"]
            if len(body) > 240:
                lines = [line.strip() for line in body.splitlines() if line.strip()]
                lines.sort(key=lambda line: _relevance(line, phrase, terms), reverse=True)
                body = "\n".join(lines)
            result.append({"page": row["page"], "excerpt": body[:240]})
        return result

    def read(self, page: str, revision: int | None = None):
        with self.wiki.connect() as db:
            row = self.read_row(db, page, revision)
            if row is None:
                raise WikiError("Wiki page or revision not found.")
            return row

    def read_row(self, db, page: str, revision: int | None = None):
        sql = """
            SELECT page, revision, body, saved_at FROM wiki_revision
            WHERE scope = ? AND member = ? AND page = ?
        """
        args = (*self.key, page)
        if revision is not None:
            return db.execute(sql + " AND revision = ?", (*args, revision)).fetchone()
        return db.execute(sql + " ORDER BY revision DESC LIMIT 1", args).fetchone()

    def write(
        self, page: str, text: str, revision: int | None = None, *, key: str | None = None
    ) -> tuple[int, bool]:
        with self.wiki.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self.read_row(db, page)
            if revision is not None and (current is None or current["revision"] != revision):
                raise WikiError("Wiki revision changed or is missing. Read the latest page first.")
            if current is None:
                pages = db.execute("""
                    SELECT COUNT(DISTINCT page) FROM wiki_revision WHERE scope = ? AND member = ?
                """, self.key).fetchone()[0]
                if pages >= MAX_PAGES:
                    raise WikiError("Member wiki page limit reached.")
            if key is not None:
                body = current["body"] if current is not None else ""
                text = _merge_fact(body, key, text)
                if current is not None and body == text:
                    return current["revision"], False
            elif current is not None and revision is None:
                text = current["body"] + "\n\n" + text
            if len(text) > MAX_CONTENT:
                raise WikiError("Wiki page size limit reached. Rewrite the page to consolidate it.")
            next_revision = current["revision"] + 1 if current else 1
            db.execute("""
                INSERT INTO wiki_revision (scope, member, page, revision, body)
                VALUES (?, ?, ?, ?, ?)
            """, (*self.key, page, next_revision, text))
            db.execute("""
                DELETE FROM wiki_revision
                WHERE scope = ? AND member = ? AND page = ? AND revision <= ?
            """, (*self.key, page, next_revision - MAX_REVISIONS))
        logger.info(
            "Member wiki %s scope=%s member=%s page=%s revision=%d",
            "remember" if key is not None else ("rewrite" if revision is not None else "append"),
            *self.key, page, next_revision,
        )
        return next_revision, True
