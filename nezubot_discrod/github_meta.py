"""Deterministic GitHub metadata for answering turns.

Authorship questions (「XのGitHub出して」「XはYを作ったの？」) used
to be answered from whatever the archive search surfaced: a message
that merely pastes a repo URL reads like authorship, and the holder
of a fork reads like the maker. On 2026-11-27 the bot credited
tanuki_dev with Vyline — tanuki_dev/vyline is a fork of testuser0627/vyline,
a fact one GitHub API call settles, and no amount of log reading
supplies it. This module fetches that metadata (owner, fork flag,
parent/source) from the GitHub API, caches it in the archive DB so
repeat questions are answered from local rows inside the retrieval
budgets, and renders it as a deterministic block in the same style
as the activity blocks: facts the model quotes, not material it
interprets. Fetch failures degrade to no block (or a partially
marked block), never to invented metadata.
"""

import json
import logging
import os
import re
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# github.com/<owner>[/<repo>...] in message text. The repo segment
# stops at the next path separator, so /issues, /releases/tag/...
# suffixes fold into the repo itself; a trailing .git is stripped.
_REF = re.compile(
    r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)"
    r"(?:/([A-Za-z0-9_.-]+))?"
)
# First path segments that are GitHub site pages, not accounts.
_RESERVED_OWNERS = {
    "about", "collections", "contact", "customer-stories", "enterprise",
    "events", "explore", "features", "home", "join", "login", "logout",
    "marketplace", "new", "notifications", "organizations", "orgs",
    "pricing", "readme", "search", "security", "settings", "signup",
    "site", "sponsors", "team", "topics", "trending", "users", "watch",
}
# Second segments that are site pages under an account, not repos.
_RESERVED_REPOS = {"followers", "following", "packages", "repositories",
                   "sponsorship", "stars"}

_REPO_TTL_SECONDS = 7 * 24 * 3600
_MISSING_TTL_SECONDS = 24 * 3600
_API_TIMEOUT_SECONDS = 4.0
_MAX_REPOS_PER_BLOCK = 3
_PERSON_MESSAGE_SCAN = 300

_BLOCK_HEADER = (
    "GitHub情報（確定情報）: 以下は GitHub API のメタデータ（repo の所有者と "
    "fork 関係）を機械的に取得したもの。repo の製作者はその所有者であり、"
    "fork の場合は fork 元（source）の所有者が製作者で、fork を保持して"
    "いる人は製作者ではない。また、repo の URL を発言に貼っただけでは"
    "その repo の製作者にはならない。"
)


def extract_github_refs(text):
    """(owner, repo-or-None) pairs a text links to, in first-seen
    order. Owners are casefolded (GitHub logins are case-insensitive
    and the archive mixes cases); repo names keep their case for
    display but compare casefolded."""
    refs = []
    seen = set()
    for match in _REF.finditer(text or ""):
        owner = match.group(1)
        if owner.casefold() in _RESERVED_OWNERS:
            continue
        repo = match.group(2)
        if repo is not None:
            repo = repo.removesuffix(".git")
            if not repo or repo.casefold() in _RESERVED_REPOS:
                repo = None
        key = (owner.casefold(), repo.casefold() if repo else None)
        if key in seen:
            continue
        seen.add(key)
        refs.append((owner, repo))
    return refs


def _api_get(path):
    """One GitHub API GET as (state, payload): state is "ok",
    "missing" (404 — private, renamed or deleted) or "error"
    (network, rate limit, anything else). The token, when configured,
    rides along; the proxy comes from the process environment, the
    same egress the rest of the bot uses."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request("https://api.github.com" + path, headers=headers)
    try:
        with urlopen(request, timeout=_API_TIMEOUT_SECONDS) as response:
            return "ok", json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return "missing", None
        return "error", None
    except (URLError, OSError, ValueError):
        return "error", None


def fetch_repo(owner, repo):
    """Live metadata for one repo as (state, facts). Facts are the
    normalized subset the block renders: the owner's login, the fork
    flag, and the parent/source full names for a fork. Kept as a
    module-level function so tests can replace the network."""
    state, payload = _api_get(f"/repos/{owner}/{repo}")
    if state != "ok" or not isinstance(payload, dict):
        return state, None
    parent = payload.get("parent") or {}
    source = payload.get("source") or {}
    return "ok", {
        "full_name": payload.get("full_name") or f"{owner}/{repo}",
        "owner": (payload.get("owner") or {}).get("login") or owner,
        "fork": bool(payload.get("fork")),
        "parent": parent.get("full_name"),
        "source": source.get("full_name") or parent.get("full_name"),
        "description": payload.get("description") or "",
    }


def _ensure_cache(db):
    db.execute(
        "CREATE TABLE IF NOT EXISTS github_meta_cache ("
        "key TEXT PRIMARY KEY, payload TEXT, fetched_at REAL NOT NULL)"
    )


def repo_meta(store, owner, repo):
    """(state, facts) for one repo, cache-first.

    Successful lookups cache for a week (ownership and fork relations
    change rarely), 404s for a day (a repo can go public or be
    renamed), and errors not at all — a transient failure must not
    fossilize into a week of 「未確認」. The cache lives in the
    archive DB, so it survives restarts and is shared by every
    retrieval thread."""
    key = f"{owner}/{repo}".casefold()
    now = time.time()
    try:
        with store.connect() as db:
            _ensure_cache(db)
            row = db.execute(
                "SELECT payload, fetched_at FROM github_meta_cache "
                "WHERE key = ?", (key,),
            ).fetchone()
    except sqlite3.Error:
        row = None
    if row is not None:
        payload, fetched_at = row[0], row[1]
        if payload is not None and now - fetched_at < _REPO_TTL_SECONDS:
            try:
                return "ok", json.loads(payload)
            except ValueError:
                pass
        if payload is None and now - fetched_at < _MISSING_TTL_SECONDS:
            return "missing", None
    state, facts = fetch_repo(owner, repo)
    if state in ("ok", "missing"):
        try:
            with store.connect() as db:
                _ensure_cache(db)
                db.execute(
                    "INSERT OR REPLACE INTO github_meta_cache "
                    "(key, payload, fetched_at) VALUES (?, ?, ?)",
                    (key, json.dumps(facts) if facts else None, now),
                )
        except sqlite3.Error:
            logger.warning("GitHub metadata cache write failed", exc_info=True)
    return state, facts


def _repo_line(store, owner, repo, account):
    """One repo's rendered fact line. `account` is the GitHub login
    attributed to the person the block is about (None for a bare
    text-URL block): it only changes the wording of whose repo it is,
    never the metadata itself."""
    state, facts = repo_meta(store, owner, repo)
    full = f"{owner}/{repo}"
    if state == "missing":
        return (f"- repo {full}: GitHub API では確認できなかった"
                "（非公開・改名・削除のいずれか）。製作者は不明として"
                "扱うこと。")
    if state != "ok" or facts is None:
        return (f"- repo {full}: メタデータを取得できなかった。"
                "製作者は未確認として扱うこと。")
    holder = facts["owner"]
    if facts["fork"]:
        source = facts["source"] or facts["parent"] or "不明"
        maker = source.split("/")[0] if "/" in source else source
        line = (f"- repo {facts['full_name']}: 所有者は {holder}。"
                f"ただし fork であり、元の repo は {source}"
                f"（製作者は {maker}）。{holder} は fork の保持者で"
                "あり、この repo の製作者ではない。")
        return line
    if account is not None and holder.casefold() == account.casefold():
        return (f"- repo {facts['full_name']}: 所有者はこの人物"
                f"（{holder}）。fork ではない。")
    if account is not None:
        return (f"- repo {facts['full_name']}: 所有者は {holder}。"
                "この人物が発言に貼った repo だが、製作者は "
                f"{holder} であってこの人物ではない。")
    return (f"- repo {facts['full_name']}: 所有者は {holder}。"
            "fork ではない。")


def _render(lines):
    if not lines:
        return None
    return "\n".join([_BLOCK_HEADER, *lines])


def person_github_block(store, channel_ids, person):
    """The deterministic GitHub block for one resolved person: the
    GitHub account their own messages point to, plus metadata for
    the repos those messages link to most (their own repos first).

    The account is the owner segment appearing in at least two of
    the person's own repo links — a single pasted URL is exactly the
    evidence this block exists to distrust. Repos are still listed
    when no account clears that bar; their lines then carry no
    person-relative wording."""
    person = str(person)
    with store.connect() as db:
        where, args = store._scope(channel_ids, author=(person,))
        rows = db.execute(
            f"SELECT m.text FROM messages m WHERE {where} "
            "AND m.text LIKE '%github.com%' "
            "ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
            [*args, _PERSON_MESSAGE_SCAN],
        ).fetchall()
    owner_counts = {}
    repo_counts = {}
    for row in rows:
        for owner, repo in extract_github_refs(row[0]):
            if repo is None:
                owner_counts[owner.casefold()] = \
                    owner_counts.get(owner.casefold(), 0) + 1
                continue
            key = (owner.casefold(), repo.casefold())
            entry = repo_counts.setdefault(key, [owner, repo, 0])
            entry[2] += 1
            owner_counts[owner.casefold()] = \
                owner_counts.get(owner.casefold(), 0) + 1
    if not repo_counts and not owner_counts:
        return None
    account = None
    if owner_counts:
        top = max(owner_counts.items(), key=lambda item: item[1])
        if top[1] >= 2:
            account = top[0]
    ranked = sorted(
        repo_counts.values(),
        key=lambda entry: (
            0 if account and entry[0].casefold() == account else 1,
            -entry[2],
            entry[0].casefold(), entry[1].casefold(),
        ),
    )
    lines = []
    if account is not None:
        lines.append(
            f"- この人物の GitHub アカウント: github.com/{account}"
            f"（本人の発言に {owner_counts[account]} 回登場）")
    for owner, repo, _count in ranked[:_MAX_REPOS_PER_BLOCK]:
        lines.append(_repo_line(store, owner, repo, account))
    return _render(lines)


def text_github_block(store, text):
    """The deterministic GitHub block for repo URLs the question
    itself carries (no resolved person): metadata for up to two
    repos, so 「この repo 誰の？」 is answered from ownership and
    fork source instead of from whoever pasted the link."""
    repos = [(owner, repo) for owner, repo in extract_github_refs(text)
             if repo is not None]
    if not repos:
        return None
    lines = [_repo_line(store, owner, repo, None)
             for owner, repo in repos[:2]]
    return _render(lines)
