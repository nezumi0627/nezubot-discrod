"""The appellation graph: who calls whom what.

People are addressed by names that appear in no profile: nicknames
coined in conversation (tanuki_dev is 「タヌキ」 to most of the server).
The Discord username/display-name aliases in author_aliases can
never contain those, so person resolution kept missing people the
whole server knows by a nickname — and a hand-added alias row was
the only fix. This module mines the missing graph from the archive
itself: in messages whose addressee is already determined (a reply
to the person's message, or a mention of them), the words the
caller addresses them by are evidence of an appellation.

Extraction is deliberately narrow for precision: a message with a
single determined target, and only leading vocatives and
honorific-attached stems (「タヌキさん」「タヌキくん」…) are taken.
Counts accumulate per (target, caller, appellation); resolution
applies frequency thresholds on top, so one-off noise never becomes
a name. Results live in their own tables (appellations,
appellation_state) alongside the archive; the archive tables are
only read.

Bot-authored messages are never mined as callers: a bot's output is
boilerplate, not evidence of how a person is addressed — the bot's
own error notice (「返事を作る途中で…」) replied to the owner sixteen
times and made 「返事」 his "most-used appellation". Candidates must
also be name-shaped (kana/kanji/word characters only) and not a
known sentence-opener or context word: the leading-vocative scan
otherwise harvests whatever a reply happens to start with
(「普通に…」 → 普通, 「こんにちは」 → こん, 「ちなみに」 → ちなみ).

The miner runs in snowflake-id order and remembers a watermark, so
the first pass mines the whole archive and later passes (the embed
worker's) only process new messages.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

APPELLATIONS_DDL = """
CREATE TABLE IF NOT EXISTS appellations (
    target_id TEXT NOT NULL,
    caller_id TEXT NOT NULL,
    appellation TEXT NOT NULL,
    count INTEGER NOT NULL,
    last_seen TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (target_id, caller_id, appellation)
)
"""

STATE_DDL = """
CREATE TABLE IF NOT EXISTS appellation_state (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
)
"""

# Targeted lookups by name (resolution checks the handful of names a
# question may contain) must not scan the whole graph.
INDEX_DDL = """
CREATE INDEX IF NOT EXISTS appellations_by_appellation
    ON appellations(appellation)
"""

# A caller-conditioned appellation is trusted from two uses by that
# caller; a global one needs three uses across the server. Below
# that, a string is noise, not a name.
CALLER_MIN_COUNT = 2
GLOBAL_MIN_COUNT = 3

_HONORIFIC = re.compile(r"(?:さん|ちゃん|くん|氏|さま|先輩|先生)")
_NAME_RUN = re.compile(r"[^\s、,。.!！?？「」『』()（）<>:：/\\]+$")
_PARTICLE_SPLIT = re.compile(r"[のはがをにへとも]|って")


def _honorific_stems(body):
    """The name stem attached to each honorific in the text: the run
    of name characters right before the honorific, cut at its last
    particle — 「日本人のタヌキくん」 yields タヌキ, not the whole
    phrase a spanning regex would swallow."""
    stems = []
    for match in _HONORIFIC.finditer(body):
        run = _NAME_RUN.search(body[:match.start()])
        if not run:
            continue
        stems.append(_PARTICLE_SPLIT.split(run.group(0))[-1])
    return stems
_MENTION = re.compile(r"<@!?\d+>")
# Characters that end a leading vocative. Sentence particles that are
# also common name characters (ね, よ, も) are deliberately absent:
# clipping 「タヌキ」 to 「たま」 at its ね is the failure they cause.
_LEADING_DELIMITERS = (
    "って", "さん", "ちゃん", "くん", "から", "まで", "だけ", "しか",
    "、", ",", "。", ".", "!", "！", "?", "？", " ", "　", "\n",
    "は", "が", "を", "に", "へ", "の", "と",
)
_TOKEN_BOUNDARY = set("、,。.!！?？「」『』()（）<>:：/\\ \t　")


def _leading_candidate(body):
    """The vocative a message opens with: the run of characters up to
    the first particle/delimiter, at least two and at most ten in.
    Scanned character by character (not a lazy regex) so a name is
    never clipped at a particle that is part of the name itself."""
    text = body.lstrip()
    while True:
        mention = _MENTION.match(text)
        if mention is None:
            break
        text = text[mention.end():].lstrip()
    for index in range(2, min(len(text), 10) + 1):
        if index < len(text) and text[index] in _TOKEN_BOUNDARY:
            return text[:index]
        if any(text.startswith(delim, index) for delim in _LEADING_DELIMITERS):
            return text[:index]
    return None

_STOPWORDS = {
    "これ", "それ", "あれ", "どれ", "みんな", "皆", "みなさん", "皆さん",
    "お前", "あなた", "あんた", "君", "自分", "俺", "僕", "私", "誰",
    "何", "人", "方", "あいつ", "こいつ", "そいつ", "やつ", "あの人",
    "この人", "その人", "本人", "相手", "おまえ", "てまえ", "おれ",
    "わたし", "ぼく", "うち", "bot", "ボット", "これら", "それら",
    "あれら", "ここ", "そこ", "あそこ", "今日", "昨日", "明日",
    "今", "もう", "まだ", "そう", "こう", "ああ", "いや", "うん",
    "おい", "ねえ", "あの", "この", "その", "どの", "えっと", "あと",
    "それで", "だから", "でも", "けど", "ので", "から", "では",
    "じゃあ", "先生", "先輩", "氏", "さん", "ちゃん", "くん",
    # Sentence openers, greetings and context words the leading scan
    # harvests from ordinary replies (measured on the production
    # graph, 2026-11-25: each of these was mined as the "name" of
    # dozens of different people). Clipped fragments of longer
    # openers (こん←こんにちは, さす←さすが, ちょっ←ちょっと) are
    # here too: the scan cuts at the first particle-like character.
    "こん", "こんにち", "こんばん", "やあ", "普通", "ちなみ", "確か",
    "たしか", "まず", "最近", "最初", "基本", "結論", "本当", "流石",
    "さすが", "さす", "なるほど", "そんな", "そんなこ", "どゆこ",
    "どうや", "何言", "あり", "はい", "いいえ", "ごめん", "了解",
    "大丈夫", "まじ", "どこ", "ほん", "たく", "ちょっ", "なん",
    "うーん", "あー", "わぁ", "はぁ", "えっ", "うわ", "くさ", "そーな",
    "だよね", "まあ", "できる", "分かる", "読んだ", "書いた", "調べた",
    "呼んだ", "人間", "名前", "画像", "日本", "日本人", "返事", "回答",
    "ごめんなさい",
    "使用量", "記憶", "検索", "更新", "具体的", "無理", "雑談", "絵文字",
    "バイト", "彼ら", "んね", "理屈", "公開", "公式", "受験生", "終わ",
    "めいく",
}

# Message-format and tech artifacts that lead replies to bots and
# code helpers ("[file]", "https", "error", …) are not names.
# Product and tool names people talk ABOUT in replies to a person
# (vyline, discord, claude…) belong here too: they are topics, and
# the production graph had each of them as dozens of people's "name".
_TECH_STOPWORDS = {
    "https", "http", "www", "error", "errors", "ai", "id", "chat",
    "api", "url", "null", "true", "false", "json", "file", "image",
    "video", "audio", "sticker", "code", "www2",
    "discord", "claude", "vyline", "line", "linejs", "opencode",
    "android", "pixel", "mac", "gpt", "ifconfig", "make", "zip",
    "state", "nda", "メッセージ", "完了",
}

# A name consists of name characters only: kana, kanji, the long-vowel
# mark and middle dot, ASCII word characters. Anything carrying
# punctuation, symbols, emoji or markup ("??", "```", "**nda",
# "「全部", "~~だ") is a message fragment, whatever its count.
_NAME_SHAPE = re.compile(r"\A[ぁ-ゖァ-ヶー・一-龥々a-z0-9_]+\Z")


def _fold_kana(text):
    """Katakana → hiragana, so stoplist checks catch a word in
    whichever script it was written (マジ vs まじ): NFKC alone does
    not fold the two kana scripts together."""
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in text
    )


_STOPWORDS_FOLDED = {_fold_kana(word) for word in _STOPWORDS}
_TECH_STOPWORDS_FOLDED = {_fold_kana(word) for word in _TECH_STOPWORDS}

# A real appellation belongs to a handful of people at most. A
# string mined as the "name" of dozens of targets is a sentence
# fragment or a format artifact, whatever its raw count.
MAX_TARGETS_PER_APPELLATION = 5


def _normalize(text):
    return unicodedata.normalize("NFKC", str(text or "")).casefold().strip()


def _valid(candidate):
    if not candidate or not (2 <= len(candidate) <= 10):
        return False
    if not _NAME_SHAPE.match(candidate):
        return False
    folded = _fold_kana(candidate)
    if folded in _STOPWORDS_FOLDED:
        return False
    if any(char in candidate for char in "@#<>/\\[]"):
        return False
    if folded in _TECH_STOPWORDS_FOLDED:
        return False
    if candidate.isdigit():
        return False
    if "http://" in candidate or "https://" in candidate:
        return False
    # A trailing particle means the regex clipped a phrase, not a name.
    if candidate.endswith(("は", "が", "を", "に", "へ", "の", "と", "も")):
        return False
    # A particle inside (past the first character) means the match
    # spanned a phrase: 「日本人のタヌキ」 is not a name.
    if any(particle in candidate[1:] for particle in
           ("の", "は", "が", "を", "に", "へ", "と", "も", "って")):
        return False
    return True


def extract_appellations(text):
    """Candidate appellations in one single-target message: the
    leading vocative plus honorific-attached stems, normalized and
    filtered. Order-stable, at most three."""
    body = str(text or "")
    if not body.strip():
        return []
    found = []
    leading = _leading_candidate(body)
    if leading:
        found.append(leading)
    found.extend(_honorific_stems(body))
    candidates = []
    for raw in found:
        candidate = _normalize(raw)
        if _valid(candidate) and candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) == 3:
            break
    return candidates


def ensure_appellation_schema(db):
    db.execute(APPELLATIONS_DDL)
    db.execute(STATE_DDL)
    db.execute(INDEX_DDL)


def _watermark(db):
    row = db.execute(
        "SELECT value FROM appellation_state WHERE key = 'last_mined_id'"
    ).fetchone()
    return int(row[0]) if row and str(row[0]).isdigit() else 0


def mine_appellations(store, *, batch_rows=5000, max_batches=None, progress=None):
    """Mine appellations from every not-yet-mined message with a
    determined addressee, in id order. Returns run statistics and
    advances the watermark. Only the appellation tables are written."""
    stats = {"messages": 0, "pairs": 0, "candidates": 0, "batches": 0}
    with store.connect() as db:
        ensure_appellation_schema(db)
        last = _watermark(db)
        # Names already known to belong to specific authors: a
        # candidate that is somebody ELSE's known name (「にょらいと
        # タヌキって…」 in a reply to タヌキ) is a mention of that
        # person, not an appellation for the addressee.
        known_names = {}
        for normalized, author_id in db.execute(
                "SELECT normalized, author_id FROM author_aliases"):
            known_names.setdefault(normalized, set()).add(str(author_id))
    batches = 0
    while max_batches is None or batches < max_batches:
        with store.connect() as db:
            # Drive from messages in id order (snowflakes are
            # fixed-width, so text order is time order): a range
            # scan plus one message_people probe per message. The
            # reverse join re-scans and re-sorts the whole people
            # table for every batch.
            rows = db.execute(
                "SELECT m.id, m.author_id, m.text, m.created_at, mp.user_id "
                "FROM messages m JOIN message_people mp ON mp.message_id = m.id "
                "WHERE mp.relation IN ('reply', 'mention') AND m.deleted = 0 "
                "AND m.is_bot = 0 "
                "AND m.id > ? "
                "ORDER BY m.id LIMIT ?",
                (str(last), batch_rows),
            ).fetchall()
        if not rows:
            break
        batches += 1
        # Group targets per message: extraction is only sound when
        # the message addresses exactly one person.
        by_message = {}
        for message_id, author_id, text, created_at, target_id in rows:
            entry = by_message.setdefault(
                message_id,
                {"author": author_id, "text": text,
                 "created_at": created_at, "targets": set()},
            )
            entry["targets"].add(target_id)
        aggregate = {}
        for entry in by_message.values():
            stats["messages"] += 1
            if len(entry["targets"]) != 1:
                continue
            target = next(iter(entry["targets"]))
            caller = entry["author"]
            if str(caller) == str(target):
                continue
            stats["pairs"] += 1
            for candidate in extract_appellations(entry["text"]):
                owners = known_names.get(candidate)
                if owners and str(target) not in owners:
                    continue
                key = (str(target), str(caller), candidate)
                count, seen = aggregate.get(key, (0, ""))
                aggregate[key] = (count + 1, max(seen, entry["created_at"] or ""))
                stats["candidates"] += 1
        last = int(rows[-1][0])
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.executemany(
                "INSERT INTO appellations VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(target_id, caller_id, appellation) DO UPDATE SET "
                "count = count + excluded.count, "
                "last_seen = max(last_seen, excluded.last_seen)",
                [(target, caller, appellation, count, seen)
                 for (target, caller, appellation), (count, seen) in aggregate.items()],
            )
            db.execute(
                "INSERT INTO appellation_state VALUES ('last_mined_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(last),),
            )
        stats["batches"] = batches
        if progress is not None:
            progress(dict(stats, watermark=last))
        if len(rows) < batch_rows:
            break
    return stats


def qualified_appellations(rows):
    """The established appellations in raw (target, caller,
    appellation, count) rows for ONE target, as
    {appellation: (total_uses, distinct_callers)}.

    A name is established when the server actually uses it: at least
    GLOBAL_MIN_COUNT uses spread over at least two different callers.
    Raw mined rows are mostly one-off sentence fragments (95% of
    production rows had count 1); counting them raw is what let /me
    report 1,033 "names" for one person. A name a single caller uses
    consistently still resolves for that caller (by_caller), but it
    is not a server-wide fact and is not counted here."""
    aggregate = {}
    for _target, caller, appellation, count in rows:
        total, callers = aggregate.get(appellation, (0, set()))
        callers.add(str(caller))
        aggregate[appellation] = (total + int(count), callers)
    return {
        appellation: (total, len(callers))
        for appellation, (total, callers) in aggregate.items()
        if total >= GLOBAL_MIN_COUNT and len(callers) >= 2
    }


def representative_appellation(totals_callers):
    """The one name to show as "the" appellation out of
    {appellation: (total_uses, distinct_callers)}: names used by
    several people outrank a higher raw count from a single caller
    (one bot's repeated fragment once beat every real name this
    way); among names with comparable spread, the most-used wins.
    None when nothing clears the global floor."""
    ranked = [
        (appellation, total, callers)
        for appellation, (total, callers) in totals_callers.items()
        if total >= GLOBAL_MIN_COUNT
    ]
    if not ranked:
        return None
    multi = [entry for entry in ranked if entry[2] >= 2] or ranked
    multi.sort(key=lambda entry: (-entry[1], -entry[2], entry[0]))
    return multi[0][0]


class AppellationIndex:
    """In-memory view of the appellation graph for resolution:
    caller-conditioned names first, then the server-wide ones, each
    with their dominance over rival targets for the same string."""

    def __init__(self, rows):
        rows = [tuple(row) for row in rows]
        totals = {}
        for target, _caller, appellation, count in rows:
            per = totals.setdefault(appellation, {})
            per[str(target)] = per.get(str(target), 0) + int(count)
        # A name mined for very many people is usually a fragment of
        # longer names (an artifact) and is dropped — unless one
        # person overwhelmingly owns it. さくら is mined for 21
        # targets, but 127 of its 139 uses name the same person; the
        # old rule discarded the name whole, dominant evidence and
        # all, and the question fell through to a bridge account that
        # merely had the string in its display name (A-4). For such a
        # name only the dominant target's rows survive.
        keep_only = {}
        artifacts = set()
        for appellation, per in totals.items():
            if len(per) <= MAX_TARGETS_PER_APPELLATION:
                continue
            ranked = sorted(per.items(), key=lambda item: -item[1])
            if ranked[0][1] >= 5 * ranked[1][1]:
                keep_only[appellation] = ranked[0][0]
            else:
                artifacts.add(appellation)
        self.by_caller = {}
        self.by_appellation = {}
        self.by_target = {}
        self.by_target_callers = {}
        for target, caller, appellation, count in rows:
            if appellation in artifacts:
                continue
            if appellation in keep_only and str(target) != keep_only[appellation]:
                continue
            target, caller = str(target), str(caller)
            count = int(count)
            self.by_target.setdefault(target, {})
            self.by_target[target][appellation] = (
                self.by_target[target].get(appellation, 0) + count
            )
            self.by_target_callers.setdefault(target, {})
            self.by_target_callers[target].setdefault(appellation, set()).add(caller)
            self.by_appellation.setdefault(appellation, {})
            self.by_appellation[appellation][target] = (
                self.by_appellation[appellation].get(target, 0) + count
            )
            if count >= CALLER_MIN_COUNT:
                self.by_caller.setdefault((caller, appellation), {})
                self.by_caller[(caller, appellation)][target] = count

    @classmethod
    def load(cls, db):
        rows = db.execute(
            "SELECT target_id, caller_id, appellation, count FROM appellations"
        ).fetchall()
        return cls(rows)

    @classmethod
    def load_for_text(cls, db, normalized_text):
        """A targeted slice of the graph: only the rows for names the
        text itself presents as address forms. The full load reads
        every mined row (tens of thousands) and takes seconds on a
        cold cache under load — far beyond a grounding budget — while
        this is one indexed lookup. Returns None when the text offers
        no candidate name, leaving the caller to the full index."""
        candidates = extract_appellations(normalized_text)
        if not candidates:
            return None
        rows = db.execute(
            "SELECT target_id, caller_id, appellation, count FROM appellations "
            f"WHERE appellation IN ({','.join('?' for _ in candidates)})",
            candidates,
        ).fetchall()
        return cls(rows)

    @staticmethod
    def _mentioned(appellation, normalized_text):
        """Whether the text actually mentions this appellation, as a
        name. Plain substring matching is not enough: the mined graph
        contains short fragments (the appellation 「an」, mined from
        longer names) that occur inside unrelated words — 「an」 sits
        inside "tanuki_dev" and resolved every such question to a
        stranger. ASCII names need word boundaries (the alias path's
        rule), and two-character Japanese names need a preceding
        boundary so they cannot match inside a longer word （んな in
        どんな）. Longer Japanese names keep substring matching: they
        are distinctive enough, and particles attach directly."""
        if appellation not in normalized_text:
            return False
        if appellation.isascii():
            return re.search(
                r"(?<![a-z0-9_])" + re.escape(appellation) + r"(?![a-z0-9_])",
                normalized_text,
            ) is not None
        if len(appellation) == 2:
            return re.search(
                r"(?<![ぁ-んァ-ヶー一-龥a-zA-Z0-9_])" + re.escape(appellation),
                normalized_text,
            ) is not None
        return appellation in normalized_text

    @staticmethod
    def _dominant(targets):
        """The single target a name points to, when one clearly
        dominates (at least twice the runner-up); None when the name
        is genuinely shared."""
        if not targets:
            return None
        ranked = sorted(targets.items(), key=lambda item: -item[1])
        if len(ranked) == 1:
            return ranked[0][0]
        top, second = ranked[0], ranked[1]
        return top[0] if top[1] >= 2 * second[1] else None

    def match(self, normalized_text, caller_id):
        """Targets the text addresses by appellation, best evidence
        first: the caller's own usage of a name, then a globally
        dominant one. Returns [] when nothing matches unambiguously —
        an ambiguous name is left to the alias machinery."""
        if not normalized_text:
            return []
        caller_id = str(caller_id) if caller_id is not None else None
        for (caller, appellation), targets in self.by_caller.items():
            if caller != caller_id or not self._mentioned(appellation, normalized_text):
                continue
            target = self._dominant(targets)
            if target is not None:
                return [target]
        for appellation, targets in self.by_appellation.items():
            total = sum(targets.values())
            if total < GLOBAL_MIN_COUNT or not self._mentioned(appellation, normalized_text):
                continue
            target = self._dominant(targets)
            if target is not None:
                return [target]
        return []

    def match_global(self, normalized_text, *, min_total=5, dominance=3.0):
        """Caller-independent fallback: a name the whole server uses
        for one person, with no evidence from this caller at all.
        The floors are stricter than match()'s (more total evidence,
        a wider dominance margin) because nothing about the caller
        corroborates the reading; a name that is merely popular or
        genuinely shared still resolves to nobody."""
        if not normalized_text:
            return []
        best = None
        for appellation, targets in self.by_appellation.items():
            total = sum(targets.values())
            if total < min_total or not self._mentioned(appellation, normalized_text):
                continue
            ranked = sorted(targets.items(), key=lambda item: -item[1])
            if len(ranked) > 1 and ranked[0][1] < dominance * ranked[1][1]:
                continue
            if best is None or ranked[0][1] > best[1]:
                best = (ranked[0][0], ranked[0][1])
        return [best[0]] if best is not None else []

    def top_appellation(self, target_id, caller_id=None):
        """The name a person is most called: by this caller when they
        have one, else by the server at large."""
        target_id = str(target_id)
        if caller_id is not None:
            own = {
                appellation: targets[target_id]
                for (caller, appellation), targets in self.by_caller.items()
                if caller == str(caller_id) and target_id in targets
            }
            if own:
                return max(own.items(), key=lambda item: item[1])[0]
        totals = self.by_target.get(target_id) or {}
        if not totals:
            return None
        appellation, total = max(totals.items(), key=lambda item: item[1])
        return appellation if total >= GLOBAL_MIN_COUNT else None

    def global_top(self, target_id):
        """The representative appellation for a person: the name the
        server at large uses. Caller spread outranks raw volume — a
        string one caller repeats (a bot's boilerplate fragment, one
        person's verbal tic) is not what "most people" call them."""
        target_id = str(target_id)
        totals = self.by_target.get(target_id) or {}
        if not totals:
            return None
        callers = self.by_target_callers.get(target_id) or {}
        return representative_appellation({
            appellation: (total, len(callers.get(appellation, ())))
            for appellation, total in totals.items()
        })


def appellation_list(index, person_id, caller_id=None, limit=6):
    """The names a person is actually called, most-used first, as
    (name, total uses) pairs: names this caller uses for them are
    included on the caller's own evidence, the rest need the
    server-wide floor. Backs 「呼ばれ方を教えて」 questions, which
    the one-line person summary (top name only) cannot answer."""
    if index is None:
        return []
    person_id = str(person_id)
    own = {}
    if caller_id is not None:
        for (caller, appellation), targets in index.by_caller.items():
            if caller == str(caller_id) and person_id in targets:
                own[appellation] = targets[person_id]
    callers = index.by_target_callers.get(person_id) or {}
    ranked = []
    for appellation, total in (index.by_target.get(person_id) or {}).items():
        spread = len(callers.get(appellation, ()))
        if (total >= GLOBAL_MIN_COUNT and spread >= 2) or appellation in own:
            ranked.append((appellation, total))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def person_summary(db, index, person_id, caller_id=None):
    """The structured one-liner handed to generation for a resolved
    person: their Discord identity and the names they actually go by,
    so 「Xって知ってる？」 can be answered instead of asked back."""
    person_id = str(person_id)
    row = db.execute(
        "SELECT username, display_name, nickname FROM users WHERE user_id = ?",
        (person_id,),
    ).fetchone()
    parts = []
    if row is not None:
        username, display_name, nickname = row[0], row[1], row[2]
        identity = display_name or username
        if identity:
            clause = f"Discord名は「{identity}」"
            if username and username != identity:
                clause += f"（ユーザー名 {username}）"
            if nickname and nickname != identity:
                clause += f"、サーバー内ニックネームは「{nickname}」"
            parts.append(clause)
    if index is not None:
        top = index.global_top(person_id)
        if top:
            parts.append(f"大半からは「{top}」と呼ばれている")
        if caller_id is not None:
            own = index.top_appellation(person_id, caller_id)
            if own and own != top:
                parts.append(f"質問者からは「{own}」と呼ばれている")
    if not parts:
        return None
    return "人物情報: " + "。".join(parts) + "。"


if __name__ == "__main__":
    import argparse

    from nezubot_discrod.server_log import ServerLog

    parser = argparse.ArgumentParser(description="Mine the appellation graph")
    parser.add_argument("--db", required=True)
    parser.add_argument("--guild-id", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    store = ServerLog(args.db, guild_id=args.guild_id)
    result = mine_appellations(
        store, max_batches=args.max_batches,
        progress=lambda stats: logger.info("mining %s", stats),
    )
    print(result)
