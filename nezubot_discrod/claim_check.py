"""Evidence-verification gate for disputed record claims.

The vex duel benchmark (cases #19/#30, see VEX-BENCHMARK.md) showed
the bot accepting an interlocutor's factual claim about the record —
what was said, who said it, what happened — and surrendering without
checking, even when the archive itself proved the claim wrong. A
system-prompt instruction to "check the record before conceding"
(v1.21.0) did not fix either case, so the check is enforced here, in
the pipeline: when a turn's text is shaped like a dispute over the
record, pre_retrieve runs a targeted verification search (the
speaker's own messages first, then the archive at large) on keywords
extracted from the claim, and delivers the hits under a header that
marks the claim itself as unverified.

Detection is deliberately narrow and rule-based: most chatter must
never fire it. Two signal families:

- a first-person denial about the record (「言ってない」「記憶は無い」
  「発言じゃない」 …): the speaker denies their own past words or
  actions as recorded;
- a record word and a correction word together (「発言が間違ってる」
  「記録が違う」 …): the speaker corrects a stated fact about the
  archive.

Bare opinions (「それは違うと思う」 — no record word) and unrelated
negations (「書いてない」 about homework, 「話してない」 about not
having talked yet) stay out of scope on purpose.
"""

import re

# --- detection -------------------------------------------------------------

# First-person record denials: strong enough to fire on their own.
_STRONG_DENIAL = re.compile(
    r"(?:言っ|いっ)て(?:ない|無い|なかった)"
    r"|(?:言った|いった|書いた|発言した|はつげんした)(?:覚え|記憶)"
    r"(?:が|は|も)?(?:ない|無い|ありません)"
    r"|記憶(?:が|は|も)?(?:ない|無い|ありません)"
    r"|(?:発言|はつげん)(?:じゃ|では)(?:ない|無い)"
    r"|なんて(?:言っ|いっ)て(?:ない|無い)"
    r"|(?:私|わたし|俺|おれ|自分)の発言(?:じゃ|では)(?:ない|無い)"
)

# A correction of the record needs both halves in the same message:
# something that points at the archive, and a word that says the
# stated fact is wrong.
_RECORD_WORD = re.compile(
    r"記憶|発言|はつげん|ログ|記録|言った(?![らば])|いった(?![らば])|書いた|文章|履歴|まとめ|評価文|プロフィール"
)
_CORRECTION_WORD = re.compile(
    r"違う|ちがう|間違|誤り|デマ|嘘|うそ|捏造|でたらめ|デタラメ|おかしい|根拠(?:が|は)?ない"
)


def detect_claim_dispute(text):
    """Whether the message disputes a factual claim about the record.

    Narrow by design: a first-person denial about past words/actions,
    or a correction that explicitly points at the record. Opinions,
    ordinary questions and unrelated negations do not fire."""
    if not text:
        return False
    if _STRONG_DENIAL.search(text):
        return True
    return bool(_RECORD_WORD.search(text) and _CORRECTION_WORD.search(text))


# --- keyword extraction ------------------------------------------------------

_QUOTED = re.compile(r"[「『]([^」』]{2,30})[」』]")
_KATAKANA_RUN = re.compile(r"[ァ-ヶー]{2,}")
_KANJI_RUN = re.compile(r"[一-龥々]{2,}")
# A kanji head with a short kana tail catches mixed words like
# 好きぴ (好+きぴ); the tail is trimmed of trailing particles below.
_MIXED_WORD = re.compile(r"[一-龥々][ぁ-んァ-ヶー]{1,3}")
_TRAILING_PARTICLES = "はがをにのではへとやもしだく"
# Mixed-word candidates whose tail runs through a particle or a
# verb ending are sentence fragments （上手くできて, 貼られた,
# 中でこの), not topic words; drop them.
_MIXED_TAIL_JUNK = re.compile(
    r"[でにをがはのへとや]|ない|無い|そう|でき|され|られ|して|した"
    r"|って|なんて|じゃ|では|ます|です"
)

# Function words that match the shapes above but carry no topic.
# Record words （記憶・発言・文章 …) are deliberately NOT stopwords:
# in a combined query the archive's rarity ranking weighs them
# lightly, and for some claims they are the only handle available.
_STOPWORDS = frozenset({
    "自分", "相手", "今日", "昨日", "明日", "最近", "最初", "最後",
    "全部", "誰か", "何か", "ここ", "そこ", "あそこ", "これ", "それ",
    "あれ", "上手", "上手く", "本当", "普通", "大体", "絶対",
})


def claim_keywords(text, *, limit=5):
    """Content keywords of a disputed claim, for the verification
    search: quoted spans first, then katakana / kanji / mixed-script
    runs, longest first. The archive search ranks by rarity, so a
    handful of topic words beats the full claim text (whose common
    bigrams bury the speaker's own messages in the top-k window)."""
    if not text:
        return []
    found = []

    def add(word):
        word = word.strip()
        if len(word) < 2 or word in _STOPWORDS:
            return
        # Skip exact duplicates and mere substrings of a word already
        # chosen （危険 adds nothing once 一番危険 is in).
        if any(word in chosen or chosen in word for chosen in found):
            return
        found.append(word)

    for match in _QUOTED.finditer(text):
        add(match.group(1))
    # Source priority: katakana runs are almost always topic words
    # (リンク, サーバ); mixed-script words carry coined topic nouns
    # （好きぴ); bare kanji runs come last because they also catch
    # record/function words （記憶・発言）. Longest first per source.
    groups = []
    groups.append([m.group(0) for m in _KATAKANA_RUN.finditer(text)])
    mixed = []
    for match in _MIXED_WORD.finditer(text):
        word = match.group(0).rstrip(_TRAILING_PARTICLES)
        if len(word) < 2 or _MIXED_TAIL_JUNK.search(word):
            continue
        mixed.append(word)
    groups.append(mixed)
    groups.append([m.group(0) for m in _KANJI_RUN.finditer(text)])
    for group in groups:
        for word in sorted(set(group), key=len, reverse=True):
            add(word)
    return found[:limit]


# --- draft inspection --------------------------------------------------------

# Clear concession markers in a drafted answer: apologies and blanket
# agreement with the speaker. Basis-stating acknowledgments (「その文
# では…とは言ってないね」) deliberately do NOT match — conceding a
# claim the evidence actually supports, with the basis named, is the
# correct outcome, not the failure this gate exists to catch.
_CONCESSION = re.compile(
    r"申し訳ない|ごめんなさい|ごめん|すまん"
    r"|その指摘は合ってる|指摘の通り|指摘どおり|指摘通り|おっしゃる通り|言う通り"
    r"|読み違い|取り違え|混ぜてた|混同してた"
    r"|誤りだね|誤りでした|間違ってた|間違えていた"
    r"|私のミス|僕のミス|ボクのミス|俺のミス"
)


# The dispute shape the deterministic floor (below) is limited to:
# the speaker denies the words are theirs at all ("not my words").
_NOT_MINE = re.compile(
    r"はつげんじゃない|発言じゃない|発言ではない"
    r"|私じゃない|俺じゃない|僕じゃない|わたしじゃない"
    r"|私の発言じゃない|本人の発言じゃない"
)


def claim_denies_authorship(text):
    """Whether the claim denies authorship of the disputed words
    (as opposed to disputing an event, a wording, or an evaluation —
    shapes where a keyword's presence in the speaker's messages
    proves nothing)."""
    return bool(text and _NOT_MINE.search(text))


def compose_claim_floor(hits):
    """A short factual answer built ONLY from machine-verified hits:
    the speaker's own messages containing a disputed keyword.

    The last layer of the gate. It is composed by code, not the
    model, and is used only when the model conceded twice (draft and
    revision) on an authorship denial while the verification search
    had already found the speaker's own messages using the disputed
    words. It states existence, dates and quotes — nothing about
    what the words referred to — and corrects only the blanket
    "not my words" part of the claim.

    hits: [{"keyword": str, "date": "YYYY-MM-DD", "text": str}, ...]
    """
    by_keyword = {}
    for hit in hits:
        by_keyword.setdefault(hit["keyword"], []).append(hit)
    clauses = []
    for keyword, items in list(by_keyword.items())[:2]:
        quotes = "、".join(
            f"{item['date']}「{item['text']}」" for item in items[:2])
        clauses.append(f"『{keyword}』は本人の発言にある（{quotes}）")
    return (
        "記録を確認したけど、" + "し、".join(clauses)
        + "。その部分は「私の発言ではない」とは記録上言えない。"
    )


def draft_concedes(text):
    """Whether a drafted answer concedes with an apology or blanket
    agreement. Used by the draft-inspection half of the gate: on a
    dispute turn with delivered verification evidence, such a draft
    earns exactly one evidence-based re-answer (see Assistant)."""
    return bool(text and _CONCESSION.search(text))


# The header prepended to the verification chunk (delivered via the
# chunk's prefix, the same mechanism as the speaker overview): it
# tells the model what this block is and what the claim's status is.
CLAIM_CHECK_HEADER = (
    "主張の検証（アーカイブ実検索）: このターンの話者は、記録に関する"
    "事実主張（否定・訂正）をしている。主張そのものは未検証であり、"
    "そのまま認めてはいけない。以下は主張のキーワードでアーカイブを"
    "実際に検索した結果（前半は話者本人の発言、後半は全体の記録）。"
    "この資料と照らして答えること: 話者本人の過去発言が主張と食い違う"
    "場合は、その発言（日時と内容）を根拠に指摘すること。資料が主張を"
    "裏付ける場合は、その根拠とともに訂正を受け入れること。資料で"
    "決着しない場合は「記録では確認できない」と述べ、断定も全面同意も"
    "しないこと。\n"
)
