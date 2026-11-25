"""Local proxy eval for conversational Japanese RAG (NOT the original gold set).

The original 120-question gold set from docs/rag-benchmark.md is not in the
repo, so this builds a deterministic synthetic corpus in the same spirit:
gold messages with distinctive content + generic distractor chatter, and
queries split into keyword / paraphrase / who / temporal categories.
Dense side uses a character-bigram hashing embedder as a local proxy for E5.
Metrics follow the benchmark doc: R@k = fraction of queries with any gold
id in top k.

Usage: .venv/bin/python hidden-benchmark/eval_rag.py [--json out.json]
"""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nezubot_discrod.server_log import ServerLog, bigrams  # noqa: E402
from nezubot_discrod.server_log_context import resolve_people  # noqa: E402

DIMS = 128


class BigramEmbedder:
    """Deterministic bag-of-bigrams hashing embedder (local dense proxy)."""

    def embed(self, texts):
        for text in texts:
            vec = [0.0] * DIMS
            for token in bigrams(text):
                digest = hashlib.md5(token.encode()).digest()
                vec[int.from_bytes(digest[:2], "little") % DIMS] += 1.0
                vec[int.from_bytes(digest[2:4], "little") % DIMS] += 0.5
            yield vec


AUTHORS = {
    "alice": {"id": "20", "username": "alice", "global_name": "Alice", "nick": "ありす"},
    "bob": {"id": "21", "username": "bob", "global_name": "Bob"},
    "carol": {"id": "22", "username": "carol", "global_name": "Carol", "nick": "きゃろ"},
    "dave": {"id": "23", "username": "dave", "global_name": "Dave"},
    "vex": {"id": "30", "username": "vexbot", "global_name": "Vexちゃん【公式bot】",
            "bot": True},
}

GOLD = [
    ("kb", "alice", "2026-11-10", "自作キーボードの静電容量スイッチを基板から設計し直した"),
    ("rust", "bob", "2026-10-02", "Rustで書いた検索エンジンの転置インデックスがやっと動いた"),
    ("cat", "carol", "2026-09-15", "保護猫のミケを動物病院に連れて行ったら腎臓の数値が改善してた"),
    ("camp", "dave", "2025-11-20", "去年は阿蘇でキャンプして星空を見ながら焚き火をした"),
    ("piano", "alice", "2026-08-01", "ピアノの発表会でショパンの夜想曲を暗譜で弾き切った"),
    ("ramen", "bob", "2026-11-01", "二郎系ラーメンの乳化スープを家で再現するのに三回失敗した"),
    ("visa", "carol", "2025-06-10", "台湾旅行のビザ申請で指紋登録のために領事館まで行った"),
    ("go", "dave", "2026-05-05", "囲碁の定石を覚えるために詰碁を毎日30問解いている"),
    ("plant", "alice", "2026-04-18", "ベランダのバジルがアブラムシにやられて全滅した"),
    ("marathon", "bob", "2025-12-01", "福岡マラソンで自己ベストを4分更新して完走した"),
]

DISTRACTORS = [
    "今日の進捗です、テストを通しました", "そのバグ明日見ます", "会議の時間変更になったらしい",
    "おつかれさまです", "それいいですね", "週末なにしてた？", "進捗どうですか",
    "ラーメン食べたい", "猫かわいい", "旅行行きたいな", "キーボード新しいの買った",
    "ピアノ練習してる", "キャンプ道具そろえたい", "最近忙しくて全然寝てない",
    "その件は前に話した気がする", "了解です", "あとで確認します", "すごいじゃん",
    "マラソン走る人尊敬する", "植物育てるの難しいよね",
]


def record(number, author_key, date, text, channel="10"):
    return {
        "id": str(number), "guild_id": "100", "channel_id": channel,
        "_channel": "general" if channel == "10" else "private", "_parent": "dev",
        "author": AUTHORS[author_key], "timestamp": f"{date}T12:00:00+00:00",
        "content": text,
    }


def build_store(root):
    store = ServerLog(root / "server_log.db", guild_id="100",
                      model="intfloat/multilingual-e5-large", embedder=BigramEmbedder())
    rows = []
    gold_ids = {}
    n = 1000
    for key, author, date, text in GOLD:
        n += 1
        gold_ids[key] = str(n)
        rows.append(record(n, author, date, text, channel="10" if n % 2 else "11"))
    authors = list(AUTHORS)
    for i in range(500):
        n += 1
        rows.append(record(n, authors[i % len(authors)], "2026-0%d-15" % (1 + i % 9),
                           DISTRACTORS[i % len(DISTRACTORS)] + f" ({i})",
                           channel="10" if i % 3 else "11"))
    store.upsert_many(rows)
    store.embed_pending()
    return store, gold_ids


QUERIES = [
    ("keyword", "静電容量スイッチ", "kb"),
    ("keyword", "転置インデックス", "rust"),
    ("keyword", "腎臓の数値", "cat"),
    ("keyword", "ショパンの夜想曲", "piano"),
    ("keyword", "乳化スープ", "ramen"),
    ("keyword", "指紋登録 領事館", "visa"),
    ("keyword", "詰碁", "go"),
    ("keyword", "アブラムシ 全滅", "plant"),
    ("paraphrase", "キーボードのスイッチを自分で設計した人いたよね", "kb"),
    ("paraphrase", "検索の仕組みをRustで作った話", "rust"),
    ("paraphrase", "猫を病院に連れて行った時の話", "cat"),
    ("paraphrase", "ピアノの発表会で暗譜した話", "piano"),
    ("paraphrase", "家で二郎系を作ろうとして失敗した話", "ramen"),
    ("paraphrase", "台湾に行くための手続きで領事館に行った話", "visa"),
    ("paraphrase", "ベランダのハーブが虫にやられた話", "plant"),
    ("paraphrase", "マラソンで自己ベスト出した話", "marathon"),
    ("who", "静電容量スイッチって誰が言ってた？", "kb"),
    ("who", "bobが前に言ってた検索エンジンの話", "rust"),
    ("who", "保護猫を病院に連れて行ったの誰だっけ", "cat"),
    ("who", "aliceが前に言ってたピアノの発表会の話", "piano"),
    ("who", "二郎系ラーメンを家で再現したって誰が言ってた？", "ramen"),
    ("who", "囲碁の詰碁を毎日解いてるの誰？", "go"),
    ("temporal", "去年キャンプで星空を見たって誰が言ってた？", "camp"),
    ("temporal", "去年マラソンで自己ベスト出した話", "marathon"),
    ("temporal", "最近ラーメンの再現に失敗した話", "ramen"),
    ("temporal", "最近キーボードのスイッチ設計した話", "kb"),
]

RESOLUTION = [
    ("aliceについて教えて", ["20"]),
    ("ありすについて", ["20"]),
    ("bobさんってどんな人", ["21"]),
    ("きゃろについて教えて", ["22"]),
    ("Vexってどう思う？", ["30"]),
    ("vexについて", ["30"]),
    ("vexbotについて", ["30"]),
    ("ＡＬＩＣＥについて", ["20"]),
    ("maliceについて", []),
    ("誰か知らない人について", []),
]


def metrics(store, gold_ids):
    per_category = {}
    latencies = {}
    for category, query, key in QUERIES:
        gold = gold_ids[key]
        store.search(query, channel_ids=[10, 11], limit=10)
        times, hits = [], []
        for _ in range(3):
            start = time.perf_counter()
            hits = store.search(query, channel_ids=[10, 11], limit=10)
            times.append((time.perf_counter() - start) * 1000)
        rank = next((i for i, hit in enumerate(hits, 1) if hit.id == gold), None)
        per_category.setdefault(category, []).append(rank)
        latencies.setdefault(category, []).extend(times)
    out = {}
    all_ranks = []
    for category, ranks in per_category.items():
        all_ranks.extend(ranks)
        times = sorted(latencies[category])
        out[category] = {
            "n": len(ranks),
            "R@1": round(sum(1 for r in ranks if r == 1) / len(ranks), 3),
            "R@10": round(sum(1 for r in ranks if r and r <= 10) / len(ranks), 3),
            "MRR@10": round(sum(1 / r for r in ranks if r and r <= 10) / len(ranks), 3),
            "p50_ms": round(times[len(times) // 2], 2),
        }
    out["ALL"] = {
        "n": len(all_ranks),
        "R@1": round(sum(1 for r in all_ranks if r == 1) / len(all_ranks), 3),
        "R@10": round(sum(1 for r in all_ranks if r and r <= 10) / len(all_ranks), 3),
        "MRR@10": round(sum(1 / r for r in all_ranks if r and r <= 10) / len(all_ranks), 3),
    }
    return out


def resolution(store):
    results = []
    for query, expected in RESOLUTION:
        got = resolve_people(store, [10, 11], query, 999)
        results.append({"query": query, "expected": expected, "got": got,
                        "ok": got == expected})
    return results


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store, gold_ids = build_store(Path(tmp))
        result = {"metrics": metrics(store, gold_ids), "resolution": resolution(store)}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
