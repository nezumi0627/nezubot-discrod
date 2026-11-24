# Discord Adaptive Memory Engine — 研究・設計・高速化成果 総合まとめ v1.0
（2026-11-24、GPT出力第3弾。ユーザーが取得し共有。原文要約保存）

## 1. 研究目的
multilingual-e5-small（384次元）を採用し、検索アルゴリズム・記憶構造・独自SIMD演算によって e5-large 以上の検索品質を目指す。共有1.5コアCPU・メモリ6GBの小型VMで、500万件以上のDiscordログを高速検索できるシステムを構築する。

## 2. 現時点での研究成果（※GPT側の記述。こちらのVMでの実測ではない）
- 独自AVX2 SIMDカーネル: NumPy INT32比 21.5倍
- 100万ベクトル全件探索: p50 14.47ms / p95 23.48ms（独自AVX2）。NumPy INT32 505.74ms、FP32変換+BLAS 290.49ms
- 5,000候補の内積演算: p95 0.077ms
- 整数内積の計算結果は1万件テストでNumPyと完全一致
- ベクトル容量削減 98.13%（Large FP32・発言単位比の理論値）
- ただし実E5モデルでの検索精度や、量子化補正を含むコサイン類似度の性能は未検証（原文明記）

## 3. 最終アーキテクチャ
```
Gateway
  → Ingestion
  → Query Router（意図・人物・権限）
  → Contextual Memory Layer（Adaptive Chunker / Context Builder / Entity Resolver / Event Extractor）
  → Hybrid Storage（SQLite+FTS5 / Server Wiki・Wiki / Entity Index / Vector Store 384次元INT8）
  → High Performance Retrieval（E5-small ONNX INT8推論 / Native SIMD C・AVX2 / Adaptive RRF / Evidence Reranker）
  → External LLM（回答生成・制限付き動的想起）
```

## 4. 採用技術一覧
| 領域 | 技術 |
|---|---|
| 埋め込み | multilingual-e5-small |
| 推論 | ONNX Runtime・INT8量子化 |
| ベクトル保存 | INT8・384次元 |
| ベクトル演算 | C言語・AVX2 SIMD |
| ストレージ | SQLite・mmap |
| 全文検索 | FTS5 bigram |
| 記憶構造 | Hierarchical Memory |
| 会話分割 | Adaptive Chunking |
| 文脈補完 | Contextual Embedding |
| 人物検索 | Entity-aware Retrieval |
| 検索統合 | Adaptive RRF |
| 再ランキング | Evidence Reranking |
| 実行制御 | Deadline-aware Scheduling |
| 動的検索 | Adaptive Recall |
| 将来 | Knowledge Distillation |

## 5. メモリ・ストレージ最適化（理論値）
120万件・平均5発言/チャンク想定:

| 構成 | ベクトル容量 |
|---|---|
| Large FP32発言単位 | 4.92GB |
| Small FP32発言単位 | 1.84GB |
| Small INT8発言単位 | 460.80MB |
| Small INT8チャンク | 92.16MB（98.13%削減） |

500万発言でもチャンク化できればベクトル本体384MB。現段階ではANN必須とせず独自SIMD完全探索を優先し、本番負荷が問題になった場合のみANN検討。

## 6. Largeを超えるための精度戦略
- 会話チャンク化
- 文脈付き埋め込み
- 人物・別名索引
- Server Wikiページ埋め込み
- 階層型RAG
- Adaptive RRF
- Evidence Reranking
- Hard Negative Mining
- Knowledge Distillation（e5-largeを教師に Discord固有の日本語・俗語・内輪ネタ特化のe5-smallを学習。Largeの誤りまで学習しないよう人間確認の正解データも併用）

## 7. 性能目標（開発目標であり実測ではない）
| 項目 | 目標 |
|---|---|
| 埋め込み次元 | 1024→384 |
| 120万発言のベクトル容量 | 4.92GB→92.16MB |
| 埋め込み生成速度 | 140件/分→10〜30倍 |
| 通常検索p95 | 200ms以内 |
| 複雑な検索p95 | 500ms以内 |
| 事前検索期限 | 850ms |
| Recall@20・MRR | Large以上 |
| 対応規模 | 120万→500万件以上 |

## 8. ロードマップ（3/13 研究済み）
- 研究済み: 独自AVX2内積カーネル / 100万ベクトル負荷試験 / 整数内積の正確性検証
- P0: E5-small実モデル比較・ONNX最適化
- P0: INT8コサイン類似度・Native Top-K
- P0: Server Wiki埋め込み・人物索引
- P0: Adaptive Chunking・文脈補完
- P1: Adaptive RRF・根拠再ランキング
- P1: mmap・キャッシュ・権限無効化
- P1: 動的想起・期限管理
- P1: Small vs Large精度検証
- P2: ドメイン特化蒸留・Hard Negative Mining
- P2: 本番VMでの500万発言相当負荷試験

## 9. 最終結論
Sakanaの記憶構造を取り入れつつ、独自ネイティブ検索カーネル・適応型検索・階層型RAGで差別化。ベクトル演算は約21.5倍高速化の成果（GPT記述）。一方、E5-smallでLarge以上の検索品質を実現できるかはまだ証明できていない。今後は速度だけでなく実際のDiscordログを用いた検索精度の改善を最優先にする。
