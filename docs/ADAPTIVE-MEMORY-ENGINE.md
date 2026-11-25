# Discord Adaptive Memory Engine — Technical Architecture v1.0
（2026-11-24、GPT設計レビュー第2弾。ユーザーが取得し共有。原文ママ保存）

multilingual-e5-smallを中核とし、検索アルゴリズム・記憶構造・文脈補完によってe5-large以上の検索品質を目指す、Discord特化型の軽量検索エンジン。

## 01. システム全体設計
- Discord Gateway（Message / Edit / Delete / Mention）
- Ingestion Engine（保存・差分更新）
- Query Engine（質問・検索意図解析）
- Memory Processing Layer: Context Builder（文脈補完）/ Adaptive Chunker（会話の構造化）/ Entity Resolver（人物・別名解決）/ Event Extractor（出来事の抽出）
- Storage & Index Layer: SQLite（原文・メタデータ）/ FTS5（日本語語彙索引）/ Vector Store（E5-small / INT8）/ Entity Index（人物・関係性）/ Server Wiki（人物・出来事記憶）/ MemberWiki（プロフィール）
- Adaptive Retrieval Engine: Permission Filter → Query Router → Candidate Generator → Adaptive Fusion → Evidence Reranker → Context Compressor
- External LLM（回答生成・必要時のみ動的想起）

## 02. 設計目標
- 対応メッセージ数: 500万件+
- 通常検索 p95: 200ms以内を目標
- 検索ハード期限: 850ms
- 検索品質: Large以上（Discord固有の評価セットで）
- ※性能値は実測ではなく開発目標。共有1.5コア・6GBのVMを前提とする。

## 03. 記憶構造 — Hierarchical Memory
- L0 Working Memory: 直近の会話・現在の質問（短期キャッシュ）
- L1 Semantic Memory: 人物・知識・要約・Server Wiki（長期意味記憶）
- L2 Episodic Memory: 出来事・会話チャンク（エピソード記憶）
- L3 Raw Memory: 全メッセージ・原文（永続的な記録）
- 各階層は元メッセージへの参照を持ち、要約から原文へ遡れる。すべての検索で全階層を経由する必要はない。

## 04. データベース設計
物理的なユーザー別DBは作らず、関係テーブルによって論理的に分離する。

関係: USERS—MESSAGES（authors）/ USERS—USER_ALIASES / USERS・MESSAGES—MESSAGE_PEOPLE / MESSAGES—CHUNK_MESSAGES—CHUNKS / CHUNKS—CHUNK_PEOPLE—USERS / CHUNKS—CHUNK_EMBEDDINGS / MEMORY_NODES—MEMORY_EVIDENCE—MESSAGES / MEMORY_NODES—MEMORY_EMBEDDINGS / EMBEDDING_MODELS—CHUNK_EMBEDDINGS・MEMORY_EMBEDDINGS

| テーブル | 責務 |
|---|---|
| messages | メッセージの正本 |
| users | プロフィール |
| user_aliases | あだ名・表記ゆれ・履歴 |
| message_people | 発言・言及・返信・人物関係 |
| chunks | 会話のまとまり |
| chunk_messages | チャンクと原文の対応 |
| chunk_people | 会話に関係する人物 |
| chunk_embeddings | 会話ベクトル |
| memory_nodes | Server Wiki・人物・出来事 |
| memory_evidence | 要約の根拠 |
| memory_embeddings | 記憶ベクトル |
| embedding_models | モデルの同一性管理 |
| indexing_jobs | 差分更新ジョブ |

SQLiteを正本とし、ベクトル索引や要約は再構築可能な派生データとして扱う。

## 05. E5-small高速化エンジン
Input Normalization（日本語正規化・別名・返信参照）→ Context Enrichment（前後の会話・話題・人物情報）→ Length Bucketing（入力長ごとのバッチ化）→ ONNX INT8 Inference（multilingual-e5-small）→ Mean Pooling + L2 Normalization → 384次元の正規化ベクトル → Vector Quantization（INT8 + scale / norm）→ Contiguous Vector Store（連続配列・SIMD検索）

| 最適化技術 | 目的 |
|---|---|
| ONNX Graph Optimization | 不要な演算・演算子の統合 |
| Dynamic INT8 | モデル推論の高速化 |
| CPU命令セット最適化 | AVX2・VNNI等を利用可能な場合に活用 |
| Length Bucketing | Paddingによる無駄な演算削減 |
| Content Hash | 重複推論の回避 |
| SIMD Dot Product | ベクトル比較の高速化 |
| Memory Mapping | 大量のBLOB展開を回避 |
| Partial Top-K | 全候補のソートを回避 |
| Incremental Indexing | 変更箇所のみ再計算 |
| Priority Scheduler | 通常検索をバックフィルより優先 |

モデル量子化とベクトル量子化は別々に評価する。CPUの命令セットによってはINT8モデルが必ずしも高速になるとは限らない。

## 06. Adaptive Chunking
固定長ではなく会話構造を考慮: 境界は時間、発言数、トークン数、返信関係、話題の変化で決定。
LLMによる全件要約は行わない。通常は原文の連結とルールベースの文脈補完を使用し、既存Server Wikiの要約が利用可能な場合だけ補助的に使う。

## 07. Adaptive Retrieval Algorithm
質問種別で経路を変更（人物/出来事/関係/意味/語句）。
重み付きRRF: S_RRF(d) = Σ_i w_i(q) / (k + r_i(d))。初期例（初期設計例であり評価で調整）: 人物質問で FTS5 15% / E5-small 20% / 人物索引 40% / Server Wiki 25%。
候補への軽量再ランキング特徴量: 人物・別名の一致度 / 返信・メンション関係 / 時間条件との一致 / 語彙・意味の一致度 / 根拠の信頼性 / 会話の連続性。初期はルールベース、評価データが蓄積したらロジスティック回帰や軽量LTRへ移行。

## 08. Deadline-aware Execution
850msを使い切るのではなく、必要な検索が完了した時点で終了する。期限到達時は取得済み結果だけで回答生成へ。キューと同時実行数も管理する。

## 09. ストレージ・メモリ設計
| 条件 | ベクトル容量（120万件想定） |
|---|---|
| Large FP32・従来構成 | 4.92 GB |
| Small FP32・チャンク（平均5件/チャンク） | 368.6 MB |
| Small INT8・チャンク | 92.2 MB |

## 10. 打倒Largeの研究計画（アブレーション）
- A: E5-large + 現行RRF
- B: E5-small + 現行RRF
- C: B + Adaptive Chunking
- D: C + Contextual Embedding
- E: D + 人物・出来事索引
- F: E + Adaptive RRF
- G: F + Evidence Reranking
- H: G + ドメイン特化蒸留
Large側にも同じ改善を適用した比較を別途行い、モデルの差とアーキテクチャの差を区別する。最終段階の蒸留（Knowledge Distillation / Hard Negative Mining）の学習処理はVM外で実施。

## 11. プロジェクト構成（案）
memory_engine/ 配下に core / ingestion / processing / embedding / storage / memory / retrieval / security / tools(recall) / evaluation を分離。権限変更・メッセージ削除時には、キャッシュ・チャンク・要約・ベクトルの無効化を一貫して行う。

## 12. 実装ロードマップ
- P0: 評価基盤（Large・Small比較、精度・速度・メモリ計測）
- P0: Embedding Engine（ONNX最適化・INT8・モデル管理）
- P0: Memory Index（Server Wiki埋め込み・人物索引・権限管理）
- P0: Adaptive Chunking（会話分割・文脈補完・差分更新）
- P1: Retrieval Engine（検索ルーティング・Adaptive RRF）
- P1: Evidence Reranking（根拠評価・重複排除・文脈圧縮）
- P1: Runtime Optimization（SIMD・mmap・キャッシュ・期限管理）
- P1: Dynamic Recall（LLMによる制限付き追加検索）
- P2: Knowledge Distillation（Discord特化型Smallの学習）
- P2: Scale Testing（500万件負荷試験・必要時のみANN導入）

最終設計思想: Small Model × Rich Context × Adaptive Retrieval
最終的な成功条件: 単にsmallが速いことではなく、同じDiscordログに対して、現行Large構成より正確な根拠を、少ない計算資源で、より短時間に取得できること。
