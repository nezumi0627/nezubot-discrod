# docs

nezubot-discrod の設計・研究・最適化に関するドキュメント置き場。導入と使い方はルートの [README](../README.md) を参照。

| ドキュメント | 内容 |
| --- | --- |
| [ADAPTIVE-MEMORY-ENGINE.md](ADAPTIVE-MEMORY-ENGINE.md) | 記憶エンジンの技術アーキテクチャ。multilingual-e5-small を中核に、検索アルゴリズム・記憶構造・文脈補完によって e5-large 以上の検索品質を目指す全体設計 |
| [ADAPTIVE-MEMORY-ENGINE-RESEARCH.md](ADAPTIVE-MEMORY-ENGINE-RESEARCH.md) | 研究・設計・高速化成果の総合まとめ。研究目的、SIMD 高速化などの成果と設計判断の記録（外部モデルの出力を要約保存したものを含む） |
| [OPTIMIZATION-AUDIT.md](OPTIMIZATION-AUDIT.md) | コードを変更せずに行った静的監査の結果。ボトルネックと改善候補の指摘一覧 |
| [OPTIMIZATION-WORKLOG.md](OPTIMIZATION-WORKLOG.md) | 最適化の作業記録。RAG レビューと構造整理で実施した変更と計測結果 |
| [windows.md](windows.md) | Windows 10/11 + WSL2 で Bot を常時稼働させるためのホスティング手順 |
