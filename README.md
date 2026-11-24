# nezubot-discrod

![Python](https://img.shields.io/badge/python-3.12%2B-blue)

メンションに答える Discord AI Bot。サーバー全体の発言ログを検索（RAG）して
過去の会話や人物について答え、メンバーごとの記憶（wiki）を持ちます。
人格（persona）は設定ファイル側に分離してあるので、名前や口調の例外を
コードを触らずに差し替えて、別のサーバーの別の Bot として再配備できます。

## 特徴

- **メンション／返信への AI 応答** — プロバイダは OpenAI 互換 API または
  Codex（ChatGPT ログイン）のどちらかを選択
- **サーバーログ RAG** — 全発言を SQLite（FTS5＋埋め込みベクトル）に蓄積し、
  キーワード検索と意味検索（RRF 融合）で過去の会話を引く
- **サーバー Wiki** — ログから人物と出来事の観察ページを自動生成し、
  出典付きで検索できる
- **メンバー Wiki** — ユーザーごとの長期記憶（好み・プロフィール）
- **読み取り専用 Web ツール** — Web 検索・ページ取得・Wikipedia
- **使い捨て Docker サンドボックス** — コマンド実行は隔離コンテナ内。
  既定で無効化も可
- **オーナー専用の自己改善** — `/improve` から承認付き PR を作る
- **長文回答のティーザー表示** — 一定長を超える回答は冒頭だけを
  チャンネルに出し、全文をスレッドに分ける
- **`n!` プレフィックスコマンド** — メンション不要で、学習状況や
  利用枠の確認、自分の発言の取り消しができる（下表）
- **会話理解** — 呼ばれた発言のリプライ先（数ホップ遡る）と直近の
  チャンネルの流れをその場で読み、メンションされた人物を表示名と
  Discord ID の対応つきで把握。「この人物」などの指示語は
  リプライ先の著者 → メンションされた人 → 直近の発言者の順で解決する
- **プロンプトキャッシュ最適化** — Codex 利用時は app-server を
  常駐させ、チャンネルごとの会話スレッドを使い回す。2 ターン目以降は
  新しい発言だけを送るため、入力の大半がキャッシュから処理される
- **更新履歴の自己申告** — Bot 自身の変更履歴を保持し、「何が変わった？」と
  聞かれたら履歴に基づいて答える。最新の変更は system prompt にも入る

## プレフィックスコマンド

メッセージの先頭に付ける。メンションは不要。

| コマンド | 内容 |
|---|---|
| `n!help` | 使い方を表示 |
| `n!del` | Bot の発言にリプライして送ると、その発言を取り消す（質問した本人かサーバーオーナーだけ） |
| `n!status` | 学習状況（総メッセージ数・学習人数・チャンネル数・学習率・サーバー Wiki 記事数など）を表示 |
| `n!limit` | ChatGPT/Codex サブスクリプションの利用枠（5 時間枠・週間枠の残り % とリセット時刻）を表示 |

## スラッシュコマンド

| コマンド | 内容 |
|---|---|
| `/search` | 閲覧できるサーバーログを検索 |
| `/wiki` | サーバー Wiki（人物・出来事）を自分だけに表示 |
| `/prompt` | 固定の system prompt を表示 |
| `/improve` | オーナー専用: 自己改善 PR を作成 |
| `/reset` | このチャンネルの会話履歴をリセット |

## 必要要件

- Python 3.12+
- Discord Bot アカウントと、対象サーバーへの招待
  （メッセージ内容インテントの有効化が必要）
- AI プロバイダ: OpenAI 互換 API キー、または Codex CLI＋ChatGPT ログイン
- （任意）Docker: `run_command` サンドボックス／自己改善を使う場合
- （任意）埋め込み用の計算資源: 初回の全件埋め込みは件数に応じて時間がかかる

## クイックスタート

```bash
git clone <この repo の URL>
cd nezubot-discrod
bash setup.sh   # venv 作成＋依存インストール＋ deploy.env の雛形配置
```

`deploy.env` を開いて、トークン・guild ID・owner ID・AI 設定を自分の値に
書き換えたら、起動する:

```bash
bash run.sh     # deploy.env を読み込んで Bot を起動
```

- `DISCORD_GUILD_ID` と `DISCORD_OWNER_ID` は必須で、未設定だと
  起動時に何が足りないか分かるエラーで止まる
- `deploy.env` は権限 600 で作られる。絶対にコミットしない
  （`.gitignore` 済み）
- 起動後、Bot が対象サーバーの発言をバックフィルし、埋め込みを生成する
  まで待ってから、メンションで応答を確認する
- 手動で同じことをやる場合: `python -m venv .venv` →
  `.venv/bin/pip install -e '.[dev,rag]'` → `deploy.env.example` を
  `deploy.env` にコピーして編集 → env を読み込んで `nezubot-discrod`

サーバー Wiki をまとめて再構築する場合:

```bash
.venv/bin/python -m nezubot_discrod.server_wiki --db "$HOME/.nezubot-discrod/server_log.db"
```

## 設定

- 全項目は `deploy.env.example` を参照（コードが読む env を網羅、
  全値プレースホルダ＋用途コメント付き）
- **ペルソナ**: `BOT_NAME`（Bot の名乗り）、`BOT_REPO_URL`（自己紹介で
  言及する URL、任意）、`PERSONA_RIVAL_USER_ID` / `PERSONA_RIVAL_NAME`
  （特定ユーザーへの口調の例外、任意）。プロンプト本文は
  `nezubot_discrod/prompts.py`、ペルソナの読み込みは
  `nezubot_discrod/persona.py` が担う
- **データ配置**: DB 等は `DATA_DIR`（既定 `~/.nezubot-discrod`）配下

## データについて（重要）

- **この repo にデータは含まれていません。**
- サーバーログ DB・サーバー Wiki・メンバー Wiki・埋め込みベクトルは、
  利用者が自分のサーバーのデータから自分で生成するものです。
  他人のサーバーのデータを持ち込むことは想定していません。
- 既存のアーカイブ DB を持ち込む場合は、そのサーバーの参加者の同意と
  プライバシーに各自で責任を持ってください。

## 運用メモ

- Bot は単一インスタンスで動かすこと（二重起動すると二重に答える。
  起動スクリプト側でのロック併用を推奨）
- ログ・DB・`deploy.env` は repo にコミットしない（`.gitignore` 済み）
- `host-tools/` は、Discord への直接接続ができない特殊なネットワーク
  環境向けのプロキシ中継の配備例。通常の起動には不要で、
  既定の起動経路（`run.sh`）には含まれていない

## ドキュメント

- `docs/windows.md` — Windows / WSL での常駐ガイド
- `docs/ADAPTIVE-MEMORY-ENGINE.md` / `docs/ADAPTIVE-MEMORY-ENGINE-RESEARCH.md`
  — 適応型メモリ（RAG）の設計検討
- `docs/OPTIMIZATION-AUDIT.md` — 検索・構造の監査記録
- `docs/OPTIMIZATION-WORKLOG.md` — RAG 最適化の作業記録（測定値付き）

## 開発

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## 参考

- Web ツールの設計は [evex-quotes の Web ツール](https://github.com/akku1139/evex-quotes/tree/main/src/aitools) を参考にしています（外部プロジェクト）。

## ライセンス

TBD（未定）
