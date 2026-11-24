"""The bot's own update history, newest first.

This is the single source the bot speaks from when asked what changed
about itself: the latest entry is injected into the system prompt (see
prompts.py), and n!status shows the latest version. Entries are factual
release notes written when a change ships; never backfill invented
changes here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateEntry:
    version: str
    date: str  # ISO calendar date the change shipped
    changes: tuple[str, ...]


UPDATES: tuple[UpdateEntry, ...] = (
    UpdateEntry(
        "1.9.2",
        "2026-11-24",
        (
            "n!help / n!status / n!limit の出力を、チャンネルへの公開投稿から"
            "コマンドを使った本人への DM 送信に変更した。届いた合図に 📩、"
            "DM が送れない場合は ⚠️ のリアクションだけを付け、"
            "公開投稿には戻さない",
            "スラッシュコマンドを監査し、/ping の応答も ephemeral"
            "（自分だけ表示）に揃えた。他のコマンド"
            "（/search /wiki /prompt /reset /improve）は元から ephemeral",
        ),
    ),
    UpdateEntry(
        "1.9.1",
        "2026-11-24",
        (
            "同じチャンネルで応答中に呼ばれたときに出していた"
            "「処理中です。少し待ってから再試行してください。」の"
            "通知メッセージを送らないようにした。取りこぼしは"
            "チャンネルには表示されず、ログにだけ残る",
        ),
    ),
    UpdateEntry(
        "1.9.0",
        "2026-11-24",
        (
            "検索をハイブリッド化: 従来の字句検索＋埋め込み検索に、"
            "学習済みモデル（ft）の埋め込み検索を3本目の経路として追加し、"
            "3経路の結果を統合してから提示するようにした",
            "投入前にオフライン評価（600クエリ）で現行方式と比較し、"
            "hit@10 が 0.0233→0.0350、MRR@10 が 0.0073→0.0093 に"
            "改善することを確認した",
            "ft 側のモデルやベクトルが使えない環境では、従来の2経路検索に"
            "自動でフォールバックする。新着メッセージの ft 埋め込みも"
            "専用ワーカーが継続して行う",
        ),
    ),
    UpdateEntry(
        "1.8.0",
        "2026-11-24",
        (
            "プロンプトキャッシュを効かせるため、Codex の app-server を"
            "常駐させ、チャンネルごとの会話スレッドを使い回すようにした。"
            "2ターン目以降は新しい発言だけを送るので、入力の大半が"
            "キャッシュから処理されるようになった",
            "常駐サーバーが落ちた場合は自動で作り直し、それもできない"
            "ときは従来の毎回起動する方式に切り替えて回答を続ける",
            "会話履歴のリセット（/reset）では、サーバー側で保持している"
            "会話スレッドも一緒に破棄するようにした",
        ),
    ),
    UpdateEntry(
        "1.7.0",
        "2026-11-24",
        (
            "プロンプトキャッシュについて調査し、会話履歴を「送信した形そのまま」"
            "再送するように直した。リクエストがターンごとに前方一致し続ける形に"
            "なった",
            "Codex の利用量にキャッシュ済み入力トークンを記録するようにした。"
            "なお現状はバックエンド側がキャッシュを返しておらず、記録は 0 のまま"
            "（Bot 側の組み立ての問題ではないことを実測で確認済み）",
        ),
    ),
    UpdateEntry(
        "1.6.1",
        "2026-11-24",
        (
            "n!limit の表示を「使用済み %」から「残り %」に変更した"
            "（5時間枠・週間枠とも、リセット時刻はそのまま併記）",
        ),
    ),
    UpdateEntry(
        "1.6.0",
        "2026-11-24",
        (
            "会話理解を強化: 呼ばれた発言のリプライ先（最大3つ前まで）と、"
            "直近のチャンネルの会話の流れをその場で読んでから答えるようにした",
            "「この人物」「この人」などの指示語は、リプライ先の著者 → "
            "メンションされた人 → 直近の発言者の順で解決し、サーバーオーナーや"
            "発言者本人に誤って結びつけないようにした",
            "メンションされた人物を、表示名と Discord ID の対応つきで"
            "把握するようにした",
        ),
    ),
    UpdateEntry(
        "1.5.0",
        "2026-11-24",
        (
            "n! プレフィックスコマンドを追加（メンション不要で n!help / n!del / n!status）",
            "n!del: 自分の発言にリプライして送ると、その発言を取り消せるようにした"
            "（質問した本人かサーバーオーナーだけ）",
            "n!status: 学習状況（総メッセージ数・学習人数・チャンネル数・学習率・"
            "Server Wiki 記事数など）を表示",
            "n!limit: ChatGPT/Codex サブスクリプションの利用枠"
            "（5時間枠・週間枠の使用率とリセット時刻）を表示",
            "自分が「nezubot-discrod」であること、テストユーザー（testuser0627）が作った Bot "
            "であることの自覚を人格に追加",
            "アップデート内容を聞かれたら、この更新履歴に基づいて答えるようにした",
        ),
    ),
    UpdateEntry(
        "1.4.0",
        "2026-11-24",
        (
            "人格を調整: ライバル設定の相手にだけ好戦的に返し、それ以外の相手には温厚・簡潔に返す",
            "長文の回答はチャンネルに約100字のティーザーを出し、全文をスレッドに分けるようにした",
            "モデルを gpt-6-luna にした",
        ),
    ),
    UpdateEntry(
        "1.3.0",
        "2026-11-24",
        (
            "RAG 最適化: 人物解決（名前・エイリアス照合）を高速化し、"
            "数百 ms かかっていた処理を数 ms に短縮",
            "RAG 最適化: 出来事に関する質問では Server Wiki の出来事記事を事前に"
            "取得してから答えるようにした",
            "RAG 最適化: システムプロンプトとツール説明の重複を整理し、"
            "毎回の固定コンテキストを削減",
        ),
    ),
    UpdateEntry(
        "1.2.0",
        "2026-11-24",
        (
            "埋め込みを教師ベクトル（multilingual-e5-small）で全件化:"
            " 静的ベクトルで代替していた分も含め、対象メッセージすべてに教師ベクトルを用意した",
        ),
    ),
)


def latest_update() -> UpdateEntry:
    return UPDATES[0]


def latest_version() -> str:
    return UPDATES[0].version


def format_entry(entry: UpdateEntry) -> str:
    lines = [f"v{entry.version}（{entry.date}）"]
    lines.extend(f"- {change}" for change in entry.changes)
    return "\n".join(lines)


def format_latest_update() -> str:
    """The latest entry, formatted for system-prompt injection."""
    return format_entry(latest_update())


def format_update_history() -> str:
    """Every entry, newest first."""
    return "\n\n".join(format_entry(entry) for entry in UPDATES)
