# Windows 10/11: WSL2 常時稼働 / Always-on hosting

## 日本語

### 構成と安全境界

Windows ネイティブ Python ではなく **Ubuntu 24.04 / WSL2 の非 root ユーザー `nezubot`** で実行します。
Windows 側の checkout は PowerShell 管理用だけです。実行用 clone、venv、データ、Codex 認証、
モデルキャッシュ、一時 checkout はすべて WSL の Linux ファイルシステム内です。
`/mnt/c` などの Windows ドライブを作業領域・データ保存先・Docker bind mount に指定しないでください。
通常の AI shell はホストを一切 mount しない、ネットワークなしの使い捨て Docker コンテナです。
`/improve` だけは bot 所有の一時 clone を mount します。Docker socket、Windows ドライブ、
ホーム全体、稼働 checkout、`.env`、Codex 認証をコンテナへ渡してはいけません。
Dockerfile・policy・プロンプト・runner のセキュリティフラグは変更しません。
Docker 管理権限は root 相当です。ホスト管理者・承認済みコード・Docker 自体は信頼する必要があり、
任意の脆弱性まで防ぐ絶対保証ではありません。専用 WSL distro、バックアップ、人手の PR レビューを推奨します。

### 1. 前提とインストール（まだ bot は起動しません）

Windows Update、仮想化、WSL2 を有効にします。管理者 PowerShell で初回のみ:

```powershell
wsl --install -d Ubuntu
# 再起動し Ubuntu の初回セットアップを完了。24.04 が必要。
wsl --list --verbose
git clone https://github.com/your-account/nezubot-discrod.git C:\Users\Administrator\repos\nezubot-discrod
Set-Location C:\Users\Administrator\repos\nezubot-discrod
```

最新 WSL と systemd が必要です（`wsl --update`、Ubuntu 内の `ps -p 1 -o comm=` が `systemd`）。
必要なら `/etc/wsl.conf` の既存設定に `[boot]` / `systemd=true` を追加し、保存後 `wsl --terminate Ubuntu`。
WSL 未導入・再起動が必要な場合、install.ps1 は手順を表示して停止します。
スクリプト実行ポリシーは組織の設定に従ってください。必要ならこの PowerShell セッションだけ
`Set-ExecutionPolicy -Scope Process RemoteSigned` を使用し、恒久的に無効化しないでください。

Docker は次の **どちらか一方** を選びます。

* **ログオン不要の boot 起動（推奨）:** WSL 内 Docker Engine。
  `install.ps1 -DockerEngine` が Ubuntu の docker.io をインストールし systemd で有効化します。
  この distro の Docker Desktop WSL integration は無効のままにします。
* **Docker Desktop:** WSL2 backend / Linux containers を選び、Settings → Resources → WSL Integration
  で Ubuntu を有効にし、ログイン時に Docker Desktop を起動する設定にします。
  `install.ps1`（引数なし）を実行します。Desktop はユーザーログオンが必要なので boot task は使いません。

```powershell
.\scripts\windows\install.ps1 -DockerEngine
# Desktop の場合: .\scripts\windows\install.ps1
```

root は apt、ユーザー作成、Docker/Node/Codex の導入のみ。clone・pip・build は `nezubot` で実行します。
Python 3.12+、Node、Codex **0.171.0**、`.[dev,rag]`、sandbox/selfmod の両 image を導入します。
実行用 repo は `/home/nezubot/nezubot-discrod`、venv は `.venv`。
既存 clone は勝手に pull/reset しません。更新は bot を停止して非 root でレビュー後 `git pull --ff-only`、
install.ps1 を再実行してください。未マージの branch を試す場合は `-Branch <branch>`（初回 clone のみ）。
private repo は PowerShell の GITHUB_TOKEN 環境変数を一時的に使用できます（Git 設定には保存しません）。
または WSL 内で owner が事前に認証し clone してください。token を URL に含めないでください。

### 2. 認証と設定

```powershell
wsl -d Ubuntu -u nezubot -- bash
```

以降は WSL shell:

```bash
cd ~/nezubot-discrod
nano .env
# DISCORD_TOKEN=... をローカルで設定。commit・共有しない。
# Codex: AI_PROVIDER=codex / AI_MODEL=gpt-6-luna / CODEX_BINARY=codex
# CODEX_HOME=~/.codex / AI_API_KEY は不要
chmod 600 .env
codex login --device-auth
codex --version
# codex-cli 0.171.0
```

Codex が認証ファイルを作るよう、`~/.codex/config.toml` に
`cli_auth_credentials_store = "file"` を設定してから login してください（既存の同じキーは置換）。
`~/.codex/auth.json` が必要です。keyring のみのログインは使えません。
個人の Codex 設定は bot に読み込まれません。bot は auth.json だけを一時的な隔離 HOME にコピーし、
固定の read-only / tool lockdown 設定と stdio app-server を使用します。
OpenAI provider を使う場合は `.env` に AI_PROVIDER=openai と AI_API_KEY を設定します。
`GITHUB_TOKEN` は `/improve` 用の任意設定です。Windows パスを `SELF_CHECKOUT` / `CODEX_HOME` に指定しないでください。
`.env` や認証情報をログ・Git・タスク引数へ含めないでください。

### 3. Linux からのデータ移行

**旧 bot を停止してから**旧 Linux 上でバックアップを作ります（SQLite WAL も含めます）。

```bash
umask 077
tar -czf ~/nezubot-data.tar.gz -C "$HOME" .nezubot-discrod
```

新 WSL の `nezubot` shell で直接コピーします。Windows ドライブを経由しません。

```bash
scp olduser@old-host:nezubot-data.tar.gz ~/nezubot-data.tar.gz
bash ~/nezubot-discrod/scripts/windows/import-data.sh ~/nezubot-data.tar.gz
```

移行は bot 停止中のみ実行。既存データがある場合は、先に別名へ手動バックアップしてください。
import は上書きせず、通常の wiki/server_log SQLite ファイルだけを受理し、リンク・絶対パス・
ディレクトリ脱出・空アーカイブを拒否します。他のファイルが含まれる場合は信頼する owner が内容を確認し、
SQLite ファイルだけの tarball を作り直してください。認証と `.env` は移行しません。

保存場所:

| 内容 | WSL パス |
|---|---|
| サーバーログ・埋め込みベクトル | `~/.nezubot-discrod/server_log.db` |
| Wiki | `~/.nezubot-discrod/wiki.db` |
| 一時 sandbox 改善 clone | `~/.nezubot-discrod/tmp/discord-selfmod-*` |
| FastEmbed モデル（再ダウンロード可） | `~/.cache/fastembed` |
| Codex ログイン | `~/.codex/auth.json` |

旧 FastEmbed cache のコピーは不要です。初回にネットワークからモデルを取得します。
DB 内のベクトルは DB と一緒に移行されます。runner は TMPDIR と FASTEMBED_CACHE_PATH を Linux home に固定します。

### 4. 検証・起動・自動復旧

WSL 非 root shell で（テスト時は運用 .env の読み込みを無効化）:

```bash
cd ~/nezubot-discrod
.venv/bin/ruff check .
PYTHON_DOTENV_DISABLED=1 .venv/bin/pytest -q
.venv/bin/python -c 'import nezubot_discrod.bot; print("Bot module import: OK")'
```

旧インスタンス停止後のみ PowerShell で手動起動:

```powershell
.\scripts\windows\run.ps1 -DockerEngine
# Desktop: -DockerEngine を省略
```

30 秒間隔で Docker を待ち、bot 終了後も再起動します。bot 自体は discord.py の再接続を使用します。
多重 launcher は flock で拒否します。手動時は Ctrl+C で停止します。
停止時は SIGINT で bot の終了処理を待ち、15 秒後は残存プロセスを強制停止します。
その bot 固有の instance label が一致する sandbox だけを削除し、他の bot のコンテナには触れません。
タスクは **WSL を所有する Windows ユーザー**で登録してください。SYSTEM ではユーザーの distro が見えません。

```powershell
# ログオン時（Desktop または Engine）
.\scripts\windows\register-task.ps1 -DockerEngine
# または boot 時: 管理者 PowerShell、Engine 専用。Windows アカウントのパスワードをローカル入力。
.\scripts\windows\register-task.ps1 -AtStartup -DockerEngine
# 登録だけでは起動しません。旧 bot 停止後:
Start-ScheduledTask -TaskName DiscordSandboxAI
Get-ScheduledTaskInfo -TaskName DiscordSandboxAI
Stop-ScheduledTask -TaskName DiscordSandboxAI
```

boot mode のパスワードは Task Scheduler が保存します。Windows Hello PIN ではなくアカウントのパスワードが必要です。
再起動後ログオン前に起動できるか owner の実機で確認してください（Windows/WSL ポリシーに依存）。
停止確認は `wsl -d Ubuntu -u nezubot -- pgrep -af nezubot-discrod`。
タスク強制停止時に WSL プロセスが残る場合は、専用 distro を `wsl --terminate Ubuntu` で停止します
（この distro 内の他の処理もすべて止まります）。再度起動する前に残存 bot がないことを確認してください。
削除は `Unregister-ScheduledTask -TaskName DiscordSandboxAI`。
AC 電源時のスリープ・休止を無効にし、Windows Update 後の復旧、ディスク容量、定期バックアップを確認します。
ログは手動実行のコンソールに出ます。Task Scheduler の履歴は launcher の状態だけです。

### 互換性メモ

ネイティブ Windows Python はサポートしません。既存の uid/gid、geteuid、O_NOFOLLOW、chmod、
Git の `/dev/null`・`/bin/false`、`/bin/sh`、execv、POSIX SQLite 所有権チェックを WSL で維持します。
fcntl や独自 Unix signal handler、Codex Unix socket は不要です。rootless Docker の workspace UID 0:0 判定も変更しません。
通常 sandbox は常に UID 65534 / network none / read-only / cap-drop ALL / no-new-privileges のままです。
Windows 全体へのアクセスをさらに絞る場合は専用 distro の `/etc/wsl.conf` で automount と Windows interop を無効化できます。
既存 distro の設定を script が自動変更することはありません。Desktop 統合への影響を owner が確認してください。

## English

### Architecture and prerequisites

Run the unchanged Linux bot as **non-root `nezubot` inside Ubuntu 24.04 on WSL2**, not native Windows Python.
The Windows checkout only holds management scripts. All runtime code, venvs, databases, credentials, model cache,
and temporary improvement clones live under `/home/nezubot`. Never use `/mnt/c` or another Windows drive as
data, working directory, or a Docker bind mount. Ordinary AI commands receive **no host mounts** and run in
disposable, offline, hardened containers. `/improve` retains its bot-owned temporary checkout mount only.
Never mount the Docker socket, credentials, home directory or live checkout. Docker access is root-equivalent;
trusted administrators, reviewed feature code, backups and a dedicated distro remain important.

Install/update WSL2 (`wsl --install -d Ubuntu`, reboot, finish Ubuntu setup; `wsl --update`). Require Ubuntu 24.04,
systemd as PID 1 and virtualization enabled. If needed, merge `[boot]` / `systemd=true` into `/etc/wsl.conf`,
then terminate and restart that distro. Clone this repo to `C:\Users\Administrator\repos\nezubot-discrod`.
Respect local PowerShell execution policy; a process-scoped RemoteSigned policy may be used if permitted.

Choose ONE Docker backend:

* **Unattended boot:** run `scripts\windows\install.ps1 -DockerEngine`. Installs Ubuntu docker.io and enables its
  systemd service. Leave Docker Desktop integration disabled for this distro.
* **Docker Desktop:** enable the WSL2 backend, Linux containers and Ubuntu WSL integration; enable Desktop
  startup at sign-in. Run `scripts\windows\install.ps1` without the switch. Requires Windows sign-in.

Setup installs Python 3.12+, Node/npm, pinned Codex 0.171.0, `.[dev,rag]`, and both Docker images.
It creates `/home/nezubot/nezubot-discrod/.venv`, but **does not start the bot**. Existing clones are not
reset or pulled. Stop the bot, review/pull updates as nezubot, then rerun setup to update dependencies/images.
`-Branch` selects an initial clone branch only. For a private repo, a temporary PowerShell GITHUB_TOKEN environment
variable is forwarded for cloning without persisting Git credentials; alternatively authenticate and clone inside
WSL first. Never embed a token in the URL.

### Configure and migrate

Open `wsl -d Ubuntu -u nezubot -- bash`. In `~/nezubot-discrod`, edit `.env` locally and chmod it to 600.
Use `DISCORD_TOKEN` from this file or the environment. For Codex choose `AI_PROVIDER=codex`, `AI_MODEL=gpt-6-luna`,
`CODEX_BINARY=codex`, `CODEX_HOME=~/.codex`; no AI_API_KEY is needed. Set
`cli_auth_credentials_store = "file"` in `~/.codex/config.toml`, then run **`codex login --device-auth`**.
The bot requires `~/.codex/auth.json`, copying only that file into an isolated temporary Codex home. Personal
config, plugins, shell tools and MCP servers are not loaded. OpenAI mode instead needs AI_API_KEY.
GITHUB_TOKEN is optional for `/improve`. Never commit secrets or put them in task arguments or logs.

Stop the old Linux bot. There, run `umask 077; tar -czf ~/nezubot-data.tar.gz -C "$HOME" .nezubot-discrod`.
From the new WSL bot shell, run:

```bash
scp olduser@old-host:nezubot-data.tar.gz ~/nezubot-data.tar.gz
bash ~/nezubot-discrod/scripts/windows/import-data.sh ~/nezubot-data.tar.gz
```

Import refuses existing nonempty data, empty archives, links, traversal and entries other than wiki/server-log
SQLite files and their WAL/SHM/journal sidecars. Back up existing data manually first. Keep both bots stopped
during migration. Do not migrate `.env` or Codex credentials through this archive. Databases live in
`~/.nezubot-discrod/{wiki,server_log}.db`; embeddings are inside server_log.db. Models redownload into
`~/.cache/fastembed`. The launcher fixes TMPDIR to `~/.nezubot-discrod/tmp` so improvement clones stay on Linux.

### Verify and operate

As the WSL bot user, run `.venv/bin/ruff check .`, `PYTHON_DOTENV_DISABLED=1 .venv/bin/pytest -q`, and
`.venv/bin/python -c 'import nezubot_discrod.bot; print("Bot module import: OK")'` from the runtime checkout.
Only after stopping the old bot, run `scripts\windows\run.ps1 -DockerEngine` (omit switch for Desktop).
The launcher waits for Docker and restarts the bot every 30 seconds after exit; flock rejects duplicate launchers.
On stop it sends SIGINT for graceful cleanup, then forces remaining processes down after 15 seconds and removes
only containers matching that bot's exact instance label, leaving unrelated containers untouched.

Register `scripts\windows\register-task.ps1 -DockerEngine` for sign-in, or, in elevated PowerShell,
`scripts\windows\register-task.ps1 -AtStartup -DockerEngine` for boot. Register as the Windows account owning
the distro, **not SYSTEM**. Boot registration prompts locally for that account's password (not Hello PIN), stored
by Task Scheduler. Registration does not start the bot. Use Start-/Stop-/Get-ScheduledTaskInfo with
`-TaskName DiscordSandboxAI`. Verify a real reboot before sign-in on the owner's PC: WSL and account policies vary.
Manual run stops with Ctrl+C. After stopping a task, check `wsl -d Ubuntu -u nezubot -- pgrep -af nezubot-discrod`;
if WSL children remain, terminate the dedicated distro with `wsl --terminate Ubuntu` before restarting.
This stops ALL work inside that distro. Remove scheduling with Unregister-ScheduledTask.
Disable sleep/hibernation on AC, plan Windows Update restarts, monitor disk space, and back up stopped databases.
Console output is available on manual runs; Task Scheduler history tracks the launcher rather than bot logs.

No security checks were bypassed for Windows: POSIX uid/gid/geteuid, O_NOFOLLOW, mode/ownership checks,
`/dev/null`, `/bin/false`, shell execution and execv retain Linux semantics. Codex uses stdio, not a Unix socket.
Rootless Docker UID mapping and all existing sandbox security flags remain unchanged. Native Windows Python
is unsupported. For additional isolation, an owner can disable Windows automount/interop in a dedicated distro;
scripts deliberately do not overwrite an existing distro's wsl.conf.
