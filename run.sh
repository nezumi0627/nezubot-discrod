#!/bin/bash
# run.sh — deploy.env を読み込んで nezubot-discrod を起動する
# Bot は単一インスタンスで動かすこと（二重起動すると二重に答える）。
# 常駐させる場合は systemd / タスクスケジューラ等から このスクリプトを呼ぶ。
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f deploy.env ]; then
  echo "deploy.env がありません。先に ./setup.sh を実行するか、" >&2
  echo "deploy.env.example を deploy.env にコピーして編集してください。" >&2
  exit 1
fi
if [ ! -x .venv/bin/nezubot-discrod ]; then
  echo ".venv が未セットアップです。先に ./setup.sh を実行してください。" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./deploy.env
set +a

exec .venv/bin/nezubot-discrod "$@"
