#!/bin/bash
# setup.sh — nezubot-discrod の初期セットアップ（冪等。何度実行してもよい）
#   1. .venv を作る
#   2. 依存を入れる（開発・RAG 込み）
#   3. deploy.env が無ければ deploy.env.example からコピーする（権限 600）
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
echo "使用する Python: $($PYTHON --version 2>&1) （$PYTHON）"

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
  echo ".venv を作成しました"
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,rag]'

if [ ! -f deploy.env ]; then
  cp deploy.env.example deploy.env
  chmod 600 deploy.env
  echo "deploy.env を作成しました（deploy.env.example のコピー、権限 600）"
  echo "→ deploy.env を開いて、トークン・guild ID・owner ID・AI 設定を自分の値に書き換えてください"
else
  echo "deploy.env は既にあるため上書きしませんでした"
fi

echo "セットアップ完了。deploy.env を編集したら ./run.sh で起動できます"
