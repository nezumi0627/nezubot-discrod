#!/bin/bash
# Launch the bot in this sandbox VM.
# - deploy.env holds the config (600, never committed)
# - NO_PROXY is sanitized: the ambient value contains bracketed IPv6
#   entries that crash httpx (HF model download) outright.
# - FASTEMBED_CACHE_PATH must live on the home volume: /tmp is a 512MB
#   tmpfs and the E5 model is ~2.2GB.
# - The local ws-relay (port 18080) must be running for the gateway.
cd "$(dirname "$0")/.." || exit 1
set -a
. ./deploy.env
set +a
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed"
# Single-instance guard: a second launch while one bot is alive makes both
# answer every message. Hold an flock for the process lifetime (fd 9 is
# inherited across exec).
exec 9>"$HOME/.nezubot-discrod/bot.lock"
if ! flock -n 9; then
  echo "run-bot: another instance is already running; refusing to start" >&2
  exit 1
fi
exec .venv/bin/nezubot-discrod >> bot.log 2>&1
