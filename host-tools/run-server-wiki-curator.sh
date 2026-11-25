#!/bin/bash
# Launch the server wiki curator.
# - The curator patrols the server wiki and verifies entries against
#   the server log. Its default backend is the local Devin bridge
#   (SWE-2.1); the bot's answer path is untouched and stays on the
#   configured AI provider (AI_PROVIDER in deploy.env).
# - deploy.env is sourced for DISCORD_GUILD_ID and shared settings.
# - NO_PROXY is pinned so the localhost Devin bridge is never proxied.
cd "$(dirname "$0")/.." || exit 1
set -a
. ./deploy.env
set +a
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export SERVER_WIKI_CURATOR_BACKEND="${SERVER_WIKI_CURATOR_BACKEND:-devin}"
export SERVER_WIKI_CURATOR_MODEL="${SERVER_WIKI_CURATOR_MODEL:-swe-2-1}"
# Single-instance guard (the systemd service is the normal launcher;
# this stops a manual second copy from double-curating the wiki).
exec 9>"$HOME/.nezubot-discrod/wiki-curator.lock"
if ! flock -n 9; then
  echo "run-server_wiki-curator: another instance is already running; refusing to start" >&2
  exit 1
fi
exec .venv/bin/python -m nezubot_discrod.server_wiki_curator \
  --db "$HOME/.nezubot-discrod/server_log.db" \
  --watch --units 4 --interval 600 --call-timeout 900 \
  >> "$HOME/.nezubot-discrod/wiki-curator.log" 2>&1
