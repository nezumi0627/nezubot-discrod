#!/bin/bash
# Launch the reading-time service in this sandbox VM.
# - Twice a day (READING_TIMES, default 12:30 / 21:30 JST) the bot
#   gets 30 minutes of reading time in two halves: 15 minutes to
#   study one random AI theme on the public web, then 15 minutes
#   patrolling the link-sharing channel (READING_PATROL_CHANNEL_ID)
#   for newly shared repositories. It writes a report and keeps
#   worthwhile pages and repos as books in
#   ~/.nezubot-discrod/reading.db. Its default backend
#   is the local Devin bridge (SWE-2.1); the bot's answer path
#   is untouched.
# - deploy.env is sourced for shared settings.
# - NO_PROXY is pinned so the localhost Devin bridge is never proxied.
cd "$(dirname "$0")/.." || exit 1
set -a
. ./deploy.env
set +a
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export READING_BACKEND="${READING_BACKEND:-devin}"
export READING_MODEL="${READING_MODEL:-swe-2-1}"
export READING_PATROL_SECONDS="${READING_PATROL_SECONDS:-900}"
export READING_PATROL_CHANNEL_ID="${READING_PATROL_CHANNEL_ID:-123456789012345678}"
# Single-instance guard (the systemd service is the normal launcher;
# this stops a manual second copy from double-running sessions).
exec 9>"$HOME/.nezubot-discrod/reading-service.lock"
if ! flock -n 9; then
  echo "run-reading: another instance is already running; refusing to start" >&2
  exit 1
fi
exec .venv/bin/python -m nezubot_discrod.reading \
  --db "$HOME/.nezubot-discrod/reading.db" \
  --watch \
  >> "$HOME/.nezubot-discrod/reading.log" 2>&1
