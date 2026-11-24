#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
[[ $(id -u) != 0 && $HOME == /home/* && $(realpath "$HOME") == "$HOME" ]]
cd "$HOME/nezubot-discrod"
[[ $(realpath .) == "$HOME/nezubot-discrod" ]]
export TMPDIR="$HOME/.nezubot-discrod/tmp"
export FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed"
umask 077
mkdir -p "$TMPDIR" "$FASTEMBED_CACHE_PATH"
exec 9>"$HOME/.nezubot-discrod/hosting.lock"
flock -n 9 || { echo 'Another launcher is running' >&2; exit 1; }
child=
instance_file=
bot_running=false
cleanup_containers() {
    local label containers
    if [[ -n $instance_file && -f $instance_file ]]; then
        label=$(cat "$instance_file")
        if [[ $label =~ ^nezubot-discrod\.instance=[0-9a-f]{32}$ ]]; then
            containers=$(timeout 10 docker ps -aq --filter "label=$label") || return 1
            if [[ -n $containers ]]; then
                timeout 10 docker rm --force $containers || return 1
            fi
        elif [[ -n $label ]]; then
            echo 'Invalid sandbox instance label; refusing cleanup' >&2
            return 1
        fi
        rm -f "$instance_file"
        instance_file=
    fi
}
stop() {
    trap - TERM INT
    if [[ -n $child ]]; then
        if $bot_running; then
            kill -INT "$child" 2>/dev/null || true
            for ((attempt = 0; attempt < 15; attempt++)); do
                kill -0 "$child" 2>/dev/null || break
                sleep 1
            done
            kill -TERM -- "-$child" 2>/dev/null || true
            sleep 1
            kill -KILL -- "-$child" 2>/dev/null || true
        else
            kill -TERM "$child" 2>/dev/null || true
        fi
        wait "$child" || true
    fi
    cleanup_containers || echo 'Sandbox cleanup failed on stop.' >&2
    exit 0
}
trap stop TERM INT
while true; do
    if ! docker info --format '{{.OSType}}' 2>/dev/null | grep -qx linux; then
        echo 'Waiting for Linux Docker daemon...' >&2
    elif ! cleanup_containers; then
        echo 'Sandbox cleanup failed; retrying in 30 seconds.' >&2
    else
        instance_file=$(mktemp "$TMPDIR/instance-XXXXXX")
        bot_running=true
        setsid .venv/bin/python scripts/windows/run-bot.py "$instance_file" &
        child=$!
        wait "$child" || true
        child=
        bot_running=false
        cleanup_containers || echo 'Sandbox cleanup failed; will retry before restart.' >&2
        echo 'Bot exited; retrying in 30 seconds. Ctrl+C or stop the task to stop.' >&2
    fi
    sleep 30 &
    child=$!
    wait "$child" || true
    child=
done
