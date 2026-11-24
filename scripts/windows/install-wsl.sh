#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
user=${1:?user required}
mode=${2:?docker mode required}
repository=${3:?repository required}
branch=${4:?branch required}
[[ $user =~ ^[a-z][a-z0-9_-]*$ && $user != root ]]
[[ $mode == engine || $mode == desktop ]]
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -o DPkg::Lock::Timeout=300 install -y --no-install-recommends python3 python3-venv python3-pip git curl ca-certificates nodejs npm
python3 -c 'import sys; assert sys.version_info >= (3,12), "Ubuntu 24.04 / Python 3.12+ required"'
if ! id "$user" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "$user"
fi
[[ $(id -u "$user") != 0 ]]
home=$(getent passwd "$user" | cut -d: -f6)
[[ $home == /home/* && $(realpath "$home") == "$home" ]]
chmod 700 "$home"
if [[ $mode == engine ]]; then
    apt-get -o DPkg::Lock::Timeout=300 install -y --no-install-recommends docker.io docker-buildx
    systemctl enable --now docker
    usermod -aG docker "$user"
else
    if ! command -v docker >/dev/null; then
        echo 'Enable Docker Desktop: Settings > Resources > WSL Integration > this distro.' >&2
        exit 1
    fi
fi
npm install -g @openai/codex@0.171.0
runuser -u "$user" -- env HOME="$home" PATH="$PATH" docker info --format '{{.OSType}}' | grep -qx linux
repo=$home/nezubot-discrod
if [[ ! -e $repo ]]; then
    runuser -u "$user" -- env GIT_TERMINAL_PROMPT=0 bash -c '
        if [[ -n ${GITHUB_TOKEN:-} ]]; then
            export GIT_CONFIG_COUNT=1
            export GIT_CONFIG_KEY_0=http.https://github.com/.extraheader
            export GIT_CONFIG_VALUE_0="Authorization: Basic $(printf "x-access-token:%s" "$GITHUB_TOKEN" | base64 -w0)"
        fi
        timeout 120 git clone --branch "$1" -- "$2" "$3" </dev/null
    ' bash "$branch" "$repository" "$repo"
fi
[[ -d $repo/.git && $(realpath "$repo") == "$repo" ]]
runuser -u "$user" -- git -C "$repo" rev-parse --verify HEAD >/dev/null
unset GITHUB_TOKEN
runuser -u "$user" -- env HOME="$home" PATH="$PATH" bash -c '
    set -euo pipefail
    cd "$HOME/nezubot-discrod"
    umask 077
    python3 -m venv .venv
    .venv/bin/python -m pip install -e ".[dev,rag]"
    mkdir -p "$HOME/.nezubot-discrod" "$HOME/.cache/fastembed"
    chmod 700 "$HOME/.nezubot-discrod"
    if [[ ! -e .env ]]; then cp .env.example .env; fi
    chmod 600 .env
    docker build -t discord-ai-sandbox:local .
    docker build --target selfmod -t discord-ai-selfmod:local .
    codex --version
'
exit 0
