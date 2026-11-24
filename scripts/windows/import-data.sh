#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
[[ $(id -u) != 0 && $HOME == /home/* && $(realpath "$HOME") == "$HOME" ]]
cd "$HOME/nezubot-discrod"
.venv/bin/python - "${1:?Usage: bash import-data.sh /home/nezubot/data.tar.gz}" <<'PY'
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

home = Path.home().resolve()
archive = Path(sys.argv[1]).resolve(strict=True)
if not archive.is_relative_to(home):
    raise SystemExit("Archive must be copied into the WSL bot home first")
target = home / ".nezubot-discrod"
if target.is_symlink() or (target.exists() and any(target.iterdir())):
    raise SystemExit("Stop the bot and move existing data to a backup first; refusing overwrite")
allowed = {db + suffix for db in ("wiki.db", "server_log.db")
           for suffix in ("", "-wal", "-shm", "-journal")}
staging = Path(tempfile.mkdtemp(prefix=".data-import-", dir=home))
try:
    count = 0
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            if member.name.rstrip("/") == ".nezubot-discrod" and member.isdir():
                continue
            parts = member.name.split("/")
            if (len(parts) != 2 or parts[0] != ".nezubot-discrod"
                    or parts[1] not in allowed or not member.isfile()):
                raise ValueError(f"Unexpected archive entry: {member.name!r}")
            with source.extractfile(member) as reader:
                fd = os.open(staging / parts[1], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as writer:
                    shutil.copyfileobj(reader, writer)
            count += 1
    if not count:
        raise ValueError("Archive contains no databases")
    if target.exists():
        target.rmdir()
    staging.rename(target)
    print(f"Imported {count} SQLite files into {target}")
finally:
    if staging.exists():
        shutil.rmtree(staging)
PY
