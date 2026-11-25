#!/bin/bash
# Supervised ws-relay: restart on exit, log to relay.log.
cd "$(dirname "$0")/.." || exit 1
ulimit -n 8192 2>/dev/null
while true; do
    python3 host-tools/ws-relay.py 18080 >> relay.log 2>&1
    echo "$(date -Is) relay exited ($?), restarting" >> relay.log
    sleep 2
done
