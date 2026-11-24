import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from contextlib import closing, contextmanager
from pathlib import Path

from dotenv import load_dotenv

from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import pre_retrieve
from nezubot_discrod.server_log_tools import LogTools


class ReadOnlyLog(ServerLog):
    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            yield db


async def diagnose(store, prompts, caller_id, channels, *, timeout=0.85, emit=print):
    for index, prompt in enumerate(prompts):
        result = await pre_retrieve(LogTools(store, channels, caller_id=caller_id), prompt,
                                    guild_id=store.guild_id, timeout=timeout)
        diagnostics = getattr(result, "diagnostics", {"chars": len(result),
                                                    "sources": len(result.sources)})
        emit(json.dumps({"index": index, **diagnostics}, ensure_ascii=True, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only automatic context diagnostics")
    parser.add_argument("command", choices=["diagnose"])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--caller-id", required=True)
    parser.add_argument("--guild-id")
    parser.add_argument("--timeout", type=float, default=0.85)
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--warmup", action="store_true",
                        help="Load a dense snapshot before timed prompts (may take seconds)")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--channel-ids", nargs="+",
                       help="Requester- AND reply-audience-safe channel IDs, including age limits")
    scope.add_argument("--all-channels", action="store_true",
                       help="Operator diagnostic only; not live Discord permissions")
    parser.add_argument("prompts", nargs="+")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 2:
        parser.error("--timeout must be greater than zero and at most 2 seconds")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    logging.basicConfig(level=logging.INFO)
    store = ReadOnlyLog(args.db, guild_id=args.guild_id)
    try:
        with store.connect() as db:
            db.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
        channels = args.channel_ids
        if args.all_channels:
            with store.connect() as db:
                channels = [row[0] for row in db.execute(
                    "SELECT DISTINCT channel_id FROM messages",
                )]
            print("Scope: all archived channels; NOT live Discord audience permissions",
                  file=sys.stderr)
        if not args.lexical_only:
            store.enable_dense()
            if args.warmup:
                store._dense.refresh_snapshot()
        asyncio.run(diagnose(store, args.prompts, args.caller_id, channels, timeout=args.timeout))
    except (OSError, sqlite3.Error, ValueError, ImportError) as error:
        parser.exit(1, f"Context diagnostic failed: {type(error).__name__}\n")


if __name__ == "__main__":
    main()
