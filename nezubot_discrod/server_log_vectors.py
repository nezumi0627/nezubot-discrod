import json
import math
import os
import sqlite3
import struct
from collections import Counter
from contextlib import closing
from pathlib import Path

from nezubot_discrod.server_log import ServerLog, text_hash


def _path(store, path):
    return ServerLog(Path(path), checkout=store.checkout)._guarded_path()


def export_vectors(store, path):
    target = _path(store, path)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    count = 0
    with store.connect() as source, closing(sqlite3.connect(target)) as output, output:
        source.execute("BEGIN")
        dimensions = [row[0] for row in source.execute(
            "SELECT DISTINCT dimensions FROM message_vectors WHERE model = ?", (store.model,)
        )]
        if len(dimensions) > 1:
            raise ValueError("Source contains inconsistent vector dimensions")
        output.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE message_vectors (
                message_id TEXT NOT NULL, model TEXT NOT NULL, text_hash TEXT NOT NULL,
                dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY(message_id, model)
            );
        """)
        output.executemany("INSERT INTO metadata VALUES (?, ?)", [
            ("format", "server-log-vectors-v1"), ("document_format", "author-name-colon-text-v1"),
            ("dtype", "float16-le"), ("model", store.model),
            ("max_length", str(store.max_length)),
            ("dimensions", str(dimensions[0] if dimensions else 0)), ("complete", "0"),
        ])
        output.commit()
        cursor = source.execute(
            "SELECT * FROM message_vectors WHERE model = ? ORDER BY message_id", (store.model,)
        )
        while rows := cursor.fetchmany(500):
            output.executemany("INSERT INTO message_vectors VALUES (?, ?, ?, ?, ?)", rows)
            output.commit()
            count += len(rows)
        output.execute("UPDATE metadata SET value = '1' WHERE key = 'complete'")
    return count


def _valid_vector(row, dimensions):
    if (
        type(row["dimensions"]) is not int or not 1 <= row["dimensions"] <= 8192
        or row["dimensions"] != dimensions or not isinstance(row["vector"], bytes)
        or len(row["vector"]) != dimensions * 2
    ):
        return False
    nonzero = False
    for value, in struct.iter_unpack("<e", row["vector"]):
        if not math.isfinite(value):
            return False
        nonzero |= value != 0
    return nonzero


def import_vectors(store, path):
    source_path = _path(store, path)
    if source_path == store.path.resolve():
        raise ValueError("Vector bundle must be separate from the archive")
    counts = Counter(imported=0, unchanged=0, missing=0, stale=0, deleted=0,
                     ineligible=0, model=0, invalid=0)
    with closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as source:
        source.row_factory = sqlite3.Row
        source.execute("BEGIN")
        metadata = dict(source.execute("SELECT key, value FROM metadata"))
        if (
            metadata.get("format") != "server-log-vectors-v1"
            or metadata.get("document_format") != "author-name-colon-text-v1"
            or metadata.get("dtype") != "float16-le" or metadata.get("complete") != "1"
            or metadata.get("model") != store.model
            or metadata.get("max_length") != str(store.max_length)
        ):
            raise ValueError("Incomplete or incompatible vector bundle metadata")
        dimensions = int(metadata["dimensions"])
        if not 0 <= dimensions <= 8192:
            raise ValueError("Invalid vector bundle dimensions")
        with store.connect() as db:
            existing = {row[0] for row in db.execute(
                "SELECT DISTINCT dimensions FROM message_vectors WHERE model = ?", (store.model,)
            )}
        if dimensions and existing and existing != {dimensions}:
            raise ValueError("Vector bundle dimensions differ from archive")
        cursor = source.execute("SELECT * FROM message_vectors ORDER BY message_id")
        while rows := cursor.fetchmany(500):
            valid = []
            for row in rows:
                if row["model"] != store.model:
                    counts["model"] += 1
                elif not _valid_vector(row, dimensions):
                    counts["invalid"] += 1
                else:
                    valid.append(row)
            if not valid:
                continue
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                ids = [row["message_id"] for row in valid]
                current = {row["id"]: row for row in db.execute(
                    "SELECT m.*, v.text_hash AS old_hash, v.vector AS old_vector, "
                    "EXISTS (SELECT 1 FROM deleted_messages d WHERE d.id = m.id) AS tombstoned "
                    "FROM messages m LEFT JOIN message_vectors v "
                    "ON v.message_id = m.id AND v.model = ? "
                    f"WHERE m.id IN ({','.join('?' for _ in ids)})", [store.model, *ids],
                )}
                imported = 0
                for row in valid:
                    message = current.get(row["message_id"])
                    if message is None:
                        counts["missing"] += 1
                    elif message["deleted"] or message["tombstoned"]:
                        counts["deleted"] += 1
                    elif message["is_bot"] or len(message["text"].strip()) < 2:
                        counts["ineligible"] += 1
                    elif row["text_hash"] != text_hash(message):
                        counts["stale"] += 1
                    elif (
                        message["old_hash"] == row["text_hash"]
                        and message["old_vector"] == row["vector"]
                    ):
                        counts["unchanged"] += 1
                    else:
                        db.execute("INSERT OR REPLACE INTO message_vectors VALUES (?, ?, ?, ?, ?)",
                                   tuple(row[key] for key in (
                                       "message_id", "model", "text_hash", "dimensions", "vector"
                                   )))
                        imported += 1
                if imported:
                    db.execute("UPDATE log_state SET generation = generation + 1 WHERE id = 1")
                counts["imported"] += imported
    return dict(counts)


def transfer(store, command, path):
    if command == "export-vectors":
        return json.dumps({"exported": export_vectors(store, path)})
    return json.dumps(import_vectors(store, path), sort_keys=True)
