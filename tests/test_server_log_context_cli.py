import json
import sqlite3

import pytest
from test_server_log import record

from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context_cli import ReadOnlyLog, main


def test_context_cli_read_only_metrics_and_explicit_scope(tmp_path, capsys):
    path = tmp_path / "archive" / "server_log.db"
    store = ServerLog(path)
    store.upsert_many([record(1, text="Rust visible"), record(2, channel="99", text="secret")])
    before = path.read_bytes()
    main(["diagnose", "--db", str(path), "--caller-id", "20", "--lexical-only",
          "--channel-ids", "10", "--", "Rust", "my Rust"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 2 and all(row["chars"] > 0 for row in rows)
    assert all("people" in row["stages_ms"] for row in rows)
    assert all(row["chunks"] and row["elapsed_ms"] >= 0 for row in rows)
    assert path.read_bytes() == before
    with (ReadOnlyLog(path).connect() as db,
          pytest.raises(sqlite3.OperationalError, match="readonly")):
        db.execute("DELETE FROM messages")


def test_context_cli_empty_and_missing_archive(tmp_path, capsys):
    path = tmp_path / "archive" / "server_log.db"
    with ServerLog(path).connect():
        pass
    main(["diagnose", "--db", str(path), "--caller-id", "20", "--lexical-only",
          "--all-channels", "Rust"])
    row = json.loads(capsys.readouterr().out)
    assert row["chars"] == 0 and row["chunks"] == {}
    missing = path.with_name("missing.db")
    with pytest.raises(SystemExit) as error:
        main(["diagnose", "--db", str(missing), "--caller-id", "20", "--lexical-only",
              "--all-channels", "Rust"])
    assert error.value.code == 1 and not missing.exists()


def test_context_cli_rejects_unbounded_deadline():
    with pytest.raises(SystemExit) as error:
        main(["diagnose", "--db", "unused.db", "--caller-id", "20", "--timeout", "3",
              "--all-channels", "Rust"])
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid"])
def test_context_cli_explicit_scope_rejects_unopenable_archive(tmp_path, capsys, kind):
    path = tmp_path / "archive.db"
    if kind == "directory":
        path.mkdir()
    elif kind == "invalid":
        path.write_text("not a SQLite archive")
    with pytest.raises(SystemExit) as error:
        main(["diagnose", "--db", str(path), "--caller-id", "20", "--lexical-only",
              "--channel-ids", "10", "--", "Rust"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out and "Context diagnostic failed:" in captured.err
    if kind == "missing":
        assert not path.exists()
