import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import nezubot_discrod.server_wiki_curator as curator_module
from nezubot_discrod.prompts import SERVER_WIKI_BUILDER_SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_wiki import ServerWiki
from nezubot_discrod.server_wiki_curator import Curator, DevinError, devin_summarize
from tests.test_server_wiki import record
from tests.test_server_wiki import summarize as build_summarize


@pytest.fixture
def wiki(tmp_path):
    return ServerWiki(ServerLog(tmp_path / "private" / "server_log.db", guild_id="123"))


def make_curator(wiki, summarize, **kwargs):
    return Curator(wiki.archive, wiki.path, summarize,
                   backend="devin", model="swe-2-1", **kwargs)


def wiki_db(wiki):
    db = sqlite3.connect(wiki.path)
    db.row_factory = sqlite3.Row
    return db


def verdicts_fake(decisions):
    """Fake curator summarizer: decisions maps summary -> verdict dict."""
    def fake(system, payload):
        assert system == curator_module.CURATOR_SYSTEM
        data = json.loads(payload)
        entries = []
        for entry in data["entries"]:
            decision = decisions.get(entry["summary"], {"verdict": "ok"})
            entries.append({
                "entry": entry["entry"],
                "evidence": entry["sources"],
                "reason": "test",
                **decision,
            })
        return json.dumps({"entries": entries, "new_entries": []}), (0, 0)
    return fake


def test_curator_updates_stale_and_removes_unsupported(wiki):
    wiki.archive.upsert_many([
        record(1, text="I prefer Rust"),
        record(2, text="I maintain the wiki bot"),
        record(5, bot=True, text="Round started"),
    ])
    wiki.build(build_summarize, model="test")
    fake = verdicts_fake({
        "I prefer Rust": {
            "verdict": "stale",
            "corrected_summary": "Prefers Rust and maintains the wiki bot",
        },
        "I maintain the wiki bot": {"verdict": "unsupported", "corrected_summary": None},
    })
    result = make_curator(wiki, fake).run_cycle(max_units=4)
    assert result.updated == 1 and result.removed == 1 and result.failures == 0
    db = wiki_db(wiki)
    summaries = {row["summary"] for row in db.execute(
        "SELECT summary FROM server_wiki_entry WHERE summary <> ''")}
    assert "Prefers Rust and maintains the wiki bot" in summaries
    assert "I maintain the wiki bot" not in summaries
    actions = {row["action"] for row in db.execute("SELECT action FROM server_wiki_curation_log")}
    assert actions == {"update_summary", "remove_entry"}
    log = db.execute(
        "SELECT * FROM server_wiki_curation_log WHERE action = 'update_summary'").fetchone()
    assert log["old_value"] == "I prefer Rust"
    assert log["backend"] == "devin" and log["model"] == "swe-2-1"
    assert json.loads(log["evidence"]) == ["1"]
    state = db.execute("SELECT * FROM server_wiki_curation_state").fetchall()
    assert state and all(row["last_verified_at"] for row in state)


def test_curator_rejects_correction_with_unknown_evidence(wiki):
    wiki.archive.upsert(record(1, text="I prefer Rust"))
    wiki.build(build_summarize, model="test")

    def fake(system, payload):
        data = json.loads(payload)
        entries = [{
            "entry": entry["entry"], "verdict": "stale",
            "corrected_summary": "Fabricated claim",
            "evidence": ["999999"], "reason": "test",
        } for entry in data["entries"]]
        return json.dumps({"entries": entries, "new_entries": []}), (0, 0)

    result = make_curator(wiki, fake).run_cycle(max_units=4)
    assert result.updated == 0 and result.notes >= 1
    db = wiki_db(wiki)
    assert db.execute(
        "SELECT count(*) FROM server_wiki_entry WHERE summary = 'Fabricated claim'"
    ).fetchone()[0] == 0


def test_curator_mechanical_dedup(wiki):
    wiki.archive.upsert(record(1, text="I prefer Rust"))
    wiki.build(build_summarize, model="test")
    db = wiki_db(wiki)
    row = db.execute(
        "SELECT * FROM server_wiki_entry WHERE kind = 'person' AND summary <> '' LIMIT 1"
    ).fetchone()
    with db:
        db.execute(
            "INSERT INTO server_wiki_entry VALUES (?,?,?,?,?,?,?,?)",
            tuple(row[name] for name in
                  ("guild", "channel", "start", "kind", "key", "title", "summary", "sources")))
    result = make_curator(wiki, verdicts_fake({})).run_cycle(max_units=4)
    assert result.deduped == 1
    assert db.execute(
        "SELECT count(*) FROM server_wiki_entry WHERE kind = 'person' AND summary = 'I prefer Rust'"
    ).fetchone()[0] == 1
    assert db.execute(
        "SELECT count(*) FROM server_wiki_curation_log WHERE action = 'dedup_remove'"
    ).fetchone()[0] == 1


def test_curator_person_slice_cursor_rounds(wiki):
    wiki.archive.upsert(record(1, text="I prefer Rust"))
    wiki.build(build_summarize, model="test")
    db = wiki_db(wiki)
    row = db.execute(
        "SELECT * FROM server_wiki_entry WHERE kind = 'person' AND summary <> '' LIMIT 1"
    ).fetchone()
    with db:
        for index in range(11):
            db.execute(
                "INSERT INTO server_wiki_entry VALUES (?,?,?,?,?,?,?,?)",
                (row["guild"], row["channel"], row["start"], "person", "42", "",
                 f"Observation {index}", row["sources"]))
    seen = []

    def fake(system, payload):
        data = json.loads(payload)
        seen.append([entry["entry"] for entry in data["entries"]])
        return json.dumps({"entries": [], "new_entries": []}), (0, 0)

    curator = make_curator(wiki, fake, person_slice=10)
    curator.run_cycle(max_units=4)
    curator.run_cycle(max_units=4)
    person_calls = [call for call in seen if len(call) > 1]
    assert len(person_calls) == 2
    assert not set(person_calls[0]) & set(person_calls[1])
    state = db.execute(
        "SELECT * FROM server_wiki_curation_state WHERE kind = 'person' AND key = '42'"
    ).fetchone()
    assert state["entry_cursor"] == 0 and state["rounds"] == 1


def test_curator_gap_fill_creates_missing_person_page(wiki):
    wiki.archive.upsert_many([
        record(1, text="I prefer Rust"),
        record(2, channel="10", author="77", name="Bob", text="Bob built a kayak"),
        record(3, channel="10", author="77", name="Bob", text="Bob paddles weekly"),
        record(4, channel="10", author="77", name="Bob", text="Bob teaches kayaking"),
    ])
    wiki.build(build_summarize, model="test")
    db = wiki_db(wiki)
    with db:
        db.execute("DELETE FROM server_wiki_entry WHERE kind = 'person' AND key = '77'")

    def fake(system, payload):
        if system == SERVER_WIKI_BUILDER_SYSTEM:
            data = json.loads(payload)
            ids = [row[0] for row in data["messages"]]
            return json.dumps({
                "people": [{"id": "77", "summary": "Builds and paddles kayaks",
                            "sources": ids[:2]}],
                "events": [],
            }), (0, 0)
        return verdicts_fake({})(system, payload)

    curator = make_curator(wiki, fake, gap_min_messages=3)
    result = curator.run_cycle(max_units=1)
    assert result.added == 1
    page = db.execute(
        "SELECT * FROM server_wiki_entry WHERE kind = 'person' AND key = '77' AND summary <> ''"
    ).fetchall()
    assert len(page) == 1 and page[0]["summary"] == "Builds and paddles kayaks"
    assert db.execute(
        "SELECT count(*) FROM server_wiki_curation_log WHERE action = 'add_entry'"
    ).fetchone()[0] == 1
    assert wiki.lookup("person", "77", [10])


def test_curator_dry_run_writes_nothing(wiki):
    wiki.archive.upsert(record(1, text="I prefer Rust"))
    wiki.build(build_summarize, model="test")
    fake = verdicts_fake({
        "I prefer Rust": {"verdict": "stale", "corrected_summary": "Changed"},
    })
    curator = make_curator(wiki, fake, dry_run=True)
    result = curator.run_cycle(max_units=4)
    assert result.updated >= 1
    db = wiki_db(wiki)
    assert db.execute(
        "SELECT count(*) FROM server_wiki_entry WHERE summary = 'Changed'").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM server_wiki_curation_log").fetchone()[0] == 0


class _BridgeHandler(BaseHTTPRequestHandler):
    jobs = {}

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        assert self.headers.get("Authorization") == "Bearer test-token"
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        assert body["model"] == "swe-2-1" and body["prompt"]
        self.jobs["job1"] = {"status": "done", "output": "tail"}
        self._send(201, {"id": "job1", "status": "running", "model": "swe-2-1"})

    def do_GET(self):
        assert self.headers.get("Authorization") == "Bearer test-token"
        self._send(200, self.jobs["job1"])


@pytest.fixture
def bridge(tmp_path):
    (tmp_path / "bridge.env").write_text("BRIDGE_TOKEN=test-token\n")
    sessions = tmp_path / "sessions" / "job1"
    sessions.mkdir(parents=True)
    (sessions / "output.log").write_text('{"entries": []}')
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield tmp_path, server.server_address[1]
    server.shutdown()


def test_devin_summarize_reads_full_output_log(bridge):
    tmp_path, port = bridge
    summarize = devin_summarize(
        model="swe-2-1", timeout=30, bridge_url=f"http://127.0.0.1:{port}",
        env_path=tmp_path / "bridge.env", sessions_dir=tmp_path / "sessions",
        poll_interval=0.01)
    content, usage = summarize("system", "payload")
    assert content == '{"entries": []}' and usage == (0, 0)


def test_devin_summarize_rejects_paid_model():
    with pytest.raises(ValueError):
        curator_module.make_summarize("devin", model="swe-2-1-lightning", timeout=30)


def test_devin_error_is_raised_without_token(tmp_path):
    with pytest.raises(DevinError):
        devin_summarize(
            model="swe-2-1", timeout=30, bridge_url="http://127.0.0.1:9",
            env_path=tmp_path / "missing.env", sessions_dir=tmp_path,
            attempts=1)
