"""v1.25.0: concurrent turns on one Codex app-server.

The client used to serialize every turn on one lock because the
AppServer held a single turn's state. These tests drive the real
AppServer against a fake server subprocess (tests/fixtures/
fake_codex_server.py) whose turns on different threads interleave
on the wire, and pin the properties the refactor must keep:

- two turns on different threads complete concurrently, each with
  only its own answer and its own usage (demux by threadId);
- a tool call is handled with the bindings of the turn its threadId
  names — never another turn's (the security boundary);
- a tool call naming a thread with no live turn is refused without
  executing anything;
- a second turn on a thread that already has one is refused
  client-side, before turn/start;
- the client's turn slots cap how many turns run at once.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nezubot_discrod.codex import AppServer, CodexClient
from nezubot_discrod.config import SandboxSettings, Settings

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_codex_server.py"


async def close_server(server):
    server.process.terminate()
    await server._stop_reader()


async def spawn_server(tmp_path):
    log_path = tmp_path / "fake-server.jsonl"
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(FAKE_SERVER), str(log_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    return AppServer(process, SandboxSettings()), log_path


def server_events(log_path):
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def wait_for_event(log_path, kind, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(event["event"] == kind for event in server_events(log_path)):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"server never logged {kind}")


class LogBinding:
    """A stand-in server_log binding that records its calls."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return self.result


def test_two_turns_interleave_without_crosstalk(tmp_path):
    async def scenario():
        server, log_path = await spawn_server(tmp_path)
        box_a, box_b = {}, {}
        started = time.monotonic()
        answer_a, answer_b = await asyncio.gather(
            server.complete(
                "gpt-6-luna", [], "/empty",
                input_text="marker:A deltas:10 delay:0.1", result_box=box_a,
            ),
            server.complete(
                "gpt-6-luna", [], "/empty",
                input_text="marker:B deltas:10 delay:0.1", result_box=box_b,
            ),
        )
        wall = time.monotonic() - started
        await close_server(server)
        return answer_a, answer_b, box_a, box_b, wall, server_events(log_path)

    answer_a, answer_b, box_a, box_b, wall, events = asyncio.run(scenario())
    assert answer_a == "".join(f"A-{i:02d} " for i in range(10))
    assert answer_b == "".join(f"B-{i:02d} " for i in range(10))
    # Each turn's usage is its own thread's totals, not the other's.
    assert box_a["usage"] == {
        "prompt_tokens": 100, "completion_tokens": 10, "cached_input_tokens": 0}
    assert box_b["usage"] == {
        "prompt_tokens": 100, "completion_tokens": 10, "cached_input_tokens": 0}
    assert box_a["thread_id"] != box_b["thread_id"]
    # The turns really overlapped: both started before either
    # completed, and the pair took far less than the ~2s that running
    # them one after another would take.
    started = [i for i, e in enumerate(events) if e["event"] == "turn_started"]
    completed = [i for i, e in enumerate(events) if e["event"] == "turn_completed"]
    assert len(started) == 2 and len(completed) == 2
    assert max(started) < min(completed)
    assert wall < 1.8


def test_tool_calls_use_the_owning_turns_bindings(tmp_path):
    binding = LogBinding("A-ONLY-RESULT")

    async def scenario():
        server, log_path = await spawn_server(tmp_path)
        answers = await asyncio.gather(
            server.complete(
                "gpt-6-luna", [], "/empty", server_log=binding,
                input_text="marker:A deltas:3 delay:0.1 tool:search_server_log",
            ),
            server.complete(
                "gpt-6-luna", [], "/empty",
                input_text="marker:B deltas:3 delay:0.1 tool:search_server_log",
            ),
        )
        await close_server(server)
        return answers, server_events(log_path)

    (answer_a, answer_b), events = asyncio.run(scenario())
    assert answer_a.startswith("A-00") and answer_b.startswith("B-00")
    # Only turn A had the binding, so only A's call executed — once,
    # against A's binding. Turn B's identical call was answered as a
    # failure by B's own (binding-less) state, never by A's bindings.
    assert binding.calls == [("search_server_log", {"query": "並行テスト"})]
    results = {
        event["threadId"]: event["success"]
        for event in events if event["event"] == "tool_result"
    }
    assert sorted(results.values()) == [False, True]


def test_tool_call_for_an_unknown_thread_is_refused_not_executed(tmp_path):
    runner = Mock(return_value="must never run")

    async def scenario():
        server, log_path = await spawn_server(tmp_path)
        server.runner = runner
        answers = await asyncio.gather(
            server.complete(
                "gpt-6-luna", [], "/empty",
                input_text="marker:A deltas:6 delay:0.1 rogue:0.3",
            ),
            server.complete(
                "gpt-6-luna", [], "/empty",
                input_text="marker:B deltas:6 delay:0.1",
            ),
        )
        # The refusal is answered from the reader, not a turn loop;
        # give it a moment to land in the server log.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(
                event["event"] == "tool_result"
                and event["threadId"] == "rogue-thread"
                for event in server_events(log_path)
            ):
                break
            await asyncio.sleep(0.05)
        await close_server(server)
        return answers, server_events(log_path)

    (answer_a, answer_b), events = asyncio.run(scenario())
    assert answer_a.startswith("A-00") and answer_b.startswith("B-00")
    rogue = [
        event for event in events
        if event["event"] == "tool_result" and event["threadId"] == "rogue-thread"
    ]
    assert len(rogue) == 1 and rogue[0]["success"] is False
    runner.assert_not_called()


def test_second_turn_on_a_busy_thread_is_refused_client_side(tmp_path):
    async def scenario():
        server, log_path = await spawn_server(tmp_path)
        first = asyncio.create_task(server.complete(
            "gpt-6-luna", [], "/empty",
            input_text="marker:A deltas:20 delay:0.1",
        ))
        await wait_for_event(log_path, "turn_started")
        # The fake server numbers its first thread fake-thread-1; a
        # second turn on it must be refused before turn/start.
        with pytest.raises(RuntimeError, match="already has an active turn"):
            await server.complete(
                "gpt-6-luna", [], "/empty", thread_id="fake-thread-1",
                input_text="marker:X deltas:1 delay:0.01",
            )
        answer = await first
        await close_server(server)
        return answer, server_events(log_path)

    answer, events = asyncio.run(scenario())
    assert answer == "".join(f"A-{i:02d} " for i in range(20))
    assert len([e for e in events if e["event"] == "turn_started"]) == 1


def test_turn_slots_cap_client_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_MAX_CONCURRENT_TURNS", "2")
    settings = Settings("", "", "", "gpt-6-luna", SandboxSettings(), provider="codex")
    client = CodexClient(settings)
    assert client.max_concurrent_turns == 2
    running = 0
    peak = 0

    async def recorder(**kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.1)
        running -= 1
        return SimpleNamespace(
            usage=None, sources=(),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None))],
        )

    client._create_persistent = recorder

    async def scenario():
        return await asyncio.gather(*[
            client.create(
                model="gpt-6-luna",
                messages=[{"role": "user", "content": "q"}],
                scope=f"s{i}",
            )
            for i in range(5)
        ])

    results = asyncio.run(scenario())
    assert len(results) == 5
    assert peak == 2


def test_max_concurrent_turns_env_parsing(monkeypatch):
    settings = Settings("", "", "", "gpt-6-luna", SandboxSettings(), provider="codex")
    monkeypatch.delenv("CODEX_MAX_CONCURRENT_TURNS", raising=False)
    assert CodexClient(settings).max_concurrent_turns == 2
    monkeypatch.setenv("CODEX_MAX_CONCURRENT_TURNS", "3")
    assert CodexClient(settings).max_concurrent_turns == 3
    monkeypatch.setenv("CODEX_MAX_CONCURRENT_TURNS", "0")
    assert CodexClient(settings).max_concurrent_turns == 2
    monkeypatch.setenv("CODEX_MAX_CONCURRENT_TURNS", "many")
    assert CodexClient(settings).max_concurrent_turns == 2
