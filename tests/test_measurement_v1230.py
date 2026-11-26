"""v1.23.0: turn correlation ids, Codex lock-wait logging, the
Devin completion line, and the persistent-thread input budget.

The 2026-11-26 measurement found
the three per-turn log systems could only be paired by file order,
that the Codex client's lock wait was recorded nowhere, that Devin
turns had no completion line, and that reused threads grew a
70k-token input tail. These tests pin the fixes.
"""
import asyncio
import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_codex import FakeProcess, finished, response
from test_codex_persistent import (
    first_turn,
    make_client,
    scripted_server,
    sent_methods,
    turn_inputs,
    turn_script,
    usage_event,
)
from test_server_log import record

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import AppServer, CodexClient
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.devin_answer import DevinAnswerError
from nezubot_discrod.prompts import SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import pre_retrieve
from nezubot_discrod.server_log_tools import LogTools
from nezubot_discrod.turn_context import CURRENT_TURN_ID, current_turn_id, start_turn

MESSAGES1 = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "q1"},
]
MESSAGES2 = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
]


def fresh_thread_turn(first_id, text, totals, thread="thread2"):
    """Script for a turn that must start a new thread (thread/start
    + turn/start round trips)."""
    return [
        response(first_id, {
            "thread": {"id": thread},
            "approvalPolicy": "never",
            "sandbox": {"type": "readOnly"},
        }),
        response(first_id + 1, {"turn": {"id": f"turn{first_id}"}}),
        usage_event(*totals, thread=thread),
        *finished(text),
    ]


def test_turn_id_reaches_retrieval_diagnostics(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    store.upsert_many([
        record(1, channel="10", text="Rust history"),
        record(2, channel="10", text="Rust engines"),
    ])
    log = LogTools(store, [10], caller_id=20)
    token = CURRENT_TURN_ID.set("turn-abc123")
    try:
        result = asyncio.run(pre_retrieve(log, "Rustについて", timeout=5))
    finally:
        CURRENT_TURN_ID.reset(token)
    assert result.diagnostics["turn_id"] == "turn-abc123"
    # Outside a turn the field is present but empty, so log joins
    # never mistake a CLI run for a turn.
    result = asyncio.run(pre_retrieve(log, "Rustについて", timeout=5))
    assert result.diagnostics["turn_id"] is None


def test_codex_logs_carry_turn_id_and_lock_wait(tmp_path, caplog):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        turn_script(5, "a2", (1500, 130, 1200)),
    )
    client, _ = make_client(tmp_path, [server])

    async def scenario():
        start_turn()
        turn_id = current_turn_id()
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")
        return turn_id

    with caplog.at_level(logging.INFO, logger="nezubot_discrod.codex"):
        turn_id = asyncio.run(scenario())
    waits = [r.getMessage() for r in caplog.records
             if "Codex client lock wait" in r.getMessage()]
    assert len(waits) == 2
    assert all(f"turn_id={turn_id}" in line for line in waits)
    assert all("waiters=0" in line for line in waits)
    finished_lines = [r.getMessage() for r in caplog.records
                      if "Codex turn finished" in r.getMessage()]
    assert len(finished_lines) == 2
    assert all(f"turn_id={turn_id}" in line for line in finished_lines)


def test_lock_wait_is_measured_while_turn_slots_are_saturated(tmp_path, caplog):
    server = scripted_server(first_turn("a1", (1000, 100, 0)))
    client, _ = make_client(tmp_path, [server])

    async def scenario():
        # Saturate the turn slots: the next create() must queue for
        # one, and the queueing is what the wait line measures. (The
        # pre-v1.24.0 form of this test held the client's single lock,
        # which is what the slots replaced.)
        for _ in range(client.max_concurrent_turns):
            await client._turn_slots.acquire()
        task = asyncio.create_task(
            client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        )
        await asyncio.sleep(0.1)
        assert client._lock_waiters == 1
        client._turn_slots.release()
        result = await task
        for _ in range(client.max_concurrent_turns - 1):
            client._turn_slots.release()
        return result

    with caplog.at_level(logging.INFO, logger="nezubot_discrod.codex"):
        result = asyncio.run(scenario())
    assert result.choices[0].message.content == "a1"
    waits = [r.getMessage() for r in caplog.records
             if "Codex client lock wait" in r.getMessage()]
    assert len(waits) == 1
    waited_ms = float(re.search(r"lock wait ([\d.]+)ms", waits[0]).group(1))
    assert waited_ms >= 50


def test_input_budget_retires_an_accumulated_thread(tmp_path, caplog):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        fresh_thread_turn(5, "a2", (700, 70, 0)),
    )
    client, _ = make_client(tmp_path, [server])
    expected_estimate = 1000 + 100 + len(
        json.dumps([MESSAGES2[-1]], ensure_ascii=False)
    )
    client.thread_input_budget = expected_estimate - 1

    async def scenario():
        start_turn()
        turn_id = current_turn_id()
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        second = await client.create(
            model="gpt-6-luna", messages=MESSAGES2, scope="s1")
        return turn_id, second

    with caplog.at_level(logging.INFO, logger="nezubot_discrod.codex"):
        turn_id, second = asyncio.run(scenario())
    assert second.choices[0].message.content == "a2"
    # The second turn started a fresh thread and re-seeded the full
    # local history instead of extending the retired one.
    assert sent_methods(server.process).count("thread/start") == 2
    assert json.loads(turn_inputs(server.process)[1]) == MESSAGES2
    retirements = [r.getMessage() for r in caplog.records
                   if "retired by input budget" in r.getMessage()]
    assert len(retirements) == 1
    assert f"turn_id={turn_id}" in retirements[0]
    assert f"estimated_input={expected_estimate}" in retirements[0]
    assert f"budget={expected_estimate - 1}" in retirements[0]
    assert "last_input=1000" in retirements[0]


def test_input_budget_leaves_a_normal_thread_alone(tmp_path):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        turn_script(5, "a2", (1500, 130, 1200)),
    )
    client, _ = make_client(tmp_path, [server])
    assert client.thread_input_budget == 40_000

    async def scenario():
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        return await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")

    second = asyncio.run(scenario())
    assert second.choices[0].message.content == "a2"
    assert sent_methods(server.process).count("thread/start") == 1
    # The entry now carries the latest turn's per-turn usage, which
    # is what the next turn's budget estimate starts from.
    (entry,) = client._threads.values()
    assert entry.last_input_tokens == 500
    assert entry.last_output_tokens == 30


def test_input_budget_zero_disables_the_gate(tmp_path):
    server = scripted_server(
        first_turn("a1", (90000, 100, 0)),
        turn_script(5, "a2", (90500, 130, 80000)),
    )
    client, _ = make_client(tmp_path, [server])
    client.thread_input_budget = 0

    async def scenario():
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        return await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")

    second = asyncio.run(scenario())
    assert second.usage["prompt_tokens"] == 500
    assert sent_methods(server.process).count("thread/start") == 1


def test_input_budget_env_override(monkeypatch):
    settings = Settings("", "", "", "gpt-6-luna", SandboxSettings(), provider="codex")
    monkeypatch.setenv("CODEX_THREAD_INPUT_BUDGET_TOKENS", "12345")
    assert CodexClient(settings).thread_input_budget == 12345
    monkeypatch.setenv("CODEX_THREAD_INPUT_BUDGET_TOKENS", "0")
    assert CodexClient(settings).thread_input_budget == 0
    monkeypatch.setenv("CODEX_THREAD_INPUT_BUDGET_TOKENS", "not-a-number")
    assert CodexClient(settings).thread_input_budget == 40_000
    monkeypatch.setenv("CODEX_THREAD_INPUT_BUDGET_TOKENS", "-1")
    assert CodexClient(settings).thread_input_budget == 40_000
    monkeypatch.delenv("CODEX_THREAD_INPUT_BUDGET_TOKENS")
    assert CodexClient(settings).thread_input_budget == 40_000


def _devin_assistant(responder):
    return Assistant(
        SimpleNamespace(), "swe-2-1", SandboxSettings(),
        devin_responder=responder,
    )


def test_devin_turn_logs_completion_with_the_turns_id(caplog):
    assistant = _devin_assistant(AsyncMock(return_value="答えです"))
    with caplog.at_level(logging.INFO, logger="nezubot_discrod.assistant"):
        reply = asyncio.run(assistant.reply(7, "質問です"))
    assert reply == "答えです"
    devin_lines = [r.getMessage() for r in caplog.records
                   if "Devin turn finished" in r.getMessage()]
    assistant_lines = [r.getMessage() for r in caplog.records
                       if "Assistant turn finished" in r.getMessage()]
    assert len(devin_lines) == 1 and len(assistant_lines) == 1
    turn_id = re.search(r"turn_id=(\w+)", devin_lines[0]).group(1)
    assert "status=ok" in devin_lines[0]
    assert "model=swe-2-1" in devin_lines[0]
    # The same id ties the Devin line to the turn's completion line.
    assert f"turn_id={turn_id}" in assistant_lines[0]


def test_devin_turn_logs_failures_with_their_kind(caplog):
    async def fail(prompt):
        raise DevinAnswerError("bridge stayed busy", kind="busy")

    assistant = _devin_assistant(fail)
    with caplog.at_level(logging.INFO, logger="nezubot_discrod.assistant"):
        reply = asyncio.run(assistant.reply(7, "質問です"))
    assert "混み合って" in reply
    devin_lines = [r.getMessage() for r in caplog.records
                   if "Devin turn finished" in r.getMessage()]
    assert len(devin_lines) == 1
    assert "status=error:busy" in devin_lines[0]
    assert re.search(r"turn_id=\w+", devin_lines[0])


def test_app_server_turn_finished_line_has_turn_id(caplog):
    # AppServer.complete is also driven outside CodexClient (the
    # per-call path); its completion line reads the ambient turn id.
    from test_codex import conversation

    process = FakeProcess(conversation(*finished("done")))
    server = AppServer(process, SandboxSettings())
    token = CURRENT_TURN_ID.set("turn-xyz789")
    try:
        with caplog.at_level(logging.INFO, logger="nezubot_discrod.codex"):
            answer = asyncio.run(server.complete("gpt-6-luna", MESSAGES1, "/empty"))
    finally:
        CURRENT_TURN_ID.reset(token)
    assert answer == "done"
    lines = [r.getMessage() for r in caplog.records
             if "Codex turn finished" in r.getMessage()]
    assert len(lines) == 1
    assert "turn_id=turn-xyz789" in lines[0]
