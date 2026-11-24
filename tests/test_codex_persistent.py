"""Plan A: persistent app-server + per-scope thread reuse.

The fakes mirror test_codex: an AppServer over a scripted FakeProcess.
Client-level tests stub prepare_runtime/direct_catalog/_verify_version
and replace client.server with a transport that hands out one scripted
server per spawn, so spawn counts and request sequences are visible.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_codex import FakeProcess, conversation, finished, packet, response

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import AppServer, CodexClient
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.prompts import SYSTEM

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


def usage_event(inputs, outputs, cached, thread="thread1"):
    return packet("thread/tokenUsage/updated", {
        "threadId": thread,
        "tokenUsage": {"total": {
            "inputTokens": inputs, "outputTokens": outputs,
            "cachedInputTokens": cached,
        }},
    })


def turn_script(first_id, text, totals):
    """Responses + events for one turn on an already-initialized server
    whose thread already exists (a bare turn/start round trip)."""
    return [
        response(first_id, {"turn": {"id": f"turn{first_id}"}}),
        usage_event(*totals),
        *finished(text),
    ]


def make_client(tmp_path, servers):
    """A CodexClient whose server spawns come from `servers` (a list of
    AppServer objects or exceptions), in order."""
    settings = Settings("", "", "", "gpt-6-luna", SandboxSettings(), provider="codex")
    client = CodexClient(settings)
    client.prepare_runtime = Mock(return_value=tmp_path)
    client.direct_catalog = AsyncMock(return_value=tmp_path / "catalog.json")
    client._verify_version = AsyncMock()
    spawns = []

    def transport(cwd, env, catalog=None):
        index = len(spawns)
        spawns.append((cwd, catalog))

        @asynccontextmanager
        async def manager():
            item = servers[index]
            if isinstance(item, BaseException):
                raise item
            yield item

        return manager()

    client.server = transport
    return client, spawns


def scripted_server(*turns):
    """One FakeProcess-backed AppServer scripted for a persistent
    session: initialize + account, then the given per-turn scripts.
    The first turn script must include its thread/start response."""
    events = [response(1, {}), response(2, {"account": {"type": "chatgpt"}})]
    for turn in turns:
        events.extend(turn)
    return AppServer(FakeProcess(events), SandboxSettings())


def first_turn(text, totals, thread="thread1"):
    return [
        response(3, {
            "thread": {"id": thread},
            "approvalPolicy": "never",
            "sandbox": {"type": "readOnly"},
        }),
        response(4, {"turn": {"id": "turn1"}}),
        usage_event(*totals, thread=thread),
        *finished(text),
    ]


def sent_methods(process):
    return [p.get("method") for p in process.sent if "method" in p]


def turn_inputs(process):
    return [
        p["params"]["input"][0]["text"]
        for p in process.sent if p.get("method") == "turn/start"
    ]


def test_persistent_reuses_thread_and_sends_only_the_new_message(tmp_path):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        turn_script(5, "a2", (1500, 130, 1200)),
    )
    client, spawns = make_client(tmp_path, [server])

    async def scenario():
        first = await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        second = await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")
        return first, second

    first, second = asyncio.run(scenario())
    assert first.choices[0].message.content == "a1"
    assert second.choices[0].message.content == "a2"
    assert len(spawns) == 1  # one persistent process for both turns
    methods = sent_methods(server.process)
    assert methods.count("initialize") == 1
    assert methods.count("thread/start") == 1
    assert methods.count("turn/start") == 2
    inputs = turn_inputs(server.process)
    assert json.loads(inputs[0]) == MESSAGES1  # cold start seeds everything
    assert json.loads(inputs[1]) == [MESSAGES2[-1]]  # then only the new message
    assert first.usage == {"prompt_tokens": 1000, "completion_tokens": 100,
                            "cached_input_tokens": 0}
    # Per-turn usage is the delta of the thread's cumulative totals.
    assert second.usage == {"prompt_tokens": 500, "completion_tokens": 30,
                             "cached_input_tokens": 1200}


def test_persistent_config_change_starts_a_fresh_thread(tmp_path):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        [
            response(5, {
                "thread": {"id": "thread2"},
                "approvalPolicy": "never",
                "sandbox": {"type": "readOnly"},
            }),
            response(6, {"turn": {"id": "turn2"}}),
            usage_event(700, 70, 0, thread="thread2"),
            *finished("a2"),
        ],
    )
    client, _ = make_client(tmp_path, [server])

    async def scenario():
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        # A different model is a different thread configuration.
        return await client.create(model="gpt-6.2-sol", messages=MESSAGES2, scope="s1")

    second = asyncio.run(scenario())
    assert second.choices[0].message.content == "a2"
    methods = sent_methods(server.process)
    assert methods.count("thread/start") == 2
    # A fresh thread is seeded with the full local history again.
    assert json.loads(turn_inputs(server.process)[1]) == MESSAGES2


def test_persistent_turn_cap_retires_the_thread(tmp_path, monkeypatch):
    monkeypatch.setattr("nezubot_discrod.codex.THREAD_MAX_TURNS", 2)
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        turn_script(5, "a2", (1500, 130, 1200)),
        [
            response(6, {
                "thread": {"id": "thread2"},
                "approvalPolicy": "never",
                "sandbox": {"type": "readOnly"},
            }),
            response(7, {"turn": {"id": "turn3"}}),
            usage_event(600, 60, 0, thread="thread2"),
            *finished("a3"),
        ],
    )
    client, _ = make_client(tmp_path, [server])
    messages3 = [*MESSAGES2, {"role": "assistant", "content": "a2"},
                 {"role": "user", "content": "q3"}]

    async def scenario():
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")
        return await client.create(model="gpt-6-luna", messages=messages3, scope="s1")

    third = asyncio.run(scenario())
    assert third.choices[0].message.content == "a3"
    assert sent_methods(server.process).count("thread/start") == 2
    assert json.loads(turn_inputs(server.process)[2]) == messages3


def test_forget_threads_forces_a_fresh_thread(tmp_path):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        [
            response(5, {
                "thread": {"id": "thread2"},
                "approvalPolicy": "never",
                "sandbox": {"type": "readOnly"},
            }),
            response(6, {"turn": {"id": "turn2"}}),
            usage_event(700, 70, 0, thread="thread2"),
            *finished("a2"),
        ],
    )
    client, _ = make_client(tmp_path, [server])

    async def scenario():
        await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        client.forget_threads(["s1"])
        return await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")

    asyncio.run(scenario())
    assert sent_methods(server.process).count("thread/start") == 2
    assert json.loads(turn_inputs(server.process)[1]) == MESSAGES2


def test_mid_turn_death_falls_back_to_a_per_call_server(tmp_path):
    # The persistent server's script ends after turn 1: the next read is
    # EOF, which is exactly how a dead process presents.
    persistent = scripted_server(first_turn("a1", (1000, 100, 0)))
    legacy = AppServer(FakeProcess(conversation(*finished("legacy"))), SandboxSettings())
    client, spawns = make_client(tmp_path, [persistent, legacy])

    async def scenario():
        first = await client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1")
        second = await client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1")
        return first, second

    first, second = asyncio.run(scenario())
    assert first.choices[0].message.content == "a1"
    assert second.choices[0].message.content == "legacy"
    assert len(spawns) == 2
    assert client._persistent is None  # dropped, restarted on a later turn
    # The fallback turn replays the full local history on its fresh thread.
    assert json.loads(turn_inputs(legacy.process)[0]) == MESSAGES2


def test_start_failure_falls_back_to_a_per_call_server(tmp_path):
    legacy = AppServer(FakeProcess(conversation(*finished("legacy"))), SandboxSettings())
    client, spawns = make_client(tmp_path, [OSError("spawn failed"), legacy])

    result = asyncio.run(client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1"))
    assert result.choices[0].message.content == "legacy"
    assert len(spawns) == 2


def test_concurrent_turns_share_one_thread_without_interleaving(tmp_path):
    server = scripted_server(
        first_turn("a1", (1000, 100, 0)),
        turn_script(5, "a2", (1500, 130, 1200)),
    )
    client, spawns = make_client(tmp_path, [server])

    async def scenario():
        return await asyncio.gather(
            client.create(model="gpt-6-luna", messages=MESSAGES1, scope="s1"),
            client.create(model="gpt-6-luna", messages=MESSAGES2, scope="s1"),
        )

    first, second = asyncio.run(scenario())
    assert (first.choices[0].message.content, second.choices[0].message.content) == ("a1", "a2")
    assert len(spawns) == 1
    assert sent_methods(server.process).count("thread/start") == 1


def test_rate_limits_prefers_the_persistent_server(tmp_path):
    client, spawns = make_client(tmp_path, [])
    request = AsyncMock(return_value={"rateLimits": {"primary": {}}})
    client._persistent = SimpleNamespace(
        process=SimpleNamespace(returncode=None), request=request, tool_free=False,
    )
    result = asyncio.run(client.rate_limits())
    assert result == {"rateLimits": {"primary": {}}}
    request.assert_awaited_once_with("account/rateLimits/read", {})
    assert spawns == []


def test_complete_with_thread_id_skips_thread_start_and_reports_delta_usage():
    process = FakeProcess(turn_script(1, "a2", (1500, 130, 1200)))
    server = AppServer(process, SandboxSettings())
    server.initialized = True  # persistent servers initialize once, up front
    answer = asyncio.run(server.complete(
        "gpt-6-luna", MESSAGES2, "/empty",
        thread_id="thread1",
        input_text=json.dumps([MESSAGES2[-1]], ensure_ascii=False),
        usage_baseline={"inputTokens": 1000, "outputTokens": 100, "cachedInputTokens": 0},
    ))
    assert answer == "a2"
    assert sent_methods(process) == ["turn/start"]
    assert turn_inputs(process) == [json.dumps([MESSAGES2[-1]], ensure_ascii=False)]
    assert server.usage == {"prompt_tokens": 500, "completion_tokens": 30,
                            "cached_input_tokens": 1200}
    assert server.thread_totals["inputTokens"] == 1500
    # Bindings never stay live between turns on a persistent server.
    assert server.wiki is None and server.server_log is None and server.web is None


def test_assistant_passes_scope_and_reset_forgets_threads():
    complete = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=None, sources=(),
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        forget_threads=Mock(),
    )
    assistant = Assistant(
        client, "test", SandboxSettings(),
        wiki_completion=complete,
    )
    asyncio.run(assistant.reply(7, "change", self_improve=AsyncMock()))
    assert complete.call_args.kwargs["scope"] == 7
    assistant.reset(7)
    client.forget_threads.assert_called_once_with([7])


def test_no_scope_keeps_the_per_call_path(tmp_path):
    legacy = AppServer(FakeProcess(conversation(*finished("legacy"))), SandboxSettings())
    client, spawns = make_client(tmp_path, [legacy])
    result = asyncio.run(client.create(model="gpt-6-luna", messages=MESSAGES1))
    assert result.choices[0].message.content == "legacy"
    assert len(spawns) == 1
    assert client._persistent is None


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "x"}]])
def test_ineligible_message_shapes_keep_the_per_call_path(tmp_path, messages):
    legacy = AppServer(FakeProcess(conversation(*finished("legacy"))), SandboxSettings())
    client, _ = make_client(tmp_path, [legacy])
    result = asyncio.run(client.create(model="gpt-6-luna", messages=messages, scope="s1"))
    assert result.choices[0].message.content == "legacy"
    assert client._persistent is None
