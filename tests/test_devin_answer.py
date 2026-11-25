import asyncio
import threading
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.devin_answer import (
    DevinAnswerer,
    DevinAnswerError,
    build_devin_prompt,
    devin_failure_message,
)
from nezubot_discrod.model_overrides import (
    clear_model_override,
    resolve_model,
    set_model_override,
)
from nezubot_discrod.server_log import ServerLog

DEFAULT = "gpt-6-luna"


@pytest.fixture()
def store(tmp_path):
    return ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")


@pytest.fixture()
def answerer(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIDGE_TOKEN", "test-token")
    return DevinAnswerer(
        bridge_url="http://127.0.0.1:8799",
        env_path=tmp_path / "no-bridge.env",
        sessions_dir=tmp_path / "sessions",
        timeout=30.0,
        poll_interval=0.01,
    )


def script_requests(answerer, handler):
    """Replace the answerer's HTTP layer with a scripted handler.

    handler(method, path, body) returns (status, dict) or raises.
    """
    calls = []

    def fake_request(method, path, token, body=None):
        assert token == "test-token"
        calls.append((method, path, body))
        return handler(method, path, body)

    answerer._request = fake_request
    return calls


# --- prompt assembly -----------------------------------------------------


def test_build_devin_prompt_orders_system_history_and_latest():
    messages = [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "latest question"},
    ]
    prompt = build_devin_prompt(messages)
    assert prompt.startswith("SYSTEM TEXT")
    assert "ツールは使えません" in prompt  # the operating constraint
    assert "ユーザー: first question" in prompt
    assert "nezubot-discrod: first answer" in prompt
    assert prompt.endswith("latest question")
    assert prompt.index("first question") < prompt.index("latest question")


def test_build_devin_prompt_without_history():
    prompt = build_devin_prompt([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "only question"},
    ])
    assert "これまでの会話" not in prompt
    assert prompt.endswith("only question")


# --- bridge client ---------------------------------------------------------


def test_answer_returns_job_output(answerer):
    states = iter(["running", "done"])

    def handler(method, path, body):
        if method == "POST":
            assert body["model"] == "swe-2-1"
            assert body["prompt"] == "the prompt"
            return 201, {"id": "job1", "status": "running"}
        return 200, {"status": next(states), "output": "  devin says hi  "}

    script_requests(answerer, handler)
    assert asyncio.run(answerer.answer("the prompt")) == "devin says hi"


def test_answer_failed_job_raises_failed_kind(answerer):
    def handler(method, path, body):
        if method == "POST":
            return 201, {"id": "job1", "status": "running"}
        return 200, {"status": "failed", "error": "boom"}

    script_requests(answerer, handler)
    with pytest.raises(DevinAnswerError) as caught:
        asyncio.run(answerer.answer("p"))
    assert caught.value.kind == "failed"


def test_answer_unreachable_bridge_raises_unavailable_kind(answerer):
    def handler(method, path, body):
        raise urllib.error.URLError("connection refused")

    script_requests(answerer, handler)
    with pytest.raises(DevinAnswerError) as caught:
        asyncio.run(answerer.answer("p"))
    assert caught.value.kind == "unavailable"


def test_answer_missing_token_raises_unavailable_kind(answerer, monkeypatch):
    monkeypatch.delenv("BRIDGE_TOKEN", raising=False)
    with pytest.raises(DevinAnswerError) as caught:
        asyncio.run(answerer.answer("p"))
    assert caught.value.kind == "unavailable"


def test_answer_retries_post_on_429_then_succeeds(answerer):
    posts = []

    def handler(method, path, body):
        if method == "POST":
            posts.append(1)
            if len(posts) == 1:
                raise urllib.error.HTTPError(
                    "http://x", 429, "busy", hdrs=None, fp=None
                )
            return 201, {"id": "job1", "status": "running"}
        return 200, {"status": "done", "output": "after the wait"}

    script_requests(answerer, handler)
    assert asyncio.run(answerer.answer("p")) == "after the wait"
    assert len(posts) == 2


def test_answer_polling_past_deadline_raises_timeout_kind(answerer):
    answerer.timeout = 0.2

    def handler(method, path, body):
        if method == "POST":
            return 201, {"id": "job1", "status": "running"}
        return 200, {"status": "running"}

    script_requests(answerer, handler)
    with pytest.raises(DevinAnswerError) as caught:
        asyncio.run(answerer.answer("p"))
    assert caught.value.kind == "timeout"


def test_answer_rejects_paid_lightning_model(answerer):
    with pytest.raises(ValueError):
        asyncio.run(answerer.answer("p", model="swe-2-1-lightning"))


def test_answer_semaphore_caps_concurrency(answerer):
    answerer.max_concurrent = 1
    answerer._slots = None
    answerer._slots_loop = None
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def slow_run(prompt, model, timeout):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return "ok"

    answerer._run_sync = slow_run

    async def two_turns():
        return await asyncio.gather(answerer.answer("a"), answerer.answer("b"))

    assert asyncio.run(two_turns()) == ["ok", "ok"]
    assert state["max"] == 1


def test_failure_messages_are_gentle_and_name_the_default():
    assert "タイムアウト" in devin_failure_message("timeout", DEFAULT)
    assert "混み合って" in devin_failure_message("busy", DEFAULT)
    other = devin_failure_message("unavailable", DEFAULT)
    assert DEFAULT in other and "/model" in other


# --- Assistant routing -----------------------------------------------------


def recording_client():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=None,
            sources=(),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="codex answer", tool_calls=None)
                )
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=create))
        )
    )
    return client, calls


def test_devin_override_routes_turn_to_responder(store):
    set_model_override(store, 111, "swe-2-1")
    prompts = []

    async def responder(prompt):
        prompts.append(prompt)
        return "devin answer"

    client, calls = recording_client()
    ai = Assistant(
        client, DEFAULT, SandboxSettings(),
        server_log=store, devin_responder=responder,
    )
    reply = asyncio.run(ai.reply(1, "hello there", member_id=111))
    assert str(reply) == "devin answer"
    assert reply.usage is None
    assert len(prompts) == 1
    assert "hello there" in prompts[0]
    # The Codex/chat path was never touched for this turn.
    assert calls == []
    # Another member still goes to the normal path.
    reply = asyncio.run(ai.reply(1, "hello there", member_id=222))
    assert str(reply) == "codex answer"
    assert len(calls) == 1 and calls[0]["model"] == DEFAULT
    assert len(prompts) == 1


def test_devin_override_with_self_improve_routes_to_responder(store):
    # Regression (v1.15.0): bot.py passes a self_improve callback on
    # every owner turn, and the Devin branch used to require
    # self_improve to be None — so the owner's swe-2-1 turns fell
    # through to the Codex loop with a non-Codex slug and died on the
    # catalog check. The callback must not gate the Devin path.
    set_model_override(store, 111, "swe-2-1")
    prompts = []

    async def responder(prompt):
        prompts.append(prompt)
        return "devin answer"

    async def self_improve(_request):
        return "n/a"

    client, calls = recording_client()
    ai = Assistant(
        client, DEFAULT, SandboxSettings(),
        server_log=store, devin_responder=responder,
    )
    reply = asyncio.run(
        ai.reply(1, "hello there", member_id=111, self_improve=self_improve)
    )
    assert str(reply) == "devin answer"
    assert len(prompts) == 1
    # swe-2-1 was never handed to the chat/Codex path.
    assert calls == []


def test_devin_turn_prompt_carries_history(store):
    prompts = []

    async def responder(prompt):
        prompts.append(prompt)
        return "second answer"

    client, _calls = recording_client()
    ai = Assistant(
        client, DEFAULT, SandboxSettings(),
        server_log=store, devin_responder=responder,
    )
    asyncio.run(ai.reply(1, "first question", member_id=111))
    set_model_override(store, 111, "swe-2-1")
    reply = asyncio.run(ai.reply(1, "second question", member_id=111))
    assert str(reply) == "second answer"
    assert "first question" in prompts[0]
    assert "codex answer" in prompts[0]
    assert "second question" in prompts[0]


def test_devin_failure_becomes_gentle_answer(store):
    set_model_override(store, 111, "swe-2-1")

    async def responder(prompt):
        raise DevinAnswerError("too slow", kind="timeout")

    client, calls = recording_client()
    ai = Assistant(
        client, DEFAULT, SandboxSettings(),
        server_log=store, devin_responder=responder,
    )
    reply = asyncio.run(ai.reply(1, "hello", member_id=111))
    assert "タイムアウト" in str(reply)
    assert DEFAULT in str(reply)
    assert calls == []


def test_devin_override_without_responder_falls_back_to_default(store):
    set_model_override(store, 111, "swe-2-1")
    client, calls = recording_client()
    ai = Assistant(client, DEFAULT, SandboxSettings(), server_log=store)
    reply = asyncio.run(ai.reply(1, "hello", member_id=111))
    assert str(reply) == "codex answer"
    assert calls[0]["model"] == DEFAULT


def test_devin_reset_restores_codex_path(store):
    set_model_override(store, 111, "swe-2-1")
    prompts = []

    async def responder(prompt):
        prompts.append(prompt)
        return "devin answer"

    client, calls = recording_client()
    ai = Assistant(
        client, DEFAULT, SandboxSettings(),
        server_log=store, devin_responder=responder,
    )
    asyncio.run(ai.reply(1, "hello", member_id=111))
    clear_model_override(store, 111)
    reply = asyncio.run(ai.reply(1, "hello again", member_id=111))
    assert str(reply) == "codex answer"
    assert calls[0]["model"] == DEFAULT
    assert len(prompts) == 1
    assert resolve_model(store, 111, DEFAULT) == DEFAULT


def test_ambient_turn_never_uses_devin(store):
    set_model_override(store, 111, "swe-2-1")
    prompts = []

    async def responder(prompt):
        prompts.append(prompt)
        return "devin answer"

    client, _calls = recording_client()
    ai = Assistant(
        client, DEFAULT, SandboxSettings(),
        server_log=store, devin_responder=responder,
    )
    asyncio.run(ai.reply(1, "ambient remark", member_id=111, ambient=True))
    assert prompts == []
