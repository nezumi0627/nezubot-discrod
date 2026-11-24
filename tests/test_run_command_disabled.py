import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.sandbox import run_command_enabled


def message(content=None, calls=None):
    data = {"role": "assistant", "content": content}
    if calls:
        data["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in calls
        ]
    return SimpleNamespace(content=content, tool_calls=calls, model_dump=lambda **_: data)


def tool():
    return SimpleNamespace(
        id="call1", function=SimpleNamespace(name="run_command", arguments='{"command":"echo hi"}')
    )


def client_for(*messages):
    snapshots = []

    async def create(**kwargs):
        snapshots.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=messages[len(snapshots) - 1])])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create)))
    )
    return client, snapshots


def test_run_command_not_offered_or_executed_when_disabled():
    client, sent = client_for(message(calls=[tool()]), message("recovered"))
    runner = Mock(return_value="should not run")
    ai = Assistant(client, "test", SandboxSettings(), runner=runner, run_command_enabled=False)
    assert asyncio.run(ai.reply(1, "hello")) == "recovered"
    assert sent[0]["tools"] == []
    runner.assert_not_called()
    assert "Tool error" in sent[1]["messages"][-1]["content"]


def test_run_command_enabled_env(monkeypatch):
    monkeypatch.delenv("RUN_COMMAND_ENABLED", raising=False)
    assert run_command_enabled() is True
    monkeypatch.setenv("RUN_COMMAND_ENABLED", "false")
    assert run_command_enabled() is False
    monkeypatch.setenv("RUN_COMMAND_ENABLED", "0")
    assert run_command_enabled() is False
    monkeypatch.setenv("RUN_COMMAND_ENABLED", "yes")
    assert run_command_enabled() is True
