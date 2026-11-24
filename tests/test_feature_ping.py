import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from nezubot_discrod.features.ping import setup


def test_ping_registers_and_replies_pong():
    registry = Mock()
    setup(registry)
    command = registry.register.call_args.kwargs
    assert command["name"] == "ping"
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    asyncio.run(command["callback"](interaction))
    interaction.response.send_message.assert_awaited_once_with("pong", ephemeral=True)
