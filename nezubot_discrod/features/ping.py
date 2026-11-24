async def ping(interaction):
    await interaction.response.send_message("pong")


def setup(registry):
    registry.register(name="ping", description="Reply with pong", callback=ping)
