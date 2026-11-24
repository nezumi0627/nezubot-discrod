async def ping(interaction):
    await interaction.response.send_message("pong", ephemeral=True)


def setup(registry):
    registry.register(name="ping", description="Reply with pong", callback=ping)
