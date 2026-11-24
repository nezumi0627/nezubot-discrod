"""Load owner-approved features, not untrusted Python. AST checks are not a sandbox."""

import importlib
import inspect
import logging
import pkgutil
import re
from functools import wraps

import discord
from discord import app_commands

from nezubot_discrod import features
from nezubot_discrod.sandbox import accepting_work

logger = logging.getLogger(__name__)
BUILTIN_COMMANDS = frozenset(
    {"reset", "prompt", "improve", "search", "wiki", "help", "status", "limit"}
)
MAX_FEATURE_COMMANDS = 20


class FeatureRegistry:
    __slots__ = ("_register",)

    def __init__(self, register):
        self._register = register

    def register(self, *, name, description, callback):
        self._register(name, description, callback)


def guild_callback(callback, guild_id):
    @wraps(callback)
    async def invoke(interaction: discord.Interaction):
        if interaction.guild_id != guild_id or not accepting_work():
            await interaction.response.send_message(
                "この操作は現在利用できません。", ephemeral=True
            )
            return
        try:
            await callback(interaction)
        except Exception:
            logger.exception("Feature command failed")
            sender = (
                interaction.followup.send
                if interaction.response.is_done()
                else (interaction.response.send_message)
            )
            await sender("機能の実行に失敗しました。", ephemeral=True)

    return invoke


def command_registry(tree, guild_id, loaded):
    guild = discord.Object(id=guild_id)
    pending = []

    def register(name, description, callback):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
            raise ValueError("Invalid feature command name")
        names = {command.name for command in tree.get_commands(guild=guild)}
        if name in BUILTIN_COMMANDS or name in names or any(p.name == name for p in pending):
            raise ValueError("Feature command cannot replace an existing command")
        if loaded + len(pending) >= MAX_FEATURE_COMMANDS:
            raise ValueError("Feature command limit exceeded")
        if not isinstance(description, str) or not 1 <= len(description) <= 100:
            raise ValueError("Invalid feature command description")
        if (
            not inspect.iscoroutinefunction(callback)
            or len(inspect.signature(callback).parameters) != 1
        ):
            raise ValueError("Feature callback must be async and accept one interaction")
        pending.append(
            app_commands.Command(
                name=name, description=description, callback=guild_callback(callback, guild_id)
            )
        )

    return FeatureRegistry(register), pending


def load_features(tree, guild_id):
    guild = discord.Object(id=guild_id)
    loaded = 0
    modules = sorted(pkgutil.iter_modules(features.__path__), key=lambda item: item.name)
    for module in modules:
        if module.ispkg:
            continue
        registry, pending = command_registry(tree, guild_id, loaded)
        added = []
        try:
            feature = importlib.import_module(f"{features.__name__}.{module.name}")
            result = feature.setup(registry)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise ValueError("Feature setup must be synchronous")
            for command in pending:
                tree.add_command(command, guild=guild)
                added.append(command.name)
            loaded += len(added)
        except (Exception, SystemExit):
            for name in added:
                tree.remove_command(name, guild=guild)
            logger.exception("Skipping failed feature %s", module.name)
