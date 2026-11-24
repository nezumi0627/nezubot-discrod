import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord
import pytest
from discord import app_commands

from nezubot_discrod.feature_loader import load_features
from nezubot_discrod.policy import (
    ALLOWED_FEATURE_IMPORTS,
    BLOCKED_FEATURE_ATTRIBUTES,
    BLOCKED_FEATURE_IMPORTS,
    BLOCKED_FEATURE_NAMES,
    PolicyViolation,
    protected_path,
    validate_feature_source,
    validate_workspace,
)


@pytest.fixture
def tree():
    return app_commands.CommandTree(discord.Client(intents=discord.Intents.none()))


async def callback(interaction):
    await interaction.response.send_message("ok")


def load_fake(tree, modules):
    discoveries = [SimpleNamespace(name=name, ispkg=False) for name in reversed(modules)]
    with (
        patch("nezubot_discrod.feature_loader.pkgutil.iter_modules", return_value=discoveries),
        patch("nezubot_discrod.feature_loader.importlib.import_module") as importer,
    ):

        def import_module(name):
            value = modules[name.rsplit(".", 1)[1]]
            if isinstance(value, BaseException):
                raise value
            return SimpleNamespace(setup=value)

        importer.side_effect = import_module
        load_features(tree, 123)
        return [call.args[0].rsplit(".", 1)[1] for call in importer.call_args_list]


def register(name):
    return lambda registry: registry.register(name=name, description="test", callback=callback)


def test_loader_is_sorted_and_registers_only_in_configured_guild(tree):
    assert load_fake(tree, {"a": register("one"), "b": register("two")}) == ["a", "b"]
    assert tree.get_commands() == []
    assert tree.get_commands(guild=discord.Object(id=999)) == []
    assert [cmd.name for cmd in tree.get_commands(guild=discord.Object(id=123))] == ["one", "two"]


def test_empty_loader(tree):
    load_fake(tree, {})
    assert tree.get_commands(guild=discord.Object(id=123)) == []


@pytest.mark.parametrize("name", ["reset", "prompt", "improve", "search", "INVALID", "", "x" * 33])
def test_registry_rejects_reserved_and_invalid_names(tree, name, caplog):
    load_fake(tree, {"bad": register(name)})
    assert tree.get_commands(guild=discord.Object(id=123)) == []
    assert "Skipping failed feature bad" in caplog.text


@pytest.mark.parametrize(
    "description,handler", [("", callback), ("x" * 101, callback), ("ok", Mock())]
)
def test_registry_validates_description_and_async_callback(tree, description, handler):
    def setup(registry):
        registry.register(name="bad", description=description, callback=handler)

    load_fake(tree, {"bad": setup})
    assert tree.get_commands(guild=discord.Object(id=123)) == []


def test_registry_rejects_callback_options(tree):
    async def extra(interaction, option):
        pass

    load_fake(
        tree,
        {"bad": lambda registry: registry.register(name="bad", description="test", callback=extra)},
    )
    assert tree.get_commands(guild=discord.Object(id=123)) == []


def test_duplicates_and_limit_cannot_replace_commands(tree):
    with patch("nezubot_discrod.feature_loader.MAX_FEATURE_COMMANDS", 1):
        load_fake(tree, {"a": register("one"), "b": register("one"), "c": register("two")})
    assert [cmd.name for cmd in tree.get_commands(guild=discord.Object(id=123))] == ["one"]


def test_failing_import_or_setup_rolls_back_and_continues(tree, caplog):
    def broken(registry):
        register("discarded")(registry)
        raise RuntimeError("setup failed")

    load_fake(tree, {"a": ImportError("broken"), "b": broken, "c": register("good")})
    assert [cmd.name for cmd in tree.get_commands(guild=discord.Object(id=123))] == ["good"]
    assert "Skipping failed feature a" in caplog.text
    assert "Skipping failed feature b" in caplog.text


def test_async_setup_is_rejected(tree):
    async def setup(registry):
        register("bad")(registry)

    load_fake(tree, {"bad": setup})
    assert tree.get_commands(guild=discord.Object(id=123)) == []


@pytest.mark.parametrize("guild", [None, 999, 123])
def test_callback_enforces_guild_at_dispatch(tree, guild):
    seen = AsyncMock()

    async def handler(interaction):
        await seen(interaction)

    load_fake(
        tree,
        {
            "feature": lambda registry: registry.register(
                name="test", description="test", callback=handler
            )
        },
    )
    interaction = SimpleNamespace(
        guild_id=guild, response=SimpleNamespace(send_message=AsyncMock())
    )
    command = tree.get_command("test", guild=discord.Object(id=123))
    with patch("nezubot_discrod.feature_loader.accepting_work", return_value=True):
        asyncio.run(command.callback(interaction))
    assert seen.await_count == (1 if guild == 123 else 0)
    if guild != 123:
        assert interaction.response.send_message.call_args.kwargs["ephemeral"]


@pytest.mark.parametrize("acknowledged", [False, True])
def test_callback_errors_are_isolated(tree, acknowledged, caplog):
    async def broken(interaction):
        raise RuntimeError("callback failed")

    load_fake(
        tree,
        {
            "feature": lambda registry: registry.register(
                name="test", description="test", callback=broken
            )
        },
    )
    interaction = SimpleNamespace(
        guild_id=123,
        response=SimpleNamespace(send_message=AsyncMock(), is_done=Mock(return_value=acknowledged)),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    command = tree.get_command("test", guild=discord.Object(id=123))
    asyncio.run(command.callback(interaction))
    assert "Feature command failed" in caplog.text
    sender = interaction.followup.send if acknowledged else interaction.response.send_message
    assert sender.call_args.kwargs["ephemeral"]


@pytest.mark.parametrize("permitted", [False, True])
def test_callback_checks_are_preserved_in_real_command_dispatch(tree, permitted):
    seen = AsyncMock()

    @app_commands.check(lambda interaction: interaction.user.id == 456)
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.guild_only()
    @app_commands.allowed_contexts(guilds=True, dms=False, private_channels=False)
    @app_commands.allowed_installs(guilds=True, users=False)
    async def restricted(interaction):
        await seen(interaction)

    load_fake(
        tree,
        {
            "feature": lambda registry: registry.register(
                name="restricted", description="test", callback=restricted
            )
        },
    )
    command = tree.get_command("restricted", guild=discord.Object(id=123))
    assert len(command.checks) == 2
    assert command.default_permissions.manage_guild
    assert command.guild_only
    assert not command.allowed_contexts.dm_channel
    assert not command.allowed_installs.user
    interaction = SimpleNamespace(
        guild_id=123,
        user=SimpleNamespace(id=456 if permitted else 999),
        permissions=discord.Permissions(manage_guild=permitted),
    )
    if permitted:
        asyncio.run(command._invoke_with_namespace(interaction, SimpleNamespace()))
        seen.assert_awaited_once_with(interaction)
    else:
        with pytest.raises(app_commands.CheckFailure):
            asyncio.run(command._invoke_with_namespace(interaction, SimpleNamespace()))
        seen.assert_not_awaited()


@pytest.mark.parametrize("module", sorted(BLOCKED_FEATURE_IMPORTS | {"requests"}))
@pytest.mark.parametrize("syntax", ["import {module}", "from {module} import thing"])
def test_ast_blocks_import_categories(module, syntax):
    with pytest.raises(PolicyViolation, match="feature.py:1"):
        validate_feature_source("feature.py", syntax.format(module=module))


@pytest.mark.parametrize("name", sorted(BLOCKED_FEATURE_NAMES))
def test_ast_blocks_calls_and_name_references(name):
    for source in [f"{name}('x')", f"reference = {name}"]:
        with pytest.raises(PolicyViolation, match="feature.py:1"):
            validate_feature_source("feature.py", source)


@pytest.mark.parametrize("attribute", sorted(BLOCKED_FEATURE_ATTRIBUTES))
@pytest.mark.parametrize("source", ["value.{attribute}", "value.{attribute}('argument')"])
def test_ast_blocks_host_access_attributes(attribute, source):
    with pytest.raises(PolicyViolation, match="feature.py:2: forbidden attribute " + attribute):
        validate_feature_source("feature.py", "# feature\n" + source.format(attribute=attribute))


@pytest.mark.parametrize("attribute", sorted(BLOCKED_FEATURE_ATTRIBUTES))
@pytest.mark.parametrize("alias", ["", " as harmless"])
def test_ast_blocks_asyncio_host_access_from_imports(attribute, alias):
    with pytest.raises(PolicyViolation, match="feature.py:1: forbidden import"):
        validate_feature_source("feature.py", f"from asyncio import {attribute}{alias}")


@pytest.mark.parametrize(
    "source",
    [
        "asyncio.create_subprocess_shell('command')",
        "asyncio.create_subprocess_exec('command')",
        "asyncio.subprocess.Process(transport, protocol, loop)",
        "loop.run_in_executor(None, callback)",
        "loop.subprocess_shell(protocol, 'command')",
        "loop.subprocess_exec(protocol, 'command')",
        "discord.utils.os.remove('file')",
        "x.sys",
        "from asyncio.subprocess import create_subprocess_shell as launch",
    ],
)
def test_ast_rejects_reported_host_access_chains(source):
    with pytest.raises(PolicyViolation):
        validate_feature_source("feature.py", source)


def test_ast_still_allows_safe_asyncio_and_discord_attributes():
    validate_feature_source(
        "feature.py",
        (
            "import asyncio\nimport discord\nfrom asyncio import sleep\n"
            "async def ping(interaction):\n"
            "    await asyncio.sleep(0)\n"
            "    await interaction.response.send_message(discord.Embed(title='pong').title)\n"
        ),
    )


@pytest.mark.parametrize(
    "source",
    [
        "x = interaction.__class__",
        "x = interaction.__dict__",
        "from . import other",
        "from nezubot_discrod import prompts",
        "import nezubot_discrod.prompts",
        "from asyncio import subprocess",
        "import json as open",
        "from json import loads as eval",
        "def open(): pass",
        "x = value.open",
    ],
)
def test_ast_blocks_attributes_bindings_and_relative_imports(source):
    with pytest.raises(PolicyViolation):
        validate_feature_source("feature.py", source)


def test_ast_allows_documented_imports_and_setup():
    source = "\n".join(f"import {name}" for name in sorted(ALLOWED_FEATURE_IMPORTS))
    source += (
        '\nasync def ping(interaction):\n    await interaction.response.send_message("pong")\n'
    )
    source += (
        'def setup(registry):\n    registry.register(name="ping", description="p", callback=ping)'
    )
    validate_feature_source("nezubot_discrod/features/ping.py", source)


def test_ast_syntax_error_reports_line():
    with pytest.raises(PolicyViolation, match="feature.py:2"):
        validate_feature_source("feature.py", "# comment\ndef")


def test_workspace_validates_feature_files_without_executing_them(tmp_path):
    directory = tmp_path / "nezubot_discrod" / "features"
    directory.mkdir(parents=True)
    feature = directory / "example.py"
    feature.write_text("raise RuntimeError('never execute on host')")
    path = "nezubot_discrod/features/example.py"
    validate_workspace(tmp_path, "", path + "\0")
    feature.write_text("import os\n")
    with pytest.raises(PolicyViolation, match=path + ":1"):
        validate_workspace(tmp_path, "", path + "\0")
    raw = ":100644 100644 " + "a" * 40 + " " + "0" * 40 + " M\0" + path + "\0"
    with pytest.raises(PolicyViolation):
        validate_workspace(tmp_path, raw, "")


def test_feature_files_editable_but_loader_and_guards_protected():
    for path in [
        "nezubot_discrod/features/ping.py",
        "tests/features/test_ping.py",
        "tests/test_feature_ping.py",
    ]:
        assert not protected_path(path)
    for path in [
        "nezubot_discrod/features/__init__.py",
        "nezubot_discrod/feature_loader.py",
        "tests/test_feature_registry_security.py",
    ]:
        assert protected_path(path)
