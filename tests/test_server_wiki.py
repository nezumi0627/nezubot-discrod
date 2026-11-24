import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import nezubot_discrod.server_wiki as server_wiki_module
from nezubot_discrod.bot import search_server_wiki
from nezubot_discrod.messages import tool_sources
from nezubot_discrod.prompts import SERVER_WIKI_BUILDER_SYSTEM
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_tools import LogTools
from nezubot_discrod.server_wiki import ServerWiki, main


def record(
    number, *, channel="10", name="Alice", text="I prefer Rust", guild="123", bot=False, author=None
):
    return {
        "id": str(number), "guild_id": guild, "channel_id": channel,
        "author": {"id": author or ("99" if bot else "42"), "username": name, "bot": bot},
        "timestamp": f"2026-01-01T00:00:{number:02}+00:00", "content": text,
    }


@pytest.fixture
def wiki(tmp_path):
    return ServerWiki(ServerLog(tmp_path / "private" / "server_log.db", guild_id="123"))


def request_rows(batch):
    payload = json.loads(batch)
    return [
        {"id": row[0], "author_id": payload["authors"][row[1]][0], "text": row[-1]}
        for row in payload["messages"]
    ]


def summarize(prompt, batch):
    assert "untrusted records" in prompt
    rows = request_rows(batch)
    authors = {row["author_id"]: row for row in reversed(rows)}
    people = [
        {"id": row["author_id"], "summary": row["text"], "sources": [row["id"]]}
        for row in authors.values()
    ]
    events = [
        {"title": "Tournament", "summary": rows[-1]["text"], "sources": [rows[-1]["id"]]}
    ]
    return json.dumps({"people": people, "events": events}), (10, 4)


def test_full_archive_checkpoint_and_channel_scope(wiki, monkeypatch):
    wiki.archive.upsert_many([
        record(1), record(2, channel="11", text="Private plan"),
        record(3, channel="12", guild="999", text="Another guild"),
        record(4, name="Alice2"), record(5, bot=True, text="Round started"),
    ])
    assert wiki.build(summarize, model="test") == (2, 0, 20, 8, 0, 0, 1)
    assert wiki.build(summarize, model="test") == (0, 2, 0, 0, 0, 0, 1)
    assert wiki.path.stat().st_mode & 0o777 == 0o600
    person, = wiki.lookup("person", "<@42>", [10])
    assert {item["title"] for item in person["aliases"]} == {"Alice", "Alice2"}
    assert person["entries"][0]["summary"] == "I prefer Rust"
    event, = wiki.lookup("event", "tournament", [10])
    assert event["id"] == "4"
    assert event["entries"][0]["sources"] == [{
        "date": "2026-01-01T00:00:04+00:00", "url": "https://discord.com/channels/123/10/4",
    }]
    assert wiki.lookup("person", "42", [12]) == []
    assert wiki.lookup("person", "99", [10]) == []

    def resolve(db, value, channels):
        assert value == "shared name"
        assert channels == ["10"]
        return ("42",)

    monkeypatch.setattr(wiki.archive, "_author_ids", resolve, raising=False)
    assert wiki.lookup("person", "shared name", [10]) == [person]


def test_edit_deletion_and_resume_invalidate_batches(wiki):
    wiki.archive.upsert_many([record(1), record(2, text="Tournament won"), record(3)])
    assert wiki.build(summarize, model="test", batch_size=2)[:2] == (2, 0)
    wiki.archive.mark_deleted("1")
    wiki.archive.upsert({**record(2, text="Score corrected"), "edited_timestamp": "2026-02-01"})
    assert wiki.lookup("event", "2", [10]) == []
    assert wiki.lookup("event", "3", [10])
    assert wiki.build(summarize, model="test", batch_size=2)[:2] == (1, 1)
    event, = wiki.lookup("event", "2", [10])
    assert event["entries"][0]["summary"] == "Score corrected"
    wiki.archive.upsert(record(4))
    assert wiki.build(summarize, model="test", batch_size=2)[:2] == (1, 2)


def test_lookup_caps_batch_validation_and_reads_current_wiki_and_archive(wiki, monkeypatch):
    wiki.archive.upsert_many([record(i, text=f"Fact {i}") for i in range(1, 41)])
    wiki.build(summarize, model="test", batch_size=1)
    messages = server_wiki_module._messages
    validations = []

    def counted(rows):
        validations.append(1)
        return messages(rows)

    monkeypatch.setattr(server_wiki_module, "_messages", counted)
    page, = wiki.lookup("person", "42", [10])
    assert len(page["entries"]) == 8 and len(validations) <= 9
    with wiki.writer() as db:
        db.execute(
            "UPDATE server_wiki_entry SET summary='Changed fact' "
            "WHERE kind='person' AND start=39"
        )
        db.commit()
    assert wiki.lookup("person", "42", [10])[0]["entries"][0]["summary"] == "Changed fact"
    wiki.archive.mark_deleted("40")
    assert all(entry["summary"] != "Changed fact"
               for entry in wiki.lookup("person", "42", [10])[0]["entries"])


def test_invalid_source_is_skipped_and_later_batches_still_build(wiki):
    wiki.archive.upsert_many([record(1), record(2), record(3)])

    def invalid_second(prompt, batch):
        if request_rows(batch)[0]["id"] != "2":
            return summarize(prompt, batch)
        return '{"people":[],"events":[{"title":"fake","summary":"fake","sources":["1"]}]}', None

    result = wiki.build(invalid_second, model="test", batch_size=1, workers=1)
    assert result == (3, 0, 20, 8, 0, 1, 0)
    assert wiki.lookup("event", "1", [10]) and wiki.lookup("event", "3", [10])
    assert not wiki.lookup("event", "2", [10])
    assert wiki.build(summarize, model="test", batch_size=1) == (0, 3, 0, 0, 0, 1, 0)


def test_garbage_splits_and_retry_only_resubmits_failed_messages(wiki):
    wiki.archive.upsert_many([record(1), record(2), record(3)])
    calls = []

    def sometimes_invalid(prompt, batch):
        ids = [row["id"] for row in request_rows(batch)]
        calls.append(ids)
        if "2" in ids:
            return "no JSON here {", (5, 2)
        return summarize(prompt, batch)

    assert wiki.build(sometimes_invalid, model="test") == (1, 0, 35, 14, 2, 1, 0)
    assert calls == [["1", "2", "3"], ["1"], ["2", "3"], ["2"], ["3"]]
    assert wiki.lookup("event", "1", [10]) and wiki.lookup("event", "3", [10])
    with wiki.writer() as db:
        skipped = db.execute("SELECT message_id, reason FROM server_wiki_skipped")
        assert [tuple(row) for row in skipped] == [("2", "invalid model output")]

    def retry(prompt, batch):
        assert [row["id"] for row in request_rows(batch)] == ["2"]
        return summarize(prompt, batch)

    assert wiki.build(retry, model="test", retry_skipped=True) == (1, 0, 10, 4, 0, 0, 0)
    assert all(wiki.lookup("event", key, [10]) for key in ["1", "2", "3"])
    assert wiki.build(retry, model="test", retry_skipped=True) == (0, 1, 0, 0, 0, 0, 0)


def test_fenced_and_partially_valid_output_with_no_split(wiki):
    wiki.archive.upsert_many([record(1), record(2)])

    def partial(prompt, batch):
        result, usage = summarize(prompt, batch)
        data = json.loads(result)
        data["people"].insert(0, {"id": "999", "summary": "wrong author", "sources": ["1"]})
        data["events"].insert(0, {"title": "fake", "summary": "bad source", "sources": ["999"]})
        data["events"].insert(0, {"title": "bad Unicode", "summary": "\ud800", "sources": ["1"]})
        data["events"].append("not an event")
        return f"Here you go:\n```json\n{json.dumps(data)}\n```\nFinished.", usage

    assert wiki.build(partial, model="test") == (1, 0, 10, 4, 0, 0, 0)
    person, = wiki.lookup("person", "42", [10])
    assert person["entries"][0]["summary"] == "I prefer Rust"
    assert not wiki.lookup("person", "999", [10])
    assert wiki.lookup("event", "2", [10])
    assert not wiki.lookup("event", "fake", [10])


def test_existing_checkpoints_are_not_regenerated_when_adding_skips_table(wiki):
    wiki.archive.upsert(record(1))
    wiki.build(summarize, model="test")
    with wiki.writer() as db:
        db.execute("DROP TABLE server_wiki_skipped")

    def unexpected(prompt, batch):
        pytest.fail("Existing checkpoint should not call the model")

    assert wiki.build(unexpected, model="test", retry_skipped=True) == (0, 1, 0, 0, 0, 0, 0)
    assert wiki.lookup("event", "1", [10])


def test_larger_batches_keep_old_checkpoints_and_filter_only_submitted_text(wiki):
    wiki.archive.upsert_many([record(1), record(2)])
    wiki.build(summarize, model="test", batch_size=1)
    wiki.archive.upsert_many([
        record(3, text=""), record(4, text="https://example.com/private"),
        record(5, text="🎉 <:emoji:123>"), record(6, text="!ping"),
        record(7, text="/rank"), record(8, text="Bot announcement", bot=True),
        record(9, text="Read https://example.com/sensitive " + "x" * 800 + " end"),
        record(10, text="I use Python"),
    ])

    def check(prompt, batch):
        rows = request_rows(batch)
        assert [row["id"] for row in rows] == ["9", "10"]
        assert len(rows[0]["text"]) <= 600 and "example.com" not in rows[0]["text"]
        assert rows[0]["text"].endswith("end")
        return '{"people":[],"events":[]}', (10, 4)

    result = wiki.build(check, model="test", batch_size=400)
    assert (result.done, result.unchanged, result.filtered) == (1, 2, 6)
    assert wiki.lookup("event", "1", [10]) and wiki.lookup("event", "2", [10])


def test_resume_keeps_later_checkpoints_when_earlier_batches_are_missing(wiki):
    wiki.archive.upsert_many([record(n) for n in range(1, 5)])
    wiki.build(summarize, model="test", batch_size=1)
    with wiki.writer() as db, db:
        db.execute("DELETE FROM server_wiki_batch WHERE start < 2")
        db.execute("DELETE FROM server_wiki_entry WHERE start < 2")

    def check(prompt, batch):
        assert [row["id"] for row in request_rows(batch)] == ["1", "2"]
        return summarize(prompt, batch)

    assert wiki.build(check, model="test")[:2] == (1, 2)
    assert wiki.lookup("event", "3", [10]) and wiki.lookup("event", "4", [10])


def test_people_pass_is_visible_before_events_and_scoped_to_sources(wiki):
    wiki.archive.upsert_many([
        record(1, text="Public Rust"), record(2, author="43", name="Bob", text="Public Python"),
        record(3, channel="11", author="44", name="Carol", text="Private plan"),
    ])
    calls = []

    def check(prompt, batch):
        profile = json.loads(batch)["task"].startswith("Build one")
        if not profile and False not in calls:
            assert not wiki.lookup("event", "tournament", [10])
            assert wiki.lookup("person", "43", [10])[0]["entries"]
            assert wiki.lookup("person", "44", [11])[0]["entries"]
        calls.append(profile)
        return summarize(prompt, batch)

    result = wiki.build(check, model="test", workers=1, people_only=True)
    assert result.done == 2 and calls == [True, True]
    assert not wiki.lookup("person", "44", [10])
    assert not wiki.lookup("event", "tournament", [10])
    wiki.build(check, model="test", workers=1, people_first=True)
    assert calls == [True, True, False, False]
    assert wiki.lookup("event", "tournament", [10])
    wiki.archive.mark_deleted("3")
    assert not wiki.lookup("person", "44", [11])


def test_lookup_tool_sources_do_not_include_other_channels(wiki):
    wiki.archive.upsert_many([record(1), record(2, channel="11", text="Private plan")])
    wiki.build(summarize, model="test")
    args = {"kind": "person", "query": "42"}
    result = LogTools(wiki.archive, [10, 11], server_wiki_channel=10).call("read_server_wiki", args)
    assert "Private plan" not in result
    assert tool_sources("read_server_wiki", result) == ["https://discord.com/channels/123/10/1"]
    denied = LogTools(wiki.archive, [10], server_wiki_channel=11).call("read_server_wiki", args)
    assert "unavailable" in denied


def test_slash_is_ephemeral_and_permission_filtered(wiki, monkeypatch):
    wiki.archive.upsert_many([record(1), record(2, channel="11", text="Private plan")])
    wiki.build(summarize, model="test")
    bot = SimpleNamespace(settings=SimpleNamespace(guild_id=123), server_log=wiki.archive)
    interaction = SimpleNamespace(
        guild_id=123, user=object(), response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    monkeypatch.setattr("nezubot_discrod.bot.allowed_channel_ids", AsyncMock(return_value=[10]))
    asyncio.run(search_server_wiki(interaction, bot, "person", "42"))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    args, kwargs = interaction.followup.send.await_args
    assert kwargs["ephemeral"] is True
    assert "I prefer Rust" in args[0] and "Private plan" not in args[0]


@pytest.mark.parametrize("override", [False, True])
def test_cli_runs_on_synthetic_db(wiki, monkeypatch, capsys, override):
    wiki.archive.upsert(record(1))
    monkeypatch.setenv("AI_PROVIDER", "codex" if override else "openai")
    monkeypatch.setenv("DISCORD_TOKEN", "")
    monkeypatch.setenv("AI_API_KEY", "test")
    monkeypatch.setenv("AI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AI_MODEL", "test")

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["base_url"] == "https://example.test/v1"
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, *, model, messages):
            assert model == "test"
            content, _ = summarize(*(message["content"] for message in messages))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage={"prompt_tokens": 10, "completion_tokens": 4},
            )

    monkeypatch.setattr("nezubot_discrod.server_wiki.OpenAI", Client)
    args = ["--db", str(wiki.archive.path), "--guild-id", "123", "--events-only"]
    if override:
        args += ["--provider", "openai"]
    main(args)
    output = capsys.readouterr().out
    assert "1 batches built, 0 unchanged; reported tokens: 10 in, 4 out" in output
    assert "0 splits, 0 skipped messages" in output
    assert wiki.lookup("person", "42", [10])


def test_cli_uses_configured_codex_login_without_keys(wiki, monkeypatch, capsys):
    wiki.archive.upsert_many([record(1), record(2)])
    monkeypatch.setenv("DISCORD_TOKEN", "")
    monkeypatch.setenv("DISCORD_OWNER_ID", "not needed for the builder")
    monkeypatch.setenv("DISCORD_GUILD_ID", "123")
    monkeypatch.setenv("AI_PROVIDER", "codex")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "gpt-6-luna")
    monkeypatch.setenv("CODEX_HOME", "/test/login")
    monkeypatch.setenv("CODEX_BINARY", "/test/bin/codex")
    loops = []

    class Client:
        def __init__(self, settings):
            assert settings.provider == "codex"
            assert settings.codex_home == "/test/login"
            assert settings.codex_binary == "/test/bin/codex"
            assert not settings.api_key

        async def create(self, *, model, messages, server_wiki):
            assert model == "gpt-6-luna" and server_wiki
            assert messages[0] == {"role": "system", "content": SERVER_WIKI_BUILDER_SYSTEM}
            loops.append(asyncio.get_running_loop())
            content, _ = summarize(*(message["content"] for message in messages))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage={"prompt_tokens": 10, "completion_tokens": 4},
            )

        async def close(self):
            loops.append(asyncio.get_running_loop())

    monkeypatch.setattr("nezubot_discrod.codex.CodexClient", Client)
    main(["--db", str(wiki.archive.path), "--batch-size", "1", "--workers", "1", "--events-only"])
    assert len(loops) == 3 and len(set(loops)) == 1 and loops[0].is_closed()
    assert "2 batches built, 0 unchanged; reported tokens: 20 in, 8 out" in capsys.readouterr().out
    assert wiki.lookup("person", "42", [10])


@pytest.mark.parametrize("failure", ["rate limit", "timeout"])
def test_codex_pool_runs_calls_concurrently_and_retries(wiki, monkeypatch, failure):
    from nezubot_discrod.codex import CodexRateLimited
    from nezubot_discrod.server_wiki import _build_codex

    wiki.archive.upsert_many([record(1), record(2)])
    active, maximum, attempts, closed = 0, 0, {}, []
    real_sleep = asyncio.sleep

    class Client:
        def __init__(self, settings):
            self.busy = False
            self.auth = None

        async def warm_server_wiki_auth(self):
            assert not attempts and not active
            await real_sleep(0)
            self.auth = object()
            warmed.append(self.auth)
            return self.auth

        def prepare_runtime(self, *, auth_file):
            assert not attempts and warmed == [auth_file]
            self.auth = auth_file

        async def create(self, *, model, messages, server_wiki):
            nonlocal active, maximum
            assert not self.busy and server_wiki and warmed == [self.auth]
            key = request_rows(messages[1]["content"])[0]["id"]
            attempts[key] = attempts.get(key, 0) + 1
            self.busy = True
            active += 1
            maximum = max(active, maximum)
            try:
                await real_sleep(0.01)
                if key == "1" and attempts[key] == 1:
                    if failure == "rate limit":
                        raise CodexRateLimited
                    await asyncio.Event().wait()
                content, _ = summarize(*(message["content"] for message in messages))
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None
                )
            finally:
                self.busy = False
                active -= 1

        async def close(self):
            closed.append(self)

    warmed = []

    async def without_wait(delay):
        minimum = 10 if failure == "rate limit" else 2
        assert minimum <= delay <= minimum + 2
        await real_sleep(0)

    monkeypatch.setattr("nezubot_discrod.codex.CodexClient", Client)
    monkeypatch.setattr("nezubot_discrod.server_wiki.asyncio.sleep", without_wait)
    result = asyncio.run(_build_codex(wiki, None, "test", workers=2, timeout=0.1, batch_size=1))
    assert maximum == 2 and attempts == {"1": 2, "2": 1}
    assert result.done == 2 and len(closed) == 2
