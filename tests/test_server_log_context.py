import asyncio
import json
import re
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_assistant import client_for, message, tool
from test_server_log import dense_store, record
from test_server_wiki import summarize

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.config import SandboxSettings
from nezubot_discrod.member_wiki import MemberWiki
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import _event_candidates, pre_retrieve, resolve_people
from nezubot_discrod.server_log_tools import LogResult, LogTools
from nezubot_discrod.server_wiki import ServerWiki


@pytest.fixture
def archive(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    for index in range(48):
        value = record(1, channel=str(10 + index % 3), text=f"Rust history {index} " + "x" * index)
        value["id"] = str(index + 1)
        value["timestamp"] = f"{2024 + index // 16}-{index % 12 + 1:02d}-01T00:00:00Z"
        rows.append(value)
    rows.append(record(59, channel="99", text="hidden secret"))
    store.upsert_many(rows)
    return store


def test_collection_diversity_paging_and_scoped_counts(archive):
    first = archive.collect(channel_ids=[10, 11, 12], author="20", limit=8)
    assert first["message_count"] == 48 and first["channel_count"] == 3
    assert len({hit.channel_id for hit in first["hits"]}) == 3
    assert {hit.date[:4] for hit in first["hits"]} == {"2024", "2025", "2026"}
    second = archive.collect(channel_ids=[10, 11, 12], author="20", limit=8,
                             offset=first["next_offset"])
    assert not {hit.id for hit in first["hits"]} & {hit.id for hit in second["hits"]}
    topic = archive.collect(channel_ids=[10, 11, 12], query="Rust")
    assert topic["message_count"] == 48
    assert all(hit.channel_id != "99" for hit in topic["hits"])
    archive.mark_deleted(first["hits"][0].id)
    after = archive.collect(channel_ids=[10, 11, 12], author="20")
    assert after["message_count"] == 47
    assert first["hits"][0].id not in {hit.id for hit in after["hits"]}
    assert archive.collect(channel_ids=[], author="20")["message_count"] == 0
    assert archive.collect(channel_ids=[10], author="nobody")["hits"] == []


@pytest.mark.parametrize("alias", ["me", "myself", "自分", "おれ", "俺", "私", "僕"])
def test_author_only_tools_and_caller_alias(archive, alias):
    log = LogTools(archive, [10, 11, 12], caller_id=20)
    recent = log.call("search_server_log", {"author": alias})
    assert len(recent.sources) == 10
    overview = log.call("collect_server_log", {"author": alias})
    assert json.loads(overview.splitlines()[0])["message_count"] == 48
    assert len(overview.sources) == 24
    assert "hidden secret" not in overview
    with pytest.raises(ValueError, match="Caller identity"):
        LogTools(archive, [10]).call("collect_server_log", {"author": alias})


def test_collection_paging_never_advertises_an_invalid_offset(archive):
    rows = []
    for index in range(1608):
        value = record(index + 5000, channel=str(index % 201 + 1000))
        value["timestamp"] = f"{2010 + index // 201}-01-01T00:00:00Z"
        rows.append(value)
    archive.upsert_many(rows)
    page = archive.collect(channel_ids=range(1000, 1201), author="20", offset=1000, limit=8)
    assert len(page["hits"]) == 8
    assert page["next_offset"] is None
    assert page["channel_count"] == 201 and page["channels_truncated"]


@pytest.mark.parametrize("args", [{}, {"author": "20", "limit": 61},
                                 {"query": "Rust", "offset": -1},
                                 {"query": "Rust", "channel_ids": [99]}])
def test_collection_validation(archive, args):
    with pytest.raises(ValueError):
        LogTools(archive, [10]).call("collect_server_log", args)


def test_identity_ambiguity_and_tombstones(archive):
    other = record(58)
    other["author"]["id"] = "21"
    archive.upsert(other)
    # A third party asking about a shared name gets the ambiguity...
    assert resolve_people(archive, [10], "aliceについて", 999) == ["20", "21"]
    # ...but when the caller's own name is among the matches, the caller
    # means themselves and wins over the ambiguity.
    assert resolve_people(archive, [10], "aliceについて", 20) == ["20"]
    assert resolve_people(archive, [10], "<@21>について", 20) == ["21"]
    assert resolve_people(archive, [10], "おれの情報", 20) == ["20"]
    assert archive.collect(channel_ids=[10], author="alice")["ambiguous_author_ids"] == ["20", "21"]
    archive.mark_deleted("58")
    assert resolve_people(archive, [10], "<@21>について", 20) == []


def test_person_alias_prefilter_keeps_normalization_and_word_boundaries(archive, monkeypatch):
    with archive.connect() as db:
        db.executemany("INSERT INTO author_aliases VALUES (?, ?, ?, ?, ?)",
                       [("100", "10", "20", f"unused-{i}", f"unused-{i}") for i in range(600)])
    search = Mock(wraps=re.search)
    monkeypatch.setattr("nezubot_discrod.server_log_context.re.search", search)
    assert resolve_people(archive, [10], "ＡＬＩＣＥについて", 999) == ["20"]
    assert search.call_count == 1
    assert resolve_people(archive, [10], "maliceについて", 999) == []
    assert search.call_count == 2
    assert resolve_people(archive, [10], "No matching person", 999) == []
    assert search.call_count == 2


def test_resolve_people_reuses_alias_snapshot_until_a_new_alias_arrives(archive):
    assert resolve_people(archive, [10], "aliceについて", 999) == ["20"]
    cache = archive.__dict__["_alias_snapshots"]
    assert len(cache) == 1
    snapshot = next(iter(cache.values()))[1]
    # No intervening alias insert: the cached snapshot is reused as-is.
    assert resolve_people(archive, [10], "aliceについて", 999) == ["20"]
    assert next(iter(cache.values()))[1] is snapshot
    # A new author inserts new alias rows, moving the MAX(rowid)
    # watermark, so the next resolve reloads and sees the new person.
    archive.upsert(record(50, author="carol"))
    assert resolve_people(archive, [10], "carolについて", 999) == ["21"]
    assert next(iter(cache.values()))[1] is not snapshot


def test_resolve_people_mentions_keep_mention_order(archive):
    other = record(58)
    other["author"]["id"] = "21"
    archive.upsert(other)
    assert resolve_people(archive, [10], "<@21>と<@20>について", 999) == ["21", "20"]
    assert resolve_people(archive, [10], "<@20>と<@21>について", 999) == ["20", "21"]


def test_event_candidates_extract_named_events_only():
    assert _event_candidates("「技術者倫理」の議論ってどうなった？") == ["技術者倫理"]
    assert _event_candidates("Paper Challenge 2って何だっけ？") == ["Paper Challenge 2"]
    assert _event_candidates("CTFの募集まだやってる？") == ["CTF"]
    assert _event_candidates("vexってどう思う？") == []
    assert _event_candidates("サーバーオーナーおしえて") == []
    assert _event_candidates("そのbot直った？") == []


def test_pre_retrieval_includes_named_event_server_wiki(archive):
    ServerWiki(archive).build(summarize, model="test", batch_size=16)
    log = LogTools(archive, [10, 11, 12], caller_id=999)
    result = asyncio.run(pre_retrieve(log, "Tournamentって何？", timeout=5))
    assert "Server Wiki event Tournament" in result
    assert result.diagnostics["chunks"].get("server_wiki", 0) >= 1
    # A resolved-person question does not run the event stage.
    person = asyncio.run(pre_retrieve(log, "aliceについて", timeout=5))
    assert "Server Wiki event" not in person


def test_automatic_context_wikis_are_id_keyed_scoped_and_deduplicated(archive, tmp_path):
    archive = dense_store(archive)
    archive.guild_id = "100"
    archive.embed_pending()
    wiki = ServerWiki(archive)
    wiki.build(summarize, model="test", batch_size=16)
    memories = MemberWiki(tmp_path / "memory" / "wiki.db")
    memories.for_member(20, 100).write("profile", "member preference")
    memories.for_member(21, 100).write("profile", "wrong person")
    memories.for_member(20, 999).write("profile", "wrong guild")
    log = LogTools(archive, [10, 11, 12], caller_id=20)
    result = asyncio.run(pre_retrieve(log, "おれの情報", member_wiki=memories,
                                     guild_id=100, timeout=5))
    assert "Speaker overview" in result and "Server Wiki person 20" in result
    assert "member preference" in result
    assert all(value not in result for value in ("hidden secret", "wrong person", "wrong guild"))
    assert len(result) <= 6000 and len(result.sources) <= 17
    server_wiki_lines = [
        line for line in result.splitlines() if line.startswith("Server Wiki person")
    ]
    assert len(server_wiki_lines) == len(set(server_wiki_lines))
    with wiki.writer() as db:
        db.execute("UPDATE server_wiki_entry SET summary = '' WHERE kind = 'person'")
        db.commit()
    empty = asyncio.run(pre_retrieve(log, "おれの情報", timeout=5))
    assert "Server Wiki person" not in empty


@pytest.mark.parametrize("error", [RuntimeError, KeyError])
def test_person_overview_survives_search_failure_and_budget_is_hard(archive, monkeypatch, error):
    archive._dense = object()
    log = LogTools(archive, [10, 11, 12], caller_id=20)

    def unavailable(*args, **kwargs):
        raise error("backend failed")

    monkeypatch.setattr(archive, "search", unavailable)
    result = asyncio.run(pre_retrieve(log, "my interests", timeout=5))
    assert "Speaker overview" in result and result.sources
    tiny = asyncio.run(pre_retrieve(log, "my interests", timeout=5, budget=20))
    assert len(tiny) <= 20 and not tiny.sources


def test_other_person_never_reads_private_member_wiki(archive, monkeypatch):
    archive.upsert(record(58, author="bob", text="Public Bob history"))
    archive._dense = object()
    monkeypatch.setattr(archive, "search", lambda *a, **kw: [])
    memory = Mock()
    result = asyncio.run(pre_retrieve(LogTools(archive, [10], caller_id=20),
                                     "<@21>について", member_wiki=memory, guild_id=100))
    assert "Public Bob history" in result
    memory.for_member.assert_not_called()


@pytest.mark.parametrize("mention", ["<@999999>", "<@!999999>"])
def test_unresolved_mentions_never_fall_back_to_caller_or_alias(archive, monkeypatch, mention):
    question = f"{mention}について私に教えて alice"
    assert resolve_people(archive, [10], question, 20) == []
    archive._dense = object()
    search = Mock(return_value=[])
    monkeypatch.setattr(archive, "search", search)
    memory = Mock()
    result = asyncio.run(pre_retrieve(LogTools(archive, [10], caller_id=20), question,
                                     member_wiki=memory, guild_id=100))
    assert not result
    assert search.call_count == 2
    assert all(call.kwargs["author"] is None and call.kwargs["limit"] == 8
               for call in search.call_args_list)
    memory.for_member.assert_not_called()


@pytest.mark.parametrize("question", ["私はPythonについて知りたい", "my code uses Python"])
def test_first_person_topic_keeps_other_authors_evidence(archive, monkeypatch, question):
    archive.upsert(record(58, author="bob", text="Python advice from another person"))
    other = archive.read("58", channel_ids=[10], limit=1)[0]
    archive._dense = object()
    calls = []
    original_collect = archive.collect

    def collect_overview(**kwargs):
        assert calls and calls[0]["author"] is None
        return original_collect(**kwargs)

    collect = Mock(side_effect=collect_overview)
    monkeypatch.setattr(archive, "collect", collect)

    def search(query, **kwargs):
        calls.append(kwargs)
        return [other] if kwargs["author"] is None else []

    monkeypatch.setattr(archive, "search", search)
    log = LogTools(archive, [10, 11, 12], caller_id=20)
    result = asyncio.run(pre_retrieve(log, question, timeout=5))
    assert "Python advice from another person" in result
    assert "Speaker overview" in result and len(result) <= 6000
    dense_calls = [call for call in calls if not call["lexical_only"]]
    assert [(call["author"], call["limit"]) for call in dense_calls] == [(None, 3), ("20", 5)]
    assert all(call["channel_ids"] == log.channel_ids for call in calls)
    assert all(call["require_dense"] for call in dense_calls)
    assert [call["infer_author"] for call in dense_calls] == [False, True]
    assert any(call["lexical_only"] and not call["require_dense"] for call in calls)
    collect.assert_called_once_with(channel_ids=log.channel_ids, author="20", limit=8,
                                    exclude_message_id=None)
    assert len(result.sources) == 9
    topic_only = asyncio.run(pre_retrieve(log, question, timeout=5, budget=200))
    assert "Python advice from another person" in topic_only
    assert len(topic_only) <= 200


def test_pre_retrieval_deduplicates_hits_across_chunks(archive, monkeypatch):
    archive._dense = object()
    shared = archive.read("1", channel_ids=[10], limit=1)[0]
    own = archive.read("2", channel_ids=[11], limit=1)[0]

    def search(query, **kwargs):
        if kwargs["lexical_only"]:
            return []
        return [shared] if kwargs["author"] is None else [shared, own]

    monkeypatch.setattr(archive, "search", search)
    log = LogTools(archive, [10, 11, 12], caller_id=20)
    result = asyncio.run(pre_retrieve(log, "私はRustについて知りたい", timeout=5))
    # The shared hit leads both the dense chunk and the author chunk but
    # is delivered (and sourced) only once; the author chunk keeps its
    # unique hit. (Line-final URLs: message 16's URL has message 1's as
    # a prefix, so a raw substring count would over-count.)
    lines = str(result).splitlines()
    assert sum(line.endswith(shared.url) for line in lines) == 1
    assert any(line.endswith(own.url) for line in lines)
    urls = [str(source) for source in result.sources]
    assert len(urls) == len(set(urls))


def test_unfiltered_search_does_not_infer_author_from_person_mentions(archive):
    archive.upsert(record(58, author="bob", text="Rust advice from another person"))
    inferred = archive.search("<@20> Rust", channel_ids=[10, 11, 12], limit=60)
    unfiltered = archive.search("<@20> Rust", channel_ids=[10, 11, 12], limit=60,
                                infer_author=False)
    assert all(hit.author_id == "20" for hit in inferred)
    assert any(hit.author_id == "21" for hit in unfiltered)


def test_lexical_fallback_never_enters_dense_backend_and_keeps_scope(archive):
    archive._dense = Mock()
    archive.upsert(record(58, channel="99", text="Rust secret"))
    hits = archive.search("Rust", channel_ids=[10], lexical_only=True)
    assert hits and all(hit.channel_id == "10" for hit in hits)
    archive._dense.search.assert_not_called()
    with pytest.raises(ValueError, match="mutually exclusive"):
        archive.search("Rust", channel_ids=[10], lexical_only=True, require_dense=True)


def test_background_writer_does_not_block_read_snapshots(archive):
    with archive.connect() as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        reader.execute("BEGIN")
        before = reader.execute("SELECT count(*) FROM messages").fetchone()[0]
        archive.upsert(record(57, text="A new passage"))
        assert reader.execute("SELECT count(*) FROM messages").fetchone()[0] == before
        with archive.connect() as latest:
            assert latest.execute("SELECT count(*) FROM messages").fetchone()[0] == before + 1


def test_person_context_includes_four_distinct_nonempty_server_wiki_summaries(archive, monkeypatch):
    archive._dense = object()
    monkeypatch.setattr(archive, "search", lambda *a, **kw: [])
    entries = [{"summary": text, "sources": [{"url": "https://discord.com/channels/100/10/1"}]}
               for text in ["", "fact 0", " fact 0 ", "fact 1", "fact 2", "fact 3", "fact 4"]]
    monkeypatch.setattr(ServerWiki, "lookup", lambda *a: [{"id": "20", "entries": entries}])
    result = asyncio.run(pre_retrieve(LogTools(archive, [10, 11, 12], caller_id=20),
                                     "my interests", timeout=5))
    lines = [line for line in result.splitlines() if line.startswith("Server Wiki person")]
    assert len(lines) == 4 and len(result) <= 6000
    assert all(f"fact {index}" in result for index in range(4))
    assert "fact 4" not in result


def test_ambiguous_names_never_inject_another_persons_wiki(archive, monkeypatch):
    value = record(58)
    value["author"]["id"] = "21"
    archive.upsert(value)
    archive._dense = object()
    monkeypatch.setattr(archive, "search", lambda *a, **kw: [])
    memory = Mock()
    # The caller is not one of the alices: the shared name stays
    # ambiguous, no person is resolved, and no member wiki is touched.
    # (A caller whose own name is among the matches resolves to
    # themselves instead; see test_identity_ambiguity_and_tombstones.)
    result = asyncio.run(pre_retrieve(LogTools(archive, [10], caller_id=999),
                                     "aliceについて", member_wiki=memory, guild_id=100))
    assert "Ambiguous people" in result and "Speaker overview" not in result
    memory.for_member.assert_not_called()


def test_pre_retrieval_delivers_sql_evidence_without_dense(archive):
    log = LogTools(archive, [10], caller_id=20)
    assert "Rust history" in asyncio.run(pre_retrieve(log, "Rust"))
    assert "Speaker overview" in asyncio.run(pre_retrieve(log, "my history"))


def test_pre_retrieval_timeout_keeps_partial_results_without_global_blocking(archive, monkeypatch):
    log = LogTools(archive, [10], caller_id=20)
    archive._dense = object()
    original = archive.search
    release = Event()
    calls = []

    def slow(*args, **kwargs):
        if not kwargs["lexical_only"]:
            calls.append(1)
            release.wait(3)
            return []
        return original(*args, **kwargs)

    monkeypatch.setattr(archive, "search", slow)

    async def run():
        start = time.monotonic()
        first = await pre_retrieve(log, "my Rust history", timeout=0.15)
        second = await pre_retrieve(log, "Rust", timeout=0.15)
        third = await pre_retrieve(log, "Rust", timeout=0.15)
        assert "Speaker overview" in first
        assert first.diagnostics["status"] == "deadline"
        assert "Rust history" in second and "Rust history" in third
        assert "dense" in third.diagnostics["skipped"]
        assert time.monotonic() - start < 0.8

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert calls == [1, 1]


def test_slow_server_wiki_does_not_block_overview_or_member_wiki(archive, monkeypatch, caplog):
    release = Event()
    def slow(*args):
        release.wait(3)
        return []
    monkeypatch.setattr(ServerWiki, "lookup", slow)
    memory = Mock()
    memory.for_member.return_value.context.return_value = [{"page": "bio", "excerpt": "A fact"}]
    try:
        with caplog.at_level("INFO", logger="nezubot_discrod.server_log_context"):
            result = asyncio.run(pre_retrieve(LogTools(archive, [10], caller_id=20),
                                              "my Rust", member_wiki=memory,
                                              guild_id=100, timeout=0.15))
        assert "Speaker overview" in result and "A fact" in result
        assert "server_wiki" in result.diagnostics["active"]
        assert len([record for record in caplog.records
                    if record.message.startswith("Automatic retrieval {")]) == 1
        async def repeated():
            log = LogTools(archive, [10], caller_id=20)
            await pre_retrieve(log, "my Rust", member_wiki=memory, guild_id=100, timeout=0.15)
            return await pre_retrieve(log, "my Rust", member_wiki=memory, guild_id=100,
                                      timeout=0.15)
        saturated = asyncio.run(repeated())
        assert "Speaker overview" in saturated and "Rust history" in saturated
        assert "A fact" in saturated and "member_wiki" not in saturated.diagnostics["skipped"]
        assert "server_wiki" in saturated.diagnostics["skipped"]
    finally:
        release.set()


@pytest.mark.parametrize("provider", ["openai", "codex"])
def test_automatic_context_reaches_model_without_tool_call_and_taints_web(archive, provider):
    archive = dense_store(archive)
    archive.embed_pending()
    web = Mock()
    web.call.return_value = '{"text":"untrusted web content"}'
    client, sent = client_for(message(calls=[tool('{"query":"a"}', "web_search")]),
                              message(calls=[tool('{"query":"b"}', "web_search")]),
                              message("done"))
    completion = AsyncMock(return_value=type("Response", (), {
        "choices": [type("Choice", (), {"message": message("done")})()],
    })()) if provider == "codex" else None
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive, web=web,
                   wiki_completion=completion)
    asyncio.run(ai.reply(10, "おれの情報", member_id=20, guild_id=100, channel_ids=[10, 11, 12]))
    payload = json.loads((completion.call_args.kwargs["messages"] if completion else
                          sent[0]["messages"])[1]["content"])
    assert "Speaker overview" in payload["retrieved_server_context"]["excerpts"]
    if completion:
        assert completion.call_args.kwargs["private_read"] is True
        assert completion.call_args.kwargs["server_log"].caller_id == "20"
    else:
        assert web.call.call_count == 1
        assert "web access disabled" in sent[-1]["messages"][-1]["content"]


@pytest.mark.parametrize(("guild_id", "channel", "expected"), [
    (100, 10, ("10",)), (100, 404, ()), (None, 10, ("10", "99")),
])
def test_automatic_scope_is_safe_for_reply_audience(
    archive, monkeypatch, guild_id, channel, expected,
):
    retrieve = AsyncMock(return_value="")
    monkeypatch.setattr("nezubot_discrod.assistant.pre_retrieve", retrieve)
    client, _ = client_for(message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive)
    asyncio.run(ai.reply(channel, "Rust", member_id=20, guild_id=guild_id, channel_ids=[10, 99]))
    assert retrieve.call_args.args[0].channel_ids == expected


def test_current_question_is_excluded_from_stats_search_and_wiki_sources(archive, monkeypatch):
    question = record(999, text="Rust question that is not historical evidence")
    question["timestamp"] = "2027-01-01T00:00:00Z"
    archive.upsert(question)
    archive = dense_store(archive)
    archive.embed_pending()
    monkeypatch.setattr(ServerWiki, "lookup", lambda *a: [{"id": "20", "entries": [{
        "summary": "Current-question wiki fact", "sources": [{
            "url": "https://discord.com/channels/100/10/999",
        }],
    }]}])
    log = LogTools(archive, [10, 11, 12], caller_id=20)
    result = asyncio.run(pre_retrieve(log, "my Rust interests", message_id=999, timeout=5))
    assert '"message_count": 48' in result and "2027" not in result
    assert "not historical evidence" not in result and "Current-question wiki fact" not in result
    assert all(not source.endswith("/999") for source in result.sources)
    stats = archive.collect(channel_ids=log.channel_ids, author="20", exclude_message_id=999)
    assert stats["message_count"] == 48 and not stats["last_seen"].startswith("2027")
    assert archive.search("Rust", channel_ids=log.channel_ids, exclude_message_id=999)
    assert archive.search("Rust", channel_ids=log.channel_ids, exclude_message_id=999,
                          before="2024-01-01") == []


def test_public_reply_never_automatically_includes_private_channel_evidence(archive):
    archive = dense_store(archive)
    archive.embed_pending()
    client, sent = client_for(message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive)
    answer = asyncio.run(ai.reply(10, "おれの情報", member_id=20, guild_id=100,
                                  channel_ids=[10, 99]))
    payload = json.loads(sent[0]["messages"][1]["content"])
    assert "Speaker overview" in payload["retrieved_server_context"]["excerpts"]
    assert "hidden secret" not in payload["retrieved_server_context"]["excerpts"]
    assert all("/99/" not in source for source in answer.sources)


@pytest.mark.parametrize("guild_id", [100, None])
def test_automatic_public_scope_stays_within_requester_permissions(archive, monkeypatch, guild_id):
    retrieve = AsyncMock(return_value="")
    monkeypatch.setattr("nezubot_discrod.assistant.pre_retrieve", retrieve)
    client, _ = client_for(message("done"))
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive)
    asyncio.run(ai.reply(10, "Rust", member_id=20, guild_id=guild_id, channel_ids=[10, 11, 99],
                         retrieval_channel_ids=[10, 11, 1000]))
    assert retrieve.call_args.args[0].channel_ids == ("10", "11")


@pytest.mark.parametrize("provider", ["openai", "codex"])
def test_explicit_tool_sources_precede_deduplicated_automatic_sources(
    archive, monkeypatch, provider,
):
    explicit = LogTools(archive, [10]).call("collect_server_log", {"author": "20", "limit": 3})
    fallback = [f"https://discord.com/channels/100/10/{value}" for value in (501, 502, 503)]
    automatic = LogResult("automatic evidence", sources=[*fallback, explicit.sources[0]])
    monkeypatch.setattr("nezubot_discrod.assistant.pre_retrieve", AsyncMock(return_value=automatic))
    client, _ = client_for(
        message(calls=[tool('{"author":"20","limit":3}', "collect_server_log")]),
        message("done"),
    )
    completion = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=message("done"))], sources=explicit.sources,
    )) if provider == "codex" else None
    ai = Assistant(client, "test", SandboxSettings(), server_log=archive,
                   wiki_completion=completion)
    answer = asyncio.run(ai.reply(10, "my interests", member_id=20, guild_id=100, channel_ids=[10]))
    assert list(answer.sources[:3]) == list(explicit.sources)
    assert list(answer.sources[3:]) == fallback
    assert len(answer.sources) == len(set(answer.sources))


def test_resolve_people_decorated_bot_name_by_short_token(archive):
    bot = record(7, text="event announcement")
    bot["id"] = "77"
    bot["author"] = {"id": "30", "username": "vexbot",
                     "global_name": "Vexちゃん【公式bot】", "bot": True}
    archive.upsert(bot)
    # Full alias never appears verbatim and the author is a bot, so the
    # primary path cannot resolve these; the token fallback must.
    assert resolve_people(archive, [10], "Vexってどう思う？", 999) == ["30"]
    assert resolve_people(archive, [10], "vexについて", 999) == ["30"]
    assert resolve_people(archive, [10], "vexbotについて", 999) == ["30"]
    # Unrelated tokens and partial-word traps still resolve nobody.
    assert resolve_people(archive, [10], "eventについて", 999) == []
    assert resolve_people(archive, [10], "maliceについて", 999) == []


def test_resolve_people_seed_grounds_a_demonstrative(archive):
    question = "この人物の過去のメッセージさがして"
    # The text alone names nobody: no mention, no self reference, no alias.
    assert resolve_people(archive, [10], question, 999) == []
    # The conversation layer's grounded person is trusted as the referent.
    assert resolve_people(archive, [10], question, 999, seed="20") == ["20"]


def test_resolve_people_seed_falls_through_when_unusable(archive):
    # Author 21 has no in-scope messages: fall through to text resolution.
    assert resolve_people(archive, [10], "この人物について", 999, seed="21") == []
    # A seed equal to the caller is not a referent.
    assert resolve_people(archive, [10], "この人物について", 20, seed="20") == []


def test_pre_retrieve_seed_delivers_that_persons_evidence(archive):
    log = LogTools(archive, [10, 11, 12], caller_id=999)
    question = "この人物の過去のメッセージさがして"
    unseeded = asyncio.run(pre_retrieve(log, question, timeout=5))
    assert "Speaker overview" not in unseeded
    seeded = asyncio.run(pre_retrieve(log, question, seed_person="20", timeout=5))
    assert "Speaker overview" in seeded
    assert seeded.sources


def test_self_reference_dialect_and_particles(archive):
    # うち is a first person in dialect; と/とか/って after a self noun
    # still refer to the speaker (俺とかはどう was a production miss).
    assert resolve_people(archive, [10], "うちについておしえて", 20) == ["20"]
    assert resolve_people(archive, [10], "俺とかはどう", 20) == ["20"]
    assert resolve_people(archive, [10], "俺ってどう思う", 20) == ["20"]
    assert resolve_people(archive, [10], "うちについておしえて", 999) == ["999"]


def test_decorated_display_name_core_resolves(archive):
    value = record(50, author="bob", text="Fluxer の話")
    value["author"]["global_name"] = "さくら#abcd"
    archive.upsert(value)
    # The stored alias is the decorated name; the question uses the core.
    assert resolve_people(archive, [10], "さくらについておしえて", 999) == ["21"]


def test_carryover_fills_signal_less_followup(archive):
    other = record(51, author="bob", text="bob の発言")
    archive.upsert(other)
    # No person signal at all: the previous turn's subject carries over.
    assert resolve_people(archive, [10], "どこ住みかわかる？", 999) == []
    assert resolve_people(archive, [10], "どこ住みかわかる？", 999,
                          carryover="21") == ["21"]
    # A name in the text always beats the carryover.
    assert resolve_people(archive, [10], "aliceについて", 999,
                          carryover="21") == ["20"]
    # A carryover with no in-scope messages is ignored.
    assert resolve_people(archive, [10], "どこ住みかわかる？", 999,
                          carryover="424242") == []


def test_decorated_bot_accounts_resolve_to_dominant_author(archive):
    # Bridge/bot accounts sharing one decorated name: the primary path
    # requires human authors, so these resolve through the bot-inclusive
    # retry, and the account with the overwhelming volume wins.
    rows = []
    for index in range(55):
        value = record(1, author="sakura-main", text=f"さくらの発言 {index}", bot=True)
        value["id"] = f"9{index:03d}"
        value["author"]["id"] = "31"
        value["author"]["global_name"] = "さくら#abcd"
        value["timestamp"] = f"2025-01-{(index % 28) + 1:02d}T00:00:00Z"
        rows.append(value)
    for index in range(4):
        value = record(1, author="sakura-alt", text=f"別さくらの発言 {index}", bot=True)
        value["id"] = f"8{index:03d}"
        value["author"]["id"] = "32"
        value["author"]["global_name"] = "さくら#abcd"
        value["timestamp"] = f"2025-02-{(index % 28) + 1:02d}T00:00:00Z"
        rows.append(value)
    archive.upsert_many(rows)
    assert resolve_people(archive, [10], "さくらについておしえて", 999) == ["31"]
