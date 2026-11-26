import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_server_log import record

from nezubot_discrod.appellations import mine_appellations
from nezubot_discrod.bot import show_me
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.understanding import (
    build_me_embed,
    collect_understanding,
    enqueue_stale_me_users,
    process_understanding_jobs,
    record_me_use,
)

GUILD_ID = 100


@pytest.fixture
def me_store(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    for index in range(6):
        value = record(1, channel="10", text=f"tanuki_devの発言 {index}")
        value["id"] = str(100 + index)
        value["author"] = {"id": "30", "username": "tanuki_dev", "bot": False}
        rows.append(value)
    for index in range(5):
        value = record(1, channel="10", text=f"タヌキさん、それはどう？ {index}")
        value["id"] = str(200 + index)
        value["author"] = {"id": "999", "username": "testuser", "bot": False}
        rows.append(value)
    # A second caller: an appellation counts as established for /me
    # only when more than one person uses it.
    for index in range(3):
        value = record(1, channel="10", text=f"タヌキ、それな {index}")
        value["id"] = str(300 + index)
        value["author"] = {"id": "888", "username": "someone", "bot": False}
        rows.append(value)
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO message_people VALUES (?, '30', 'reply', 1.0)",
            [(str(200 + index),) for index in range(5)]
            + [(str(300 + index),) for index in range(3)],
        )
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, display_name, nickname) "
            "VALUES ('30', '100', 'tanuki_dev', 'のの', '')"
        )
    mine_appellations(store)
    return store


def test_collect_understanding_scores_from_real_rows(me_store):
    report = collect_understanding(me_store, "30")
    assert 0 < report["total"] <= 100
    parts = report["parts"]
    # The appellation graph knows タヌキ (5 uses by one caller).
    assert parts["appellations"]["score"] > 0
    assert "タヌキ" in parts["appellations"]["detail"]
    # Identity is partially known (username + display name, no nickname).
    assert parts["identity"]["score"] > 0
    assert parts["person_graph"]["score"] > 0
    # Nobody has used /me yet.
    assert report["priority_tier"] is False


def test_collect_understanding_counts_only_established_appellations(me_store):
    # Pre-repair shape: one caller's repeated fragment plus a pile of
    # one-off strings. None of it may inflate the /me counts, and the
    # representative name stays the genuinely shared one.
    with me_store.connect() as db:
        db.execute(
            "INSERT INTO appellations VALUES ('30', '999', 'へんじ', 16, '')"
        )
        db.executemany(
            "INSERT INTO appellations VALUES ('30', ?, ?, 1, '')",
            [(str(500 + index), f"かけら{index}") for index in range(30)],
        )
    report = collect_understanding(me_store, "30")
    part = report["parts"]["appellations"]
    # Only タヌキ (8 uses, 2 callers) is established.
    assert "呼び名 1 種" in part["detail"]
    assert "証拠 8 件" in part["detail"]
    assert "呼び手 2 人" in part["detail"]
    assert "「タヌキ」" in part["detail"]
    assert part["score"] < 1.0


def test_collect_understanding_graph_counts_only_interpersonal_links(me_store):
    with me_store.connect() as db:
        # A self-reply and a self-mention on the member's own messages
        # are not relations with anyone and must not count.
        db.execute("INSERT INTO message_people VALUES ('100', '30', 'reply', 1.0)")
        db.execute("INSERT INTO message_people VALUES ('101', '30', 'mention', 1.0)")
        # An outgoing mention on the member's own message does count.
        db.execute("INSERT INTO message_people VALUES ('102', '999', 'mention', 1.0)")
    report = collect_understanding(me_store, "30")
    part = report["parts"]["person_graph"]
    # The fixture's 8 received replies plus the 1 outgoing mention.
    # The member's own author rows and the two self-relations are
    # excluded even though they sit in message_people.
    assert "関係 9 件" in part["detail"]


def test_collect_understanding_server_wiki_counts_only_content_entries(me_store, tmp_path):
    server_wiki_path = tmp_path / "server_wiki.db"
    with sqlite3.connect(server_wiki_path) as wiki:
        wiki.execute(
            "CREATE TABLE server_wiki_entry (guild TEXT, channel TEXT, start INTEGER, "
            "kind TEXT, key TEXT, title TEXT, summary TEXT, sources TEXT)"
        )
        wiki.executemany(
            "INSERT INTO server_wiki_entry VALUES ('100', '10', 0, 'person', '30', ?, ?, '')",
            [("のの", "")] * 20 + [("", "活動の要約")] * 25,
        )
    report = collect_understanding(me_store, "30", server_wiki_path=server_wiki_path)
    part = report["parts"]["server_wiki"]
    # 20 alias-index rows (empty summary) are not articles; the 25
    # content entries are, and the score is 25 / saturation 50 with
    # no free base for merely existing in the index.
    assert "人物記事 25 件" in part["detail"]
    assert part["score"] == pytest.approx(0.5)


def test_collect_understanding_unknown_person_is_zero(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    report = collect_understanding(store, "424242")
    assert report["total"] == 0
    assert report["priority_tier"] is False


def test_record_me_use_registers_and_queues_once(me_store):
    use_count, queued = record_me_use(me_store, "30")
    assert (use_count, queued) == (1, True)
    # A second use while the job is still pending does not stack jobs.
    use_count, queued = record_me_use(me_store, "30")
    assert (use_count, queued) == (2, False)
    report = collect_understanding(me_store, "30")
    assert report["priority_tier"] is True
    assert report["use_count"] == 2


def test_process_job_deepens_and_records_score(me_store):
    record_me_use(me_store, "30")
    done = process_understanding_jobs(me_store, max_jobs=1)
    assert done == 1
    with me_store.connect() as db:
        status, detail = db.execute(
            "SELECT status, detail FROM understanding_jobs"
        ).fetchone()
        last_score, processed = db.execute(
            "SELECT last_score, last_processed_at FROM me_users WHERE user_id = '30'"
        ).fetchone()
    assert status == "done"
    assert "score" in detail
    assert last_score is not None and processed
    # The queue is drained; a further pass completes nothing.
    assert process_understanding_jobs(me_store, max_jobs=1) == 0


def test_stale_tier_members_are_requeued(me_store):
    record_me_use(me_store, "30")
    process_understanding_jobs(me_store, max_jobs=1)
    # Freshly processed members are not stale yet.
    assert enqueue_stale_me_users(me_store, stale_hours=24) == 0
    # With a zero-hour horizon the same member is re-queued.
    assert enqueue_stale_me_users(me_store, stale_hours=0) == 1


def test_me_embed_shows_total_breakdown_and_tier(me_store):
    record_me_use(me_store, "30")
    report = collect_understanding(me_store, "30")
    embed = build_me_embed(report, display_name="のの", queued=True)
    assert f"{report['total']:.0f}%" in embed.description
    field_names = " ".join(field.name for field in embed.fields)
    for needle in ("呼ばれ方", "埋め込み", "別名", "Server Wiki", "人物関係", "裏機能"):
        assert needle in field_names
    field_values = " ".join(field.value for field in embed.fields)
    assert "優先対象" in field_values


def _me_interaction(guild_id=GUILD_ID):
    return SimpleNamespace(
        guild_id=guild_id,
        user=SimpleNamespace(id=30, display_name="のの"),
        response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def test_show_me_defers_sends_embed_and_registers(me_store):
    bot = SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID, model="test-model"),
        server_log=me_store,
    )
    interaction = _me_interaction()
    asyncio.run(show_me(interaction, bot))
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()
    kwargs = interaction.followup.send.call_args.kwargs
    assert kwargs["ephemeral"] is True
    assert "embed" in kwargs
    # The use was recorded and a deepening job queued.
    with me_store.connect() as db:
        assert db.execute("SELECT count(*) FROM me_users").fetchone()[0] == 1
        assert db.execute(
            "SELECT count(*) FROM understanding_jobs WHERE status = 'pending'"
        ).fetchone()[0] == 1


def test_show_me_wrong_guild_refuses_without_registering(me_store):
    bot = SimpleNamespace(
        settings=SimpleNamespace(guild_id=GUILD_ID, model="test-model"),
        server_log=me_store,
    )
    interaction = _me_interaction(guild_id=999)
    asyncio.run(show_me(interaction, bot))
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.call_args.kwargs["ephemeral"] is True
    interaction.response.defer.assert_not_awaited()
    with me_store.connect() as db:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "me_users" not in tables
