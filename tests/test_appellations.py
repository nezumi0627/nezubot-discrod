import asyncio
import sqlite3
from time import perf_counter

import pytest
from test_server_log import record

import nezubot_discrod.server_log_context as context_module
from nezubot_discrod.appellations import (
    AppellationIndex,
    appellation_list,
    extract_appellations,
    mine_appellations,
    person_summary,
    qualified_appellations,
    representative_appellation,
)
from nezubot_discrod.server_log import ServerLog
from nezubot_discrod.server_log_context import pre_retrieve, resolve_people
from nezubot_discrod.server_log_tools import LogTools


def test_extract_honorific_and_vocative_forms():
    assert extract_appellations("<@30> タヌキさんもpoypはじめよ") == ["タヌキ"]
    assert extract_appellations("タヌキってどれ？") == ["タヌキ"]
    assert "タヌキ" in extract_appellations("日本人のタヌキくんはどう思う？")
    # The leading name of a different person is extracted as a raw
    # candidate; the miner's known-name guard rejects it (below).
    assert extract_appellations("にょらいとタヌキって一緒なんだ") == ["にょらい"]


def test_extract_rejects_pronouns_and_noise():
    assert extract_appellations("お前はどう思う？") == []
    assert extract_appellations("あいつさん来た") == []
    assert extract_appellations("今日は暑い") == []
    assert extract_appellations("") == []
    assert extract_appellations("あ") == []


def _index():
    return AppellationIndex([
        ("30", "999", "タヌキ", 5),
        ("30", "888", "タヌキ", 4),
        ("30", "999", "のの", 3),
        ("31", "999", "あきら", 2),
        ("32", "888", "あきら", 3),
        ("31", "777", "あきら", 2),
    ])


def test_index_global_and_caller_match():
    index = _index()
    # A name only one person is called by resolves for anyone.
    assert index.match("タヌキって知ってる？", 424242) == ["30"]
    # The caller's own usage is honoured.
    assert index.match("タヌキって知ってる？", 999) == ["30"]
    # A genuinely shared name stays unresolved globally...
    assert index.match("あきらって知ってる？", 424242) == []
    # ...but a caller who uses it for exactly one person resolves.
    assert index.match("あきらって知ってる？", 999) == ["31"]
    assert index.match("関係ない話", 999) == []


def test_index_top_appellation():
    index = _index()
    assert index.global_top("30") == "タヌキ"
    assert index.top_appellation("30", 999) == "タヌキ"
    assert index.top_appellation("31", 999) == "あきら"
    assert index.global_top("999") is None


@pytest.fixture
def people_store(tmp_path):
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
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO message_people VALUES (?, '30', 'reply', 1.0)",
            [(str(200 + index),) for index in range(5)],
        )
        db.execute(
            "INSERT INTO users (user_id, guild_id, username, display_name, nickname) "
            "VALUES ('30', '100', 'tanuki_dev', 'のの', '')"
        )
    return store


def test_mining_accumulates_and_is_incremental(people_store):
    stats = mine_appellations(people_store)
    assert stats["pairs"] == 5
    with people_store.connect() as db:
        rows = db.execute(
            "SELECT target_id, caller_id, appellation, count FROM appellations"
        ).fetchall()
    assert [tuple(row) for row in rows] == [("30", "999", "タヌキ", 5)]
    # A second run mines nothing new: counts do not double.
    stats = mine_appellations(people_store)
    assert stats["messages"] == 0
    with people_store.connect() as db:
        count = db.execute("SELECT count FROM appellations").fetchone()[0]
    assert count == 5


def test_screenshot_reproduction_tamane_resolves_first_try(people_store):
    mine_appellations(people_store)
    # The incident: 「タヌキって知ってる？」 in the channel, first
    # turn, no alias row for the nickname anywhere — resolution must
    # come from the appellation graph alone.
    assert resolve_people(people_store, [10], "タヌキって知ってる？", 999) == ["30"]
    log = LogTools(people_store, [10], caller_id=999)
    result = asyncio.run(pre_retrieve(log, "タヌキって知ってる？", timeout=5))
    assert result.diagnostics["people"] == ["30"]
    assert result.diagnostics["chunks"].get("person_summary", 0) == 1
    assert "人物情報" in result
    assert "のの" in result
    assert "タヌキ" in result


def test_budget_interrupt_keeps_verified_appellation(people_store, monkeypatch):
    # The production incident: the appellation had already matched
    # when the budget's progress handler interrupted the (cold, heavy)
    # alias snapshot load, and the whole resolution degraded to
    # nobody. Verified matches must survive the interrupt.
    mine_appellations(people_store)

    def interrupted_snapshot(store, channel_ids, db):
        raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(context_module, "_alias_snapshot", interrupted_snapshot)
    deadline = perf_counter() + 60.0
    assert resolve_people(
        people_store, [10], "タヌキって知ってる？", 999, deadline=deadline
    ) == ["30"]


def test_targeted_lookup_resolves_without_full_index(people_store, monkeypatch):
    # Cold-cache conditions (the incident): the full graph load is
    # unavailable, but the question presents the name as an address
    # form, so the targeted slice alone must resolve it.
    mine_appellations(people_store)
    monkeypatch.setattr(context_module, "_appellation_index", lambda *a: None)
    assert resolve_people(people_store, [10], "タヌキって知ってる？", 999) == ["30"]


def test_mining_rejects_other_peoples_names(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    value = record(1, channel="10", text="にょらいの発言")
    value["id"] = "300"
    value["author"] = {"id": "40", "username": "にょらい", "bot": False}
    rows.append(value)
    for index in range(3):
        value = record(1, channel="10", text="にょらいとタヌキって一緒なんだ")
        value["id"] = str(400 + index)
        value["author"] = {"id": "999", "username": "testuser", "bot": False}
        rows.append(value)
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO message_people VALUES (?, '30', 'reply', 1.0)",
            [(str(400 + index),) for index in range(3)],
        )
    mine_appellations(store)
    with store.connect() as db:
        found = db.execute(
            "SELECT count(*) FROM appellations WHERE appellation = 'にょらい'"
        ).fetchone()[0]
    assert found == 0


def test_extract_rejects_context_words_and_fragments():
    # Sentence openers a reply happens to start with are not names,
    # however often they recur (the production graph had each of
    # these as dozens of people's "appellation").
    assert extract_appellations("普通に考えてそれは違うと思う") == []
    assert extract_appellations("ちなみに聞くけど、それどうなった？") == []
    assert extract_appellations("こんにちは、元気にしてた？") == []
    assert extract_appellations("確か明日だよね") == []
    # The bot's own error boilerplate, whose leading fragment 返事
    # became the owner's top "appellation" in production.
    assert extract_appellations("返事を作る途中でモデル側のエラーが起きたよ") == []
    # Markup/symbol fragments are not name-shaped.
    assert extract_appellations("**ndaさん、署名して添付しました") == []
    assert extract_appellations("??さん、それはどう？") == []
    # Genuine names still extract, Japanese and ASCII alike.
    assert extract_appellations("タヌキさん、それはどう？") == ["タヌキ"]
    assert "poyp" in extract_appellations("poypさん、聞いて")


def test_mining_skips_bot_callers(tmp_path):
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = []
    value = record(1, channel="10", text="tanuki_devの発言")
    value["id"] = "100"
    value["author"] = {"id": "30", "username": "tanuki_dev", "bot": False}
    rows.append(value)
    # A bot "calling" the target — its boilerplate is not evidence.
    for index in range(4):
        value = record(1, channel="10", text=f"返事を作る途中でエラー {index}")
        value["id"] = str(200 + index)
        value["author"] = {"id": "40", "username": "somebot", "bot": True}
        rows.append(value)
    # A human caller in the same shape the bot used.
    for index in range(3):
        value = record(1, channel="10", text=f"タヌキさん、それはどう？ {index}")
        value["id"] = str(300 + index)
        value["author"] = {"id": "999", "username": "testuser", "bot": False}
        rows.append(value)
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO message_people VALUES (?, '30', 'reply', 1.0)",
            [(str(200 + index),) for index in range(4)]
            + [(str(300 + index),) for index in range(3)],
        )
    stats = mine_appellations(store)
    assert stats["pairs"] == 3
    with store.connect() as db:
        found = db.execute(
            "SELECT target_id, caller_id, appellation, count FROM appellations"
        ).fetchall()
    assert [tuple(row) for row in found] == [("30", "999", "タヌキ", 3)]


def test_index_global_top_prefers_spread_over_single_caller_volume():
    # The production failure shape: one caller's repeated fragment
    # (16 uses) out-volumes the real name used by several people.
    index = AppellationIndex([
        ("30", "999", "へんじ", 16),
        ("30", "888", "タヌキ", 5),
        ("30", "777", "タヌキ", 4),
    ])
    assert index.global_top("30") == "タヌキ"
    # With no spread anywhere, the single-caller name still stands.
    solo = AppellationIndex([("30", "999", "タヌキ", 5)])
    assert solo.global_top("30") == "タヌキ"


def test_appellation_list_requires_spread_for_global_names():
    index = AppellationIndex([
        ("30", "999", "へんじ", 16),
        ("30", "888", "タヌキ", 5),
        ("30", "777", "タヌキ", 4),
    ])
    assert [name for name, _total in appellation_list(index, "30")] == ["タヌキ"]
    # The caller who uses the single-caller name still sees it as
    # their own name for the person.
    own = [name for name, _total in appellation_list(index, "30", caller_id="999")]
    assert "へんじ" in own and "タヌキ" in own


def test_qualified_appellations_need_spread_and_volume():
    rows = [
        ("30", "999", "タヌキ", 5),
        ("30", "888", "タヌキ", 3),
        ("30", "999", "へんじ", 16),   # one caller only: not established
        ("30", "777", "よび", 1),      # one-off fragment
        ("30", "777", "すこ", 2),      # spread but below the floor
        ("30", "888", "すこ", 1),
    ]
    qualified = qualified_appellations(rows)
    # すこ totals 3 across two callers and does qualify.
    assert qualified == {"タヌキ": (8, 2), "すこ": (3, 2)}
    assert representative_appellation(qualified) == "タヌキ"
    assert representative_appellation({}) is None


def test_person_summary_wording(people_store):
    mine_appellations(people_store)
    with people_store.connect() as db:
        index = AppellationIndex.load(db)
        summary = person_summary(db, index, "30", 999)
    assert summary is not None
    assert "Discord名は「のの」" in summary
    assert "ユーザー名 tanuki_dev" in summary
    assert "大半からは「タヌキ」" in summary


# --- Standalone mining (messages with no determined addressee) ---
#
# The owner's /me 呼ばれ方 stayed at zero while every other score
# filled in: people address him by bare vocative in open channel flow
# (「テストユーザーさん、…」 with no reply or mention attached), and the
# addressed miner structurally never sees those messages. The
# standalone pass mines them, anchored on known profile names only.


def _msg(number, author_id, text, minute=0):
    value = record(number)
    value["id"] = str(number)
    value["author"] = {"id": author_id, "username": f"user{author_id}",
                       "bot": False}
    value["content"] = text
    value["timestamp"] = f"2026-01-01T00:{minute:02d}:00+00:00"
    return value


def _standalone_store(tmp_path):
    """A store where テストユーザー (user 40) is a known profile name and
    callers use it in messages that address nobody mechanically."""
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = [
        _msg(1, "999", "テストユーザーさん、その件どうなった？", 1),
        _msg(2, "888", "テストユーザーくん見てる？", 2),
        _msg(3, "777", "テストユーザーちゃん、おはよう", 3),
        _msg(4, "999", "テストユーザーさんありがとう", 4),
        _msg(5, "888", "テストユーザーくん、それな", 5),
        _msg(6, "777", "今日は暑いね", 6),       # no name at all
        _msg(7, "999", "たろうさん来た", 7),     # unknown name
        _msg(8, "40", "テストユーザーさんって俺のこと？", 8),  # self-address
    ]
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO author_aliases VALUES (?, ?, ?, ?, ?)",
            [("100", "10", "40", "テストユーザー", "テストユーザー"),
             ("100", "10", "40", "testuser0627", "testuser0627")])
    return store


def test_standalone_mining_attributes_known_names(tmp_path):
    store = _standalone_store(tmp_path)
    stats = mine_appellations(store)
    assert stats["standalone_messages"] == 8
    assert stats["standalone_pairs"] == 5
    with store.connect() as db:
        rows = db.execute(
            "SELECT caller_id, appellation, count FROM appellations "
            "WHERE target_id = '40' ORDER BY caller_id").fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("777", "テストユーザー", 1), ("888", "テストユーザー", 2), ("999", "テストユーザー", 2)]
    # Qualified: 5 uses across 3 callers — the /me score can fill in.
    qualified = qualified_appellations(
        [("40", r[0], r[1], r[2]) for r in rows])
    assert qualified == {"テストユーザー": (5, 3)}
    # Second run is incremental: nothing new is mined.
    again = mine_appellations(store)
    assert again["standalone_messages"] == 0
    assert again["standalone_pairs"] == 0


def test_standalone_mining_skips_addressed_messages(tmp_path):
    # A message WITH a reply target belongs to the addressed pass;
    # the standalone pass must not double-count it.
    store = _standalone_store(tmp_path)
    with store.connect() as db:
        db.execute(
            "INSERT INTO message_people VALUES ('1', '40', 'reply', 1.0)")
    stats = mine_appellations(store)
    assert stats["standalone_messages"] == 7
    with store.connect() as db:
        total = db.execute(
            "SELECT sum(count) FROM appellations WHERE target_id = '40'"
        ).fetchone()[0]
    # 4 standalone + 1 addressed = 5, not 6.
    assert total == 5


def test_standalone_shared_name_goes_to_dominant_owner(tmp_path):
    # A name carried by a main account and its near-silent alt is the
    # main account's name (the owner's real shape: テストユーザー is also the
    # display name of his 316-message alt).
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = [_msg(i, "50", f"発言{i}", i % 60) for i in range(1, 61)]
    rows += [_msg(100, "51", "サブの発言", 40)]
    rows += [
        _msg(200, "999", "テストユーザーさん、確認して", 41),
        _msg(201, "888", "テストユーザーくん頼む", 42),
    ]
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO author_aliases VALUES (?, ?, ?, ?, ?)",
            [("100", "10", "50", "テストユーザー", "テストユーザー"),
             ("100", "10", "51", "テストユーザー", "テストユーザー")])
    mine_appellations(store)
    with store.connect() as db:
        targets = {r[0] for r in db.execute(
            "SELECT DISTINCT target_id FROM appellations")}
    assert targets == {"50"}


def test_standalone_shared_name_between_peers_not_attributed(tmp_path):
    # Two active people genuinely sharing a name: nobody is credited
    # (the たぬ shape — attribution would manufacture a false fact).
    store = ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")
    rows = [_msg(i, "50", f"発言{i}", i % 60) for i in range(1, 61)]
    rows += [_msg(100 + i, "51", f"発言{i}", i % 60) for i in range(1, 61)]
    rows += [_msg(300, "999", "たぬさん来た", 50)]
    store.upsert_many(rows)
    with store.connect() as db:
        db.executemany(
            "INSERT INTO author_aliases VALUES (?, ?, ?, ?, ?)",
            [("100", "10", "50", "たぬ", "たぬ"),
             ("100", "10", "51", "たぬ", "たぬ")])
    mine_appellations(store)
    with store.connect() as db:
        count = db.execute("SELECT count(*) FROM appellations").fetchone()[0]
    assert count == 0
