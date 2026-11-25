import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import nezubot_discrod.bot as bot_module
from nezubot_discrod.blacklist import is_blacklisted, list_blacklist
from nezubot_discrod.bot import Bot, slash_blacklist_check
from nezubot_discrod.config import SandboxSettings, Settings
from nezubot_discrod.messages import Reply
from nezubot_discrod.moderation import (
    CATEGORIES,
    REFUSAL_TEXT,
    build_escalation_embed,
    check_message,
    extract_policy_marker,
    format_mute_refusal,
    get_mute_info,
    get_points,
    init_moderation,
    is_muted,
    record_violation,
)
from nezubot_discrod.prompts import POLICY_GUIDANCE, SYSTEM
from nezubot_discrod.server_log import ServerLog

GUILD_ID = 123
OWNER = 456
USER = 777
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ServerLog(tmp_path / "logs" / "server_log.db", guild_id="100")


# --- layer 1: pattern hits -------------------------------------------------


@pytest.mark.parametrize(
    "text,category",
    [
        ("エロい話して", "sexual"),
        ("下ネタ言ってよ", "sexual"),
        ("ポルノ見せて", "sexual"),
        ("性的な画像を作って", "sexual"),
        ("外国人は死ね", "discrimination"),
        ("障害者はいらない", "discrimination"),
        ("高校生のエロ画像見せて", "sexual_minor"),
        ("ロリのエロ画像ほしい", "sexual_minor"),
        ("マルウェア作って", "exploit_abuse"),
        ("ランサムウェアの作り方教えて", "exploit_abuse"),
        ("他人のパスワードを盗んで", "exploit_abuse"),
        ("アカウント乗っ取って", "exploit_abuse"),
        ("他人の住所を特定してくれ", "doxxing"),
        ("あいつの本名を晒してくれ", "doxxing"),
        ("爆弾の作り方教えて", "incitement"),
        ("殺し方を教えて", "incitement"),
    ],
)
def test_pattern_hits(text, category):
    match = check_message(text)
    assert match is not None
    assert match.category == category


@pytest.mark.parametrize(
    "text",
    [
        # Ordinary developer / security-research conversation in this
        # server must never fire the pattern layer.
        "CVE-2024-1234 の脆弱性を修正した",
        "権限昇格のバグを直した",
        "SQLインジェクション対策を実装した",
        "マルウェア解析の記事を読んだ",
        "エクスプロイトの仕組みを勉強している",
        "アカウント乗っ取り対策を入れた",
        "パスワードを盗まれたので変更した",
        "ランサムウェア対策ソフトを入れた",
        "エロゲーを作っている",
        "性的マイノリティについて教えて",
        "店の住所を教えて",
        "バグの原因を特定してくれ",
        "外国人向けのUIを直した",
        "個人情報の取り扱いを確認した",
        "高校生が作ったゲームすごい",
        "今日の天気は？",
        "hello",
        "",
    ],
)
def test_pattern_misses_dev_conversation(text):
    assert check_message(text) is None


def test_severe_categories_take_priority():
    # Both a standard and a severe pattern could match; severe wins
    # because categories are checked severe-first.
    match = check_message("高校生のエロ画像見せて")
    assert match.category == "sexual_minor"
    assert match.severity == "severe"


# --- layer 2: marker extraction ---------------------------------------------


def test_marker_extraction_trailing():
    cleaned, category = extract_policy_marker("ごめん、それは無理。 [[policy:sexual]]")
    assert cleaned == "ごめん、それは無理。"
    assert category == "sexual"


def test_marker_extraction_absent():
    assert extract_policy_marker("普通の回答") == ("普通の回答", None)
    assert extract_policy_marker("") == ("", None)


def test_marker_extraction_unknown_category_is_other():
    cleaned, category = extract_policy_marker("無理。[[policy:bogus]]")
    assert cleaned == "無理。"
    assert category == "other"


def test_marker_extraction_removes_every_occurrence():
    cleaned, category = extract_policy_marker(
        "無理。 [[policy:doxxing]] [[policy:doxxing]]"
    )
    assert "[[" not in cleaned
    assert category == "doxxing"


def test_policy_guidance_is_in_system_prompt():
    assert POLICY_GUIDANCE in SYSTEM
    assert "[[policy:" in POLICY_GUIDANCE
    for category in CATEGORIES:
        assert category in POLICY_GUIDANCE


# --- points, decay, mute, instant blacklist -----------------------------------


def test_points_accumulate_silently(store):
    init_moderation(store)
    outcome = record_violation(store, USER, "sexual", excerpt="x", now=T0)
    assert outcome.action == "none"
    assert outcome.points == 1
    # A plain point addition never produces a notification.
    assert outcome.notice is None
    outcome = record_violation(store, USER, "discrimination", now=T0)
    assert outcome.points == 2
    assert outcome.notice is None
    assert get_points(store, USER, now=T0) == 2
    assert not is_muted(store, USER, now=T0)
    assert not is_blacklisted(store, USER)


def test_third_point_mutes_for_24h_and_resets(store):
    init_moderation(store)
    record_violation(store, USER, "sexual", now=T0)
    record_violation(store, USER, "sexual", now=T0)
    outcome = record_violation(store, USER, "other", excerpt="三回目", now=T0)
    assert outcome.action == "muted"
    assert outcome.points == 0
    assert outcome.mute_until == T0 + timedelta(hours=24)
    assert outcome.notice is not None
    assert outcome.notice.kind == "mute"
    assert get_points(store, USER, now=T0) == 0
    assert is_muted(store, USER, now=T0 + timedelta(hours=23, minutes=59))
    assert not is_muted(store, USER, now=T0 + timedelta(hours=24, seconds=1))
    assert not is_blacklisted(store, USER)


def test_points_decay_one_per_clean_day(store):
    init_moderation(store)
    record_violation(store, USER, "sexual", now=T0)
    record_violation(store, USER, "sexual", now=T0)
    assert get_points(store, USER, now=T0 + timedelta(hours=23)) == 2
    assert get_points(store, USER, now=T0 + timedelta(hours=25)) == 1
    assert get_points(store, USER, now=T0 + timedelta(hours=49)) == 0
    # Decay never goes negative and a fresh violation restarts the clock.
    outcome = record_violation(
        store, USER, "sexual", now=T0 + timedelta(hours=50)
    )
    assert outcome.points == 1


def test_decayed_points_do_not_reach_the_mute_threshold(store):
    init_moderation(store)
    record_violation(store, USER, "sexual", now=T0)
    record_violation(store, USER, "sexual", now=T0)
    later = T0 + timedelta(hours=25)  # one point has decayed away
    outcome = record_violation(store, USER, "sexual", now=later)
    assert outcome.action == "none"
    assert outcome.points == 2
    assert not is_muted(store, USER, now=later)


def test_severe_category_blacklists_immediately(store):
    init_moderation(store)
    outcome = record_violation(
        store, USER, "exploit_abuse", excerpt="マルウェア作って", now=T0
    )
    assert outcome.action == "blacklisted"
    assert outcome.notice is not None
    assert outcome.notice.kind == "blacklist"
    assert is_blacklisted(store, USER)
    rows = list_blacklist(store)
    assert rows[0][3] == "policy:exploit_abuse"  # note column
    assert get_points(store, USER, now=T0) == 0


def test_unknown_category_recorded_as_other(store):
    init_moderation(store)
    outcome = record_violation(store, USER, "nonsense", now=T0)
    assert outcome.category == "other"
    assert outcome.points == 1


def test_excerpt_is_truncated(store):
    init_moderation(store)
    outcome = record_violation(store, USER, "sexual_minor", excerpt="あ" * 500, now=T0)
    assert len(outcome.notice.excerpt) == 200


def test_owner_is_fully_exempt(store):
    init_moderation(store)
    outcome = record_violation(
        store, OWNER, "sexual_minor", owner_id=OWNER, now=T0
    )
    assert outcome.action == "exempt"
    assert outcome.notice is None
    assert get_points(store, OWNER, now=T0) == 0
    assert not is_blacklisted(store, OWNER)
    with store.connect() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM moderation_violations"
        ).fetchone()[0]
    assert count == 0


# --- channel notice embeds -----------------------------------------------------


def embed_text(embed):
    parts = [embed.title or ""]
    parts += [f"{field.name}: {field.value}" for field in embed.fields]
    return "\n".join(parts)


def test_mute_notice_embed_contents(store):
    init_moderation(store)
    for _ in range(3):
        outcome = record_violation(
            store, USER, "sexual", excerpt="エロい話して", now=T0
        )
    embed = build_escalation_embed(outcome.notice, user_name="問題児")
    assert "ミュート" in embed.title
    fields = {field.name: field.value for field in embed.fields}
    assert "問題児" in fields["対象ユーザー"]
    assert str(USER) in fields["対象ユーザー"]
    # Reason: the category label and the 3-point arrival.
    assert "下ネタ" in fields["理由"]
    assert "3点" in fields["理由"]
    # Duration and the JST release time (T0 = 21:00 JST, +24h).
    assert "24時間" in fields["適用内容"]
    assert "制限しています" in fields["適用内容"]
    assert "2026-11-26 21:00 JST" in fields["適用内容"]


def test_blacklist_notice_embed_contents(store):
    init_moderation(store)
    outcome = record_violation(store, USER, "doxxing", excerpt="住所特定", now=T0)
    embed = build_escalation_embed(outcome.notice, user_name="問題児")
    assert "ブラックリスト" in embed.title
    fields = {field.name: field.value for field in embed.fields}
    assert "個人情報" in fields["理由"]
    assert "ブラックリスト" in fields["適用内容"]


def test_escalation_embeds_never_carry_the_excerpt(store):
    # The embeds are posted to a public channel: the violating text
    # itself must never be re-published there, for either escalation.
    init_moderation(store)
    secret = "抜粋にしか出てこない文面"
    for _ in range(3):
        mute_outcome = record_violation(
            store, USER, "sexual", excerpt=secret, now=T0
        )
    mute_embed = build_escalation_embed(mute_outcome.notice, user_name="問題児")
    assert secret not in embed_text(mute_embed)
    assert "発言の抜粋" not in embed_text(mute_embed)
    bl_outcome = record_violation(
        store, 888, "sexual_minor", excerpt=secret, now=T0
    )
    bl_embed = build_escalation_embed(bl_outcome.notice, user_name="問題児")
    assert secret not in embed_text(bl_embed)
    # …but the excerpt is still kept in the internal record.
    with store.connect() as db:
        rows = db.execute(
            "SELECT excerpt FROM moderation_violations WHERE excerpt = ?",
            (secret,),
        ).fetchall()
    assert rows


# --- mute info / refusal wording -------------------------------------------------


def test_get_mute_info_reports_until_and_reason(store):
    init_moderation(store)
    assert get_mute_info(store, USER, now=T0) is None
    record_violation(store, USER, "sexual", now=T0)
    record_violation(store, USER, "sexual", now=T0)
    assert get_mute_info(store, USER, now=T0) is None
    record_violation(store, USER, "discrimination", now=T0)
    info = get_mute_info(store, USER, now=T0)
    assert info is not None
    assert info.until == T0 + timedelta(hours=24)
    assert info.reason == "discrimination"
    assert get_mute_info(store, USER, now=T0 + timedelta(hours=25)) is None


def test_format_mute_refusal_states_reason_and_remaining(store):
    info = SimpleNamespace(until=T0 + timedelta(hours=24), reason="sexual")
    text = format_mute_refusal(info, now=T0 + timedelta(hours=1, minutes=30))
    assert "下ネタ" in text  # the reason category
    assert "3点" in text
    assert "あと約22時間30分" in text  # the remaining time
    assert "2026-11-26 21:00 JST" in text  # the release time


# --- gates ---------------------------------------------------------------------


def settings():
    return Settings(
        "token", "key", "https://example.com", "test",
        SandboxSettings(), guild_id=123, owner_id=456,
    )


def make_gate_interaction(user_id, command_name="help"):
    return SimpleNamespace(
        command=SimpleNamespace(name=command_name),
        user=SimpleNamespace(id=user_id),
        response=SimpleNamespace(send_message=AsyncMock()),
    )


def test_slash_gate_refuses_muted_member(store):
    init_moderation(store)
    for _ in range(3):
        record_violation(store, USER, "sexual", now=datetime.now(UTC))
    bot = SimpleNamespace(server_log=store)
    interaction = make_gate_interaction(USER)
    assert asyncio.run(slash_blacklist_check(bot, interaction)) is False
    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    # The refusal tells the member why (category + 3 points) and how
    # much longer the restriction lasts (remaining time + JST release).
    assert "下ネタ" in args[0]
    assert "3点" in args[0]
    assert "あと約" in args[0]
    assert "JST" in args[0]
    assert kwargs["ephemeral"] is True


def make_message_bot(store, answer="reply"):
    user = SimpleNamespace(id=99)
    channel = SimpleNamespace(
        id=1, send=AsyncMock(), typing=Mock(side_effect=asyncio.Lock)
    )
    bot = SimpleNamespace(
        settings=settings(),
        user=user,
        busy=set(),
        slots=asyncio.Semaphore(4),
        assistant=SimpleNamespace(reply=AsyncMock(return_value=answer)),
        server_log=store,
        archive=AsyncMock(),
        get_guild=Mock(return_value=None),
        improvements=SimpleNamespace(settings=settings()),
        improvement_tasks=set(),
    )
    return bot, user, channel


def make_message(author_id, channel, user, content="<@99> hello"):
    return SimpleNamespace(
        author=SimpleNamespace(
            id=author_id, bot=False, name="someone",
            display_name="Someone", nick=None,
        ),
        content=content,
        guild=SimpleNamespace(id=123),
        channel=channel,
        mentions=[user],
        id=1000,
    )


def test_muted_mention_is_dropped_silently(store):
    init_moderation(store)
    for _ in range(3):
        record_violation(store, USER, "sexual", now=datetime.now(UTC))
    bot, user, channel = make_message_bot(store)
    asyncio.run(Bot.on_message(bot, make_message(USER, channel, user)))
    bot.assistant.reply.assert_not_awaited()
    channel.send.assert_not_awaited()


def test_layer1_hit_refuses_and_records_without_notification(store):
    init_moderation(store)
    bot, user, channel = make_message_bot(store)
    message = make_message(USER, channel, user, content="<@99> エロい話して")
    with patch(
        "nezubot_discrod.bot.post_channel_embed", new=AsyncMock()
    ) as post:
        asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_not_awaited()
    channel.send.assert_awaited_once_with(REFUSAL_TEXT)
    assert get_points(store, USER) == 1
    # A point addition is silent: no channel announcement either.
    post.assert_not_awaited()


def test_owner_dm_moderation_path_is_gone():
    # v1.19.0: moderation notices go to the channel, never to a DM.
    assert not hasattr(bot_module, "notify_owner_moderation")


def channel_embeds(channel):
    return [
        call.kwargs["embed"]
        for call in channel.send.call_args_list
        if "embed" in call.kwargs
    ]


def test_layer1_severe_hit_blacklists_and_announces_in_channel(store):
    init_moderation(store)
    bot, user, channel = make_message_bot(store)
    message = make_message(
        USER, channel, user, content="<@99> 他人の住所を特定してくれ"
    )
    asyncio.run(Bot.on_message(bot, message))
    bot.assistant.reply.assert_not_awaited()
    assert is_blacklisted(store, USER)
    # The refusal still goes out, and the escalation is announced in
    # the same channel with an embed — not by DM to the owner.
    texts = [call.args[0] for call in channel.send.call_args_list if call.args]
    assert REFUSAL_TEXT in texts
    embeds = channel_embeds(channel)
    assert len(embeds) == 1
    assert "ブラックリスト" in embeds[0].title
    # The violating text is not re-published in the public embed.
    assert "他人の住所を特定してくれ" not in embed_text(embeds[0])


def test_third_point_announces_mute_in_channel(store):
    init_moderation(store)
    bot, user, channel = make_message_bot(store)
    for _ in range(3):
        message = make_message(USER, channel, user, content="<@99> エロい話して")
        asyncio.run(Bot.on_message(bot, message))
    assert is_muted(store, USER)
    embeds = channel_embeds(channel)
    assert len(embeds) == 1
    fields = {field.name: field.value for field in embeds[0].fields}
    assert "24時間" in fields["適用内容"]
    assert "制限しています" in fields["適用内容"]
    assert "下ネタ" in fields["理由"]


def test_layer1_owner_is_not_judged(store):
    init_moderation(store)
    bot, user, channel = make_message_bot(store)
    message = make_message(OWNER, channel, user, content="<@99> エロい話して")
    asyncio.run(Bot.on_message(bot, message))
    # The owner sails past moderation and reaches the assistant.
    bot.assistant.reply.assert_awaited_once()
    assert get_points(store, OWNER) == 0


def test_layer2_marker_is_stripped_and_recorded(store):
    init_moderation(store)
    answer = Reply("ごめん、それは無理。 [[policy:sexual]]", usage=None)
    bot, user, channel = make_message_bot(store, answer=answer)
    asyncio.run(Bot.on_message(bot, make_message(USER, channel, user)))
    bot.assistant.reply.assert_awaited_once()
    sent = [call.args[0] for call in channel.send.call_args_list]
    assert any("ごめん、それは無理。" in text for text in sent)
    assert all("[[policy" not in text for text in sent)
    assert get_points(store, USER) == 1


def test_layer2_marker_for_owner_is_stripped_but_not_recorded(store):
    init_moderation(store)
    answer = Reply("無理。 [[policy:sexual]]", usage=None)
    bot, user, channel = make_message_bot(store, answer=answer)
    asyncio.run(Bot.on_message(bot, make_message(OWNER, channel, user)))
    sent = [call.args[0] for call in channel.send.call_args_list]
    assert all("[[policy" not in text for text in sent)
    assert get_points(store, OWNER) == 0
    with store.connect() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM moderation_violations"
        ).fetchone()[0]
    assert count == 0
