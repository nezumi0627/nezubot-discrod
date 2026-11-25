import json

from nezubot_discrod.web_guard import OutboundGuard

PRIVATE = (
    "2026-11-24 21:03 タヌキ: 来週の月曜日にサーバーの移行作業を行う予定で、"
    "障害が出たら即座にロールバックすることになりました。"
    "詳細は https://internal.example.org/secret-migration-plan-2026 です "
    "(message 123456789012345679)"
)
FRAGMENT = (
    "来週の月曜日にサーバーの移行作業を行う予定で、"
    "障害が出たら即座にロールバックすることになりました"
)


def guard():
    value = OutboundGuard()
    value.note_private(PRIVATE)
    return value


def test_verbatim_fragment_is_blocked_in_query_and_url():
    value = guard()
    assert value.check(FRAGMENT, kind="query")
    assert value.check("https://example.org/?q=" + FRAGMENT, kind="url")


def test_normalized_variants_of_fragment_are_blocked():
    value = OutboundGuard()
    value.note_private(
        "The quarterly migration window opens on Monday and the "
        "rollback plan is documented in the runbook"
    )
    messy = (
        "  THE QUARTERLY   MIGRATION WINDOW OPENS ON MONDAY AND THE "
        "ROLLBACK PLAN IS DOCUMENTED IN THE RUNBOOK "
    )
    assert value.check(messy, kind="query")


def test_short_or_paraphrased_queries_pass():
    value = guard()
    assert value.check("サーバー移行 ニュース", kind="query") is None
    assert value.check("来週の月曜", kind="query") is None
    assert value.check("今日のニュース", kind="query") is None
    assert value.check("https://example.org/news/1", kind="url") is None


def test_long_token_from_private_is_blocked():
    value = guard()
    assert value.check(
        "https://internal.example.org/secret-migration-plan-2026", kind="query")
    assert value.check(
        "see https://internal.example.org/secret-migration-plan-2026", kind="url")


def test_message_id_blocked_in_query_but_not_in_url():
    value = guard()
    assert value.check("これ 123456789012345679 を調べて", kind="query")
    # Structural IDs inside a fetch URL (e.g. a Discord message link)
    # are not treated as a leak on their own.
    assert value.check(
        "https://discord.com/channels/111111111111111111/123456789012345679/"
        "123456789012345679", kind="url") is None


def test_user_words_are_exempt():
    value = OutboundGuard()
    value.note_user_text(f"「{FRAGMENT}」ってどういう意味？")
    value.note_private(PRIVATE)
    assert value.check(FRAGMENT, kind="query") is None


def test_seed_from_user_payload_separates_message_from_context():
    value = OutboundGuard()
    payload = {
        "message": "検索して今日のニュースを",
        "retrieved_server_context": {"excerpts": PRIVATE},
        "recent_messages": [{"text": "他人の発言"}],
    }
    value.seed_from_user_payload(json.dumps(payload, ensure_ascii=False))
    assert value.check(FRAGMENT, kind="query")
    assert value.check("検索して今日のニュースを", kind="query") is None


def test_seed_from_wrapped_message_list():
    value = OutboundGuard()
    payload = {"message": "hi", "member_wiki": PRIVATE}
    wrapped = json.dumps([{"role": "user", "content": json.dumps(payload)}])
    value.seed_from_user_payload(wrapped)
    assert value.check(FRAGMENT, kind="query")


def test_check_call_covers_each_web_tool():
    value = guard()
    assert value.check_call("web_search", {"query": FRAGMENT})
    assert value.check_call("wikipedia_search", {"query": FRAGMENT})
    assert value.check_call("wikipedia_read", {"title": FRAGMENT})
    assert value.check_call("read_web_page", {"url": "https://x.example/?q=" + FRAGMENT})
    assert value.check_call("web_search", {"query": "普通の検索"}) is None
    assert value.check_call("read_web_page", {"url": "https://example.org/a"}) is None
    assert value.check_call("get_weather", {"place": FRAGMENT}) is None
    assert "blocked" in value.check_call("web_search", {"query": FRAGMENT})
