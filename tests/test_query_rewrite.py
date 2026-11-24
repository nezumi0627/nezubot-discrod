from nezubot_discrod.query_rewrite import (
    build_context_query,
    needs_rewrite,
    rewrite_query,
)

CLAIM = "キャッシュを全件走査する時点でヒットしないキーが絶対にでてきて非効率だ"


def state_with_person():
    return {
        "topic": "FTSが検索予算を使い切る問題",
        "active_subject": {"kind": "person", "id": "555", "label": "ZephyrusZ",
                           "author_id": None},
    }


def conversation_with_target():
    return {
        "reply_chain": [
            {"author": "ゆるふわモカ", "author_id": "302", "message_id": "90",
             "content": CLAIM},
        ],
        "reply_target": {
            "author": "ゆるふわモカ", "author_id": "302", "message_id": "90",
            "content": CLAIM,
        },
        "transcript": (
            "[21:51] ゆるふわモカ: " + CLAIM + "\n"
            "[21:51] sora_server: なら自己学習つくったら\n"
            "[21:52] sora_server: これ解決できないの ← いま応答する発言"
        ),
    }


def test_needs_rewrite_detection():
    assert needs_rewrite("これ解決できないの")
    assert needs_rewrite("俺とかはどう")
    assert needs_rewrite("さっきのやつ見せて")
    assert needs_rewrite("それ以外は？")
    assert needs_rewrite("どこ住み？")
    assert not needs_rewrite("Rustの所有権について詳しく教えて")
    assert not needs_rewrite("vexってどう思う？")
    assert not needs_rewrite("")


def test_rewrite_demonstrative_uses_reply_target_words():
    rewritten = rewrite_query(
        "これ解決できないの", state=None, conversation=conversation_with_target(),
    )
    assert rewritten is not None
    assert CLAIM[:20] in rewritten
    assert rewritten.endswith("これ解決できないの")


def test_rewrite_followup_uses_subject_label():
    rewritten = rewrite_query(
        "どこ住み？", state=state_with_person(), conversation={"reply_chain": []},
    )
    assert rewritten == "ZephyrusZ どこ住み？"


def test_rewrite_self_compare_uses_speaker_and_topic():
    rewritten = rewrite_query(
        "俺とかはどう", state=state_with_person(), conversation={"reply_chain": []},
        speaker_name="テストユーザー",
    )
    assert rewritten is not None
    assert rewritten.startswith("テストユーザー")
    assert "FTS" in rewritten


def test_rewrite_returns_none_without_context_or_need():
    assert rewrite_query("これ解決できないの", state=None, conversation=None) is None
    assert rewrite_query(
        "Rustの所有権について詳しく教えて", state=state_with_person(),
        conversation={"reply_chain": []},
    ) is None
    # Subject already named in the question: nothing to add.
    assert rewrite_query(
        "ZephyrusZってどこ住み？", state=state_with_person(),
        conversation={"reply_chain": []},
    ) is None


def test_context_query_from_conversation():
    query = build_context_query(conversation_with_target(), state_with_person())
    assert query is not None
    assert CLAIM[:20] in query
    assert "自己学習" in query
    assert "いま応答する発言" not in query
    assert build_context_query({}, None) is None
    assert build_context_query({"transcript": ""}, None) is None
