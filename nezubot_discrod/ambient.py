import re


def ambient_excluded(conversation):
    """Mechanical pre-filter for uninvited (ambient) participation.

    A message that @-mentions specific other people — another member,
    another bot — is addressed to them, not an open question the bot
    may join, however interesting it looks (D-1: a question addressed
    to another bot; D-3: a 「どう」 addressed to a different member).
    The conversation layer's mentions list already excludes the bot
    itself, so any entry in it names somebody else. The prompt-level
    arbiter keeps judging the remaining, genuinely unaddressed cases;
    this filter only removes the mechanically decidable ones before
    a model call is spent on them."""
    if not conversation:
        return False
    return bool(conversation.get("mentions"))


def skip_answer(text):
    return (
        not text
        or not text.strip()
        or bool(
            re.fullmatch(
                r"[\s(（\[「『`*]*skip[\s)）\]」』.。!！`*]*",
                text,
                re.IGNORECASE,
            )
        )
    )
