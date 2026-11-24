import re


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
