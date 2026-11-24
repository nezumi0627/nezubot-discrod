import pytest

from nezubot_discrod.web_tools import decode_text, readable


@pytest.mark.parametrize("html", [
    "<head><title>Hidden</title><body><p>Visible</p>",
    "<head><title>Hidden</title><p>Visible</p>",
    "<head><script>hidden()<body><p>Visible</p>",
    "<style>hidden{}<div>Visible</div>",
])
def test_missing_hidden_end_tags_do_not_hide_body(html):
    assert readable(html) == "Visible"


def test_complete_script_and_style_stay_hidden():
    assert readable('<script>const x = "<body><p>secret";</script>'
                    '<style>p:after{content:"<body>"}</style><p>Visible</p>') == "Visible"


@pytest.mark.parametrize("meta,encoding", [
    ('<meta charset="Shift_JIS">', "shift_jis"),
    ('<meta content="text/html; charset=Shift_JIS" http-equiv="Content-Type">', "shift_jis"),
    ('<meta charset="EUC-JP">', "euc_jp"),
])
def test_html_meta_encoding_without_header(meta, encoding):
    html = meta + "<body><p>日本語のページです</p>"
    assert decode_text(html.encode(encoding), "text/html", None) == html


def test_header_encoding_wins_and_unknown_meta_falls_back():
    html = '<meta charset="Shift_JIS"><p>日本語</p>'
    assert decode_text(html.encode(), "text/html", "utf-8") == html
    html = '<meta charset="unknown-codec"><p>日本語</p>'
    assert decode_text(html.encode(), "text/html", None) == html
    assert decode_text(b"\xff", "text/plain", None) == "\ufffd"


def test_table_and_block_separators():
    assert readable("<table><tr><th>Name</th><th>Value</th></tr>"
                    "<tr><td>Count</td><td>42</td></tr></table>") == "Name Value\nCount 42"
    assert readable("<h4>Heading</h4><p>One<br>Two</p><div>Three</div>"
                    "<ul><li>Four</li><li>Five</li></ul>") == "Heading\nOne\nTwo\nThree\nFour\nFive"


@pytest.mark.parametrize("tag", ["template", "noscript"])
@pytest.mark.parametrize("head_end", ["", "</head>"])
def test_head_body_detection_ignores_nested_hidden_tags(tag, head_end):
    assert readable(f"<head><{tag}><p>Hidden</p><div>Also hidden</div></{tag}>"
                    f"<title>Hidden title</title>{head_end}<body><p>Visible</p>") == "Visible"


def test_utf8_bom_precedes_meta_charset():
    html = '<meta charset="shift_jis"><p>日本語</p>'
    assert decode_text(b"\xef\xbb\xbf" + html.encode("utf-8"), "text/html", None) == html
