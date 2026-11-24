import aiohttp

from nezubot_discrod.bot import proxy_from_env

PROXY_VARS = (
    "DISCORD_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy",
)


def clear(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_proxy_env(monkeypatch):
    clear(monkeypatch)
    assert proxy_from_env() == {}


def test_discord_proxy_takes_precedence(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://upstream.local:3128")
    monkeypatch.setenv("DISCORD_PROXY", "http://127.0.0.1:18080")
    assert proxy_from_env() == {"proxy": "http://127.0.0.1:18080"}


def test_proxy_without_credentials(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    assert proxy_from_env() == {"proxy": "http://proxy.local:3128"}


def test_proxy_credentials_split_into_auth(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("https_proxy", "http://user:p%40ss@proxy.local:3128")
    options = proxy_from_env()
    assert options["proxy"] == "http://proxy.local:3128"
    assert options["proxy_auth"] == aiohttp.BasicAuth("user", "p@ss")
