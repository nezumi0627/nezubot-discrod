#!/bin/bash
# Install the real-browser search tier's runtime for the bot:
# - the Playwright Python package into the bot's .venv
# - a Chromium binary to drive, via Playwright's managed download
#   (lands in ~/.cache/ms-playwright). A system Chrome/Chromium
#   also works: the bot prefers Playwright's managed build when
#   present and falls back to system executables (web_browser.py).
# Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/pip install "playwright>=1.49,<2"

# Headless shell only: the bot never renders a visible window, and
# the shell build is roughly half the download and RAM of Chromium.
.venv/bin/playwright install --only-shell chromium

# Smoke check through the same executable resolution the bot uses:
# the browser must actually launch on this host. No network is
# touched, so no proxy is needed here.
.venv/bin/python - <<'PY'
import asyncio

from nezubot_discrod.web_browser import BrowserSearch


async def main():
    tier = BrowserSearch()
    browser = await tier._ensure_browser()
    if browser is None:
        raise SystemExit("install-browser: no usable Chromium found")
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("about:blank")
    print(f"install-browser: chromium launches OK (version {browser.version})")
    await tier._drop_browser()


asyncio.run(main())
PY
