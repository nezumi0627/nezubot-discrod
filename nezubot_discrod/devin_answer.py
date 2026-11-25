"""One-shot Devin (SWE-2.1) answer backend, via the local bridge.

A member who picks ``swe-2-1`` in /model gets their answer turns
answered by Devin instead of Codex. Devin runs through the same
local bridge the Server Wiki curator uses (~/.nezubot-discrod/devin-bridge,
127.0.0.1:8799): POST /sessions starts a ``devin -p`` job and
GET /sessions/<id> polls it to completion.

Devin print mode has no tools, so the bot assembles everything the
turn would have given Codex — the system prompt, the conversation
history and the final payload with its pre-retrieved context — into
a single prompt string (see build_devin_prompt) and the model's
stdout becomes the answer text.

The bridge bearer token is read from the bridge's own env file
(bridge.env, mode 600) or the BRIDGE_TOKEN environment variable,
exactly like the curator's client. The value is never logged.

Concurrency: the bridge runs at most 2 jobs at once, so Devin
answer turns are gated by a semaphore of the same size; extra turns
wait within their own time budget. The whole call is bounded by a
deadline comfortably inside the 600s turn budget, and every failure
mode raises DevinAnswerError with a ``kind`` the caller maps to a
gentle user-facing message — a Devin turn must never kill the reply.
"""

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from nezubot_discrod.model_overrides import DEVIN_MODEL

logger = logging.getLogger(__name__)

# A Devin answer turn gets at most this long end to end (queueing
# for a bridge slot included), staying inside the 600s turn budget
# the Codex path enforces.
DEFAULT_TIMEOUT = 480.0
DEFAULT_POLL_INTERVAL = 3.0
MAX_CONCURRENT = 2

# Appended to the assembled prompt: the system prompt describes the
# tool-using Codex setup, so the one-shot backend is told plainly
# that tools do not exist here and only the answer text may come back.
DEVIN_ANSWER_CONSTRAINT = (
    "【実行制約】あなたはツールを持たない one-shot の回答バックエンドとして"
    "動作しています。サーバーログ検索・Web検索などのツールは使えません。"
    "system 指示のうちツール利用に関する部分は無視し、与えられた文脈"
    "（会話履歴・取得済みの資料・現在の日時など）だけを使って、最後の"
    "ユーザー発言への回答本文だけを返してください。前置き・説明・JSON"
    "ではなく、Discord にそのまま投稿できる回答文だけを出力してください。"
    "文脈に無い事実は推測で断定しないでください。"
)


class DevinAnswerError(RuntimeError):
    """A Devin answer turn failed; ``kind`` says how, for the reply."""

    def __init__(self, message, *, kind="failed"):
        super().__init__(message)
        self.kind = kind


def devin_failure_message(kind, default_model):
    """The gentle answer text a failed Devin turn resolves to."""
    if kind == "timeout":
        return (
            "swe-2-1 での回答生成がタイムアウトしちゃった。"
            "もう一度試すか、/model でデフォルト"
            f"（{default_model}）に戻してね。"
        )
    if kind == "busy":
        return (
            "swe-2-1 が混み合っていて回答できなかった。"
            "少し時間をおいて再試行してね。"
        )
    return (
        "swe-2-1 で回答を生成できなかったよ。"
        "時間をおいて再試行するか、/model でデフォルト"
        f"（{default_model}）に戻してね。"
    )


def build_devin_prompt(messages):
    """Flatten a chat-completions message list into one prompt.

    The system text goes first, then the operating constraint, then
    the replayed history (whose user entries are the exact payload
    strings earlier turns sent, pre-retrieved context included),
    and finally the current turn's payload, marked as the message
    to answer.
    """
    system = ""
    turns = list(messages)
    if turns and turns[0].get("role") == "system":
        system = turns[0].get("content") or ""
        turns = turns[1:]
    parts = [system, DEVIN_ANSWER_CONSTRAINT]
    if len(turns) > 1:
        parts.append("【これまでの会話】")
        for message in turns[:-1]:
            speaker = (
                "ユーザー" if message.get("role") == "user" else "nezubot-discrod"
            )
            parts.append(f"{speaker}: {message.get('content') or ''}")
    parts.append("【今回のユーザー発言】")
    parts.append(turns[-1].get("content") or "" if turns else "")
    return "\n\n".join(part for part in parts if part)


def _bridge_token(env_path):
    """The bridge bearer token: BRIDGE_TOKEN env, else bridge.env.

    The value is returned to the caller for the Authorization
    header only; it is never logged or stored anywhere else.
    """
    token = os.getenv("BRIDGE_TOKEN", "").strip()
    if token:
        return token
    try:
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("BRIDGE_TOKEN="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


class DevinAnswerer:
    """Async front end for one-shot Devin answer turns."""

    def __init__(
        self,
        *,
        bridge_url=None,
        env_path=None,
        sessions_dir=None,
        timeout=None,
        poll_interval=DEFAULT_POLL_INTERVAL,
        max_concurrent=MAX_CONCURRENT,
    ):
        self.bridge_url = (
            bridge_url or os.getenv("DEVIN_BRIDGE_URL") or "http://127.0.0.1:8799"
        ).rstrip("/")
        self.env_path = Path(
            env_path
            or os.getenv("DEVIN_BRIDGE_ENV", "~/.nezubot-discrod/devin-bridge.env")
        ).expanduser()
        self.sessions_dir = Path(
            sessions_dir
            or os.getenv(
                "DEVIN_BRIDGE_SESSIONS_DIR", "~/.nezubot-discrod/devin-bridge-sessions"
            )
        ).expanduser()
        if timeout is None:
            try:
                timeout = float(os.getenv("DEVIN_ANSWER_TIMEOUT", ""))
            except ValueError:
                timeout = DEFAULT_TIMEOUT
        self.timeout = timeout if timeout and timeout > 0 else DEFAULT_TIMEOUT
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self._slots = None
        self._slots_loop = None
        # Loopback only: the bridge is on this host, and the ambient
        # egress proxy must not see (or be needed for) these calls.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _semaphore(self):
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots = asyncio.Semaphore(self.max_concurrent)
            self._slots_loop = loop
        return self._slots

    async def answer(self, prompt, *, model=DEVIN_MODEL):
        """Run one Devin job for ``prompt`` and return its answer text."""
        if model.endswith("-lightning") or "-lightning-" in model:
            # Paid per-token Devin variants are never an answer model.
            raise ValueError(f"Paid Devin model is not allowed: {model}")
        try:
            async with asyncio.timeout(self.timeout):
                async with self._semaphore():
                    return await asyncio.to_thread(
                        self._run_sync, prompt, model, self.timeout
                    )
        except TimeoutError:
            raise DevinAnswerError(
                f"devin answer timed out after {self.timeout}s", kind="timeout"
            ) from None

    # -- synchronous bridge client (runs in a worker thread) ---------

    def _request(self, method, path, token, body=None):
        request = urllib.request.Request(
            self.bridge_url + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with self._opener.open(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode())

    def _run_sync(self, prompt, model, timeout):
        deadline = time.monotonic() + timeout
        token = _bridge_token(self.env_path)
        if not token:
            raise DevinAnswerError(
                "BRIDGE_TOKEN not available (bridge.env unreadable)",
                kind="unavailable",
            )
        job_id = self._create_job(prompt, model, token, deadline)
        return self._poll_job(job_id, token, deadline, timeout)

    def _create_job(self, prompt, model, token, deadline):
        # The bridge caps concurrent jobs at 2 and answers 429 while
        # full (the curator may be holding the slots); queue here
        # within this turn's own deadline instead of failing.
        while True:
            try:
                _, created = self._request(
                    "POST", "/sessions", token,
                    body={"prompt": prompt, "model": model},
                )
                return created["id"]
            except urllib.error.HTTPError as error:
                if error.code != 429:
                    raise DevinAnswerError(
                        f"bridge POST failed: HTTP {error.code}", kind="failed"
                    ) from error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DevinAnswerError(
                        "bridge stayed busy (429) until the deadline",
                        kind="busy",
                    ) from error
                time.sleep(min(10.0, remaining))
            except (urllib.error.URLError, OSError) as error:
                raise DevinAnswerError(
                    f"bridge unreachable: {error}", kind="unavailable"
                ) from error
            except (KeyError, TypeError, ValueError) as error:
                raise DevinAnswerError(
                    f"bridge returned an unusable response: {error}",
                    kind="failed",
                ) from error

    def _poll_job(self, job_id, token, deadline, timeout):
        while True:
            try:
                _, job = self._request("GET", f"/sessions/{job_id}", token)
            except (urllib.error.URLError, OSError) as error:
                raise DevinAnswerError(
                    f"bridge unreachable while polling: {error}",
                    kind="unavailable",
                ) from error
            status = job.get("status")
            if status == "done":
                text = self._job_output(job_id, job)
                if not text:
                    raise DevinAnswerError(
                        "devin job finished with empty output", kind="empty"
                    )
                return text
            if status == "failed":
                raise DevinAnswerError(
                    f"devin job {job_id} failed:"
                    f" {job.get('error') or job.get('exit_code')}",
                    kind="failed",
                )
            if time.monotonic() > deadline:
                raise DevinAnswerError(
                    f"devin job {job_id} timed out after {timeout}s",
                    kind="timeout",
                )
            time.sleep(self.poll_interval)

    def _job_output(self, job_id, job):
        output = Path(self.sessions_dir, job_id, "output.log")
        try:
            text = output.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = job.get("output") or ""
        return text.strip()
