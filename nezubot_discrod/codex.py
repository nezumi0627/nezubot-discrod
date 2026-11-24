import asyncio
import base64
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from nezubot_discrod.assistant import SELF_IMPROVEMENT_TOOLS, TOOLS, request_improvement
from nezubot_discrod.member_wiki import TOOL_PARAMETERS, WIKI_TOOLS, WikiError
from nezubot_discrod.messages import collect_tool_sources, token_usage
from nezubot_discrod.prompts import (
    ARBITER_SYSTEM,
    CODEX_TOOL_INSTRUCTIONS,
    CODEX_WIKI_TOOL_INSTRUCTIONS,
    SELF_MOD_PROMPT,
    SERVER_WIKI_BUILDER_SYSTEM,
    conversation_prompt,
)
from nezubot_discrod.sandbox import run_command, truncate_output
from nezubot_discrod.server_log_tools import LOG_PARAMETERS, LOG_TOOLS
from nezubot_discrod.web_tools import WEB_PARAMETERS, WEB_TOOLS, has_web_content

CODEX_VERSION = "codex-cli 0.171.0"
PASSIVE_ITEM_TYPES = {
    "userMessage",
    "agentMessage",
    "reasoning",
    "plan",
    "functionCallOutput",
    "contextCompaction",
    "enteredReviewMode",
    "exitedReviewMode",
    "hookPrompt",
}
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "unified_exec_tty",
    "shell_snapshot",
    "shell_snapshot_v2",
    "code_mode",
    "code_mode_only",
    "code_mode_host",
    "code_mode_host_grpc",
    "code_mode_prewarm",
    "code_mode_tool_search",
    "incremental_exec_tools",
    "apps",
    "enable_mcp_apps",
    "plugins",
    "remote_plugin",
    "recommended_plugins",
    "tool_suggest",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "browser_annotation_api",
    "in_app_browser",
    "computer_use",
    "image_generation",
    "view_image",
    "multi_agent",
    "multi_agent_v2",
    "multi_agent_v2_dynamic_tools",
    "agent_message_board",
    "agent_message_board_mcp",
    "hooks",
    "skill_search",
    "skill_mcp_dependency_install",
    "goals",
    "memories",
    "deferred_executor",
    "request_permissions_tool",
    "standalone_web_search",
    "web_search_cached",
    "web_search_request",
    "sleep_tool",
    "current_time_reminder",
    "utc_environment_date",
    "default_mode_request_user_input",
)
SAFE_CONFIG = {
    "approval_policy": "never",
    "approvals_reviewer": "user",
    "sandbox_mode": "read-only",
    "web_search": "disabled",
    "mcp_servers": {},
    "notify": [],
    "allow_login_shell": False,
    "forced_login_method": "chatgpt",
    "cli_auth_credentials_store": "file",
    "check_for_update_on_startup": False,
    "project_doc_max_bytes": 0,
    "skills.bundled.enabled": False,
    "skills.include_instructions": False,
    "plugins._default.enabled": False,
    "tools.update_plan.enabled": False,
    "tools.experimental_request_user_input.enabled": False,
    "shell_environment_policy.inherit": "none",
    "history.persistence": "none",
    "features.skip_host_skill_discovery": True,
    **{f"features.{name}": False for name in DISABLED_FEATURES},
}


logger = logging.getLogger(__name__)


class CodexSecurityError(RuntimeError):
    pass


class CodexRateLimited(RuntimeError):
    pass


class CodexConnectionError(RuntimeError):
    """The app-server connection died; the turn produced no answer and
    may be retried on a fresh server."""


# Persistent-thread lifetime bounds (plan A):
# a reused thread keeps its history server-side, so cap how far it can
# grow beyond the assistant's own replayed history, and drop threads
# that have been idle long past the backend's cache retention.
THREAD_MAX_TURNS = 24
THREAD_MAX_AGE_SECONDS = 6 * 3600
THREAD_MAX_IDLE_SECONDS = 90 * 60
THREAD_REGISTRY_LIMIT = 256


def _usage_delta(value, baseline):
    """Per-turn usage from a thread's cumulative token totals."""
    if type(value) is not int or value < 0:
        return None
    base = baseline if type(baseline) is int and baseline >= 0 else 0
    return max(value - base, 0)


@dataclass
class _ThreadEntry:
    thread_id: str
    created_at: float
    last_used_at: float
    turns: int = 0
    totals: dict | None = None


def _thread_config(model, messages, *, wiki=None, server_log=None, web=None,
                   ambient=False, arbiter=False, server_wiki=False, self_improve=None,
                   run_command_enabled=True):
    """Everything that defines a Codex thread, plus its config hash.

    The persistent-thread registry keys on the hash: when the model,
    instructions or tool set change, the hash changes and the next turn
    starts a fresh thread instead of reusing one built for a different
    configuration.
    """
    tool_free = arbiter or server_wiki
    if (arbiter and server_wiki) or (
        tool_free and (wiki is not None or server_log is not None or web is not None
                       or ambient or self_improve is not None)
    ):
        raise CodexSecurityError("Tool-free Codex turn cannot receive conversation tools")
    effective_self_improve = None if ambient or tool_free else self_improve
    if arbiter:
        default_prompt = ARBITER_SYSTEM
    elif server_wiki:
        default_prompt = SERVER_WIKI_BUILDER_SYSTEM
    else:
        default_prompt = conversation_prompt(
            wiki=wiki is not None, log=server_log is not None,
            web=web is not None, ambient=ambient,
        )
    prompt = (
        messages[0]["content"]
        if messages and messages[0].get("role") == "system"
        else default_prompt
    )
    if prompt != default_prompt and not (
        wiki is None
        and server_log is None
        and web is None
        and not ambient
        and not tool_free
        and prompt == SELF_MOD_PROMPT
    ):
        raise CodexSecurityError("Only fixed system prompts are supported")
    tools = [
        *(TOOLS if run_command_enabled else []),
        *(WIKI_TOOLS if wiki is not None else []),
        *(LOG_TOOLS if server_log is not None else []),
        *(WEB_TOOLS if web is not None else []),
        *(SELF_IMPROVEMENT_TOOLS if effective_self_improve is not None else []),
    ]
    if tool_free:
        tools = []
    developer_instructions = (
        "Do not use any tools. Return only the short decision JSON."
        if arbiter else "Do not use any tools. Return only the Server Wiki JSON."
        if server_wiki else CODEX_TOOL_INSTRUCTIONS
        if wiki is None and server_log is None and web is None
        else CODEX_WIKI_TOOL_INSTRUCTIONS
    )
    dynamic_tools = [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "inputSchema": tool["function"]["parameters"],
            "deferLoading": False,
        }
        for tool in tools
    ]
    digest = hashlib.sha256(json.dumps(
        {
            "model": model,
            "baseInstructions": prompt,
            "developerInstructions": developer_instructions,
            "dynamicTools": dynamic_tools,
        },
        ensure_ascii=False, sort_keys=True,
    ).encode()).hexdigest()
    return SimpleNamespace(
        tool_free=tool_free,
        self_improve=effective_self_improve,
        prompt=prompt,
        developer_instructions=developer_instructions,
        dynamic_tools=dynamic_tools,
        config_hash=digest,
    )


def _rate_limited(error):
    if not isinstance(error, dict):
        return False
    info = error.get("codexErrorInfo")
    return (
        error.get("code") == 429
        or info in ("usageLimitExceeded", "rateLimitExceeded", "serverOverloaded")
        or (
            isinstance(info, dict)
            and any(
                isinstance(detail, dict) and detail.get("httpStatusCode") == 429
                for detail in info.values()
            )
        )
    )


def _auth_needs_refresh(path):
    now = datetime.now(UTC)
    auth = json.loads(path.read_bytes())
    try:
        token = auth["tokens"]["access_token"].split(".")[1]
        expires = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))["exp"]
        if type(expires) in (int, float):
            return not (expires > (now + timedelta(minutes=5)).timestamp())
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(auth["last_refresh"]) < now - timedelta(days=8)
    except (KeyError, TypeError, ValueError):
        return True


def app_server_arguments(binary: str) -> list[str]:
    args = [binary, "app-server", "--listen", "stdio://", "--strict-config"]
    for key, value in SAFE_CONFIG.items():
        encoded = "{}" if value == {} else json.dumps(value)
        args.extend(["-c", f"{key}={encoded}"])
    return args


class AppServer:
    def __init__(self, process, sandbox, runner=run_command, max_tool_calls=20,
                 run_command_enabled=True):
        self.process = process
        self.sandbox = sandbox
        self.runner = runner
        self.run_command_enabled = run_command_enabled
        self.sequence = 0
        self.events = []
        self.tool_calls = 0
        self.thread_id = None
        self.max_tool_calls = max_tool_calls
        self.wiki = None
        self.server_log = None
        self.web = None
        self.self_improve = None
        self.tool_free = False
        self.private_read = False
        self.web_read = False
        self.usage = None
        self.sources = []
        self.initialized = False
        self.thread_totals = None
        self._usage_baseline = {}

    async def send(self, data):
        self.process.stdin.write((json.dumps(data) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self):
        line = await self.process.stdout.readline()
        if not line:
            raise CodexConnectionError("Codex app-server closed unexpectedly")
        data = json.loads(line)
        if "method" in data and "id" in data:
            await self.handle_request(data)
            return None
        if "error" in data:
            if _rate_limited(data["error"]):
                raise CodexRateLimited("Codex rate limited")
            raise RuntimeError("Codex app-server returned an error")
        if data.get("method") in {"item/started", "item/completed"}:
            kind = data.get("params", {}).get("item", {}).get("type")
            if self.tool_free and kind == "dynamicToolCall":
                raise CodexSecurityError("Tool-free Codex turn cannot call tools")
            if kind not in PASSIVE_ITEM_TYPES | {"dynamicToolCall"}:
                raise CodexSecurityError("Unexpected built-in tool item; stopping Codex")
        return data

    async def request(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        await self.send({"id": request_id, "method": method, "params": params})
        while True:
            data = await self.receive()
            if data is None:
                continue
            if data.get("id") == request_id:
                return data["result"]
            if "method" in data:
                if len(self.events) >= 1000:
                    raise RuntimeError("Codex notification limit exceeded")
                self.events.append(data)

    async def handle_request(self, data):
        if self.tool_free:
            raise CodexSecurityError(
                "Tool-free Codex turn cannot call tools or request host actions"
            )
        method = data["method"]
        if method != "item/tool/call":
            if method in {"execCommandApproval", "applyPatchApproval"}:
                result = {"decision": "abort"}
            elif method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
            }:
                result = {"decision": "cancel"}
            else:
                await self.send(
                    {
                        "id": data["id"],
                        "error": {
                            "code": -32601,
                            "message": "Denied: only registered dynamic tools are allowed",
                        },
                    }
                )
                raise CodexSecurityError("Denied unexpected Codex request")
            await self.send({"id": data["id"], "result": result})
            raise CodexSecurityError("Denied Codex host execution or file change approval")
        params = data.get("params", {})
        args = params.get("arguments")
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise RuntimeError("Codex tool call limit exceeded")
        name = params.get("tool")
        scoped = (
            self.thread_id is not None
            and params.get("threadId") == self.thread_id
            and not params.get("namespace")
            and isinstance(name, str)
        )
        valid = False
        text = (
            "Tool error: only run_command with a nonempty command string is allowed."
            if self.wiki is None and self.server_log is None and self.web is None
            else "Tool error: unknown tool or invalid arguments."
        )
        if (
            scoped
            and self.run_command_enabled
            and name == "run_command"
            and isinstance(args, dict)
            and set(args) == {"command"}
            and isinstance(args["command"], str)
            and bool(args["command"].strip())
            and len(args["command"]) <= 8000
        ):
            text = await asyncio.to_thread(self.runner, args["command"], self.sandbox)
            valid = True
        elif scoped and self.self_improve is not None and name == "request_self_improvement":
            try:
                text = await request_improvement(self.self_improve, args)
                valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
        elif scoped and self.web is not None and name in WEB_PARAMETERS:
            try:
                if self.private_read and self.web_read:
                    text = ("Tool error: web access disabled after reading private data "
                            "and web content in the same turn")
                else:
                    text = await asyncio.to_thread(self.web.call, name, args)
                    self.web_read = self.web_read or has_web_content(text)
                    valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
        elif scoped and self.server_log is not None and name in LOG_PARAMETERS:
            self.private_read = True
            try:
                text = await asyncio.to_thread(self.server_log.call, name, args)
                valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
        elif scoped and self.wiki is not None and name in TOOL_PARAMETERS:
            self.private_read = True
            try:
                text = await asyncio.to_thread(self.wiki.call, name, args)
                valid = True
            except WikiError as error:
                text = f"Tool error: {error}"
        if valid:
            self.sources = collect_tool_sources(self.sources, name, text)
        await self.send(
            {
                "id": data["id"],
                "result": {
                    "success": valid,
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": truncate_output(text, self.sandbox.output_limit),
                        }
                    ],
                },
            }
        )

    async def initialize(self, *, refresh_token=False):
        await self.request(
            "initialize",
            {
                "clientInfo": {"name": "nezubot_discrod", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.send({"method": "initialized", "params": {}})
        account = await self.request("account/read", {"refreshToken": refresh_token})
        if (account.get("account") or {}).get("type") != "chatgpt":
            raise RuntimeError("Codex requires a ChatGPT login; run codex login first")
        self.initialized = True

    async def ensure_initialized(self, *, refresh_token=False):
        if not self.initialized:
            await self.initialize(refresh_token=refresh_token)

    async def complete(
        self, model, messages, cwd, *, wiki=None, server_log=None, web=None, ambient=False,
        arbiter=False,
        server_wiki=False,
        private_read=False,
        self_improve=None,
        thread_id=None,
        input_text=None,
        usage_baseline=None,
        scope=None,
    ):
        # `scope` is accepted (and ignored here) so adapters that forward
        # the client's completion arguments verbatim keep working; thread
        # reuse is orchestrated by CodexClient, not by this class.
        config = _thread_config(
            model, messages, wiki=wiki, server_log=server_log, web=web,
            ambient=ambient, arbiter=arbiter, server_wiki=server_wiki,
            self_improve=self_improve,
            run_command_enabled=self.run_command_enabled,
        )
        self.tool_free = config.tool_free
        self.self_improve = config.self_improve
        self.private_read = private_read
        self.web_read = False
        # Per-turn state reset: a persistent server runs many turns on
        # this one object, and nothing from a previous turn (buffered
        # notifications, tool budget, usage, sources) may leak into the
        # next one.
        self.events.clear()
        self.tool_calls = 0
        self.usage = None
        self.sources = []
        self.thread_totals = None
        self._usage_baseline = usage_baseline or {}
        self.wiki = wiki
        self.server_log = server_log
        self.web = web
        try:
            await self.ensure_initialized()
            if thread_id is None:
                thread = await self.request(
                    "thread/start",
                    {
                        "model": model,
                        "modelProvider": "openai",
                        "cwd": cwd,
                        "approvalPolicy": "never",
                        "approvalsReviewer": "user",
                        "sandbox": "read-only",
                        "ephemeral": True,
                        "environments": [],
                        "runtimeWorkspaceRoots": [],
                        "selectedCapabilityRoots": [],
                        "baseInstructions": config.prompt,
                        "developerInstructions": config.developer_instructions,
                        "dynamicTools": config.dynamic_tools,
                    },
                )
                if (
                    thread.get("approvalPolicy") != "never"
                    or thread.get("sandbox", {}).get("type") != "readOnly"
                    or thread.get("sandbox", {}).get("networkAccess", False)
                ):
                    raise CodexSecurityError("Codex did not apply the requested security policy")
                self.thread_id = thread["thread"]["id"]
            else:
                # Reused thread: its policy was verified when it was
                # created under the same config hash.
                self.thread_id = thread_id
            await self.request(
                "turn/start",
                {
                    "threadId": self.thread_id,
                    "input": [{
                        "type": "text",
                        "text": (
                            input_text if input_text is not None
                            else json.dumps(messages, ensure_ascii=False)
                        ),
                    }],
                    **({"effort": "low"} if self.tool_free else {}),
                },
            )
            answer = ""
            while True:
                data = self.events.pop(0) if self.events else await self.receive()
                if data is None:
                    continue
                params = data.get("params", {})
                if (data.get("method") == "thread/tokenUsage/updated"
                        and params.get("threadId") == self.thread_id):
                    total = params.get("tokenUsage", {}).get("total", {})
                    self.thread_totals = total
                    base = self._usage_baseline
                    counts = token_usage({
                        "prompt_tokens": _usage_delta(
                            total.get("inputTokens"), base.get("inputTokens")),
                        "completion_tokens": _usage_delta(
                            total.get("outputTokens"), base.get("outputTokens")),
                    })
                    if counts is not None:
                        self.usage = {
                            "prompt_tokens": counts[0],
                            "completion_tokens": counts[1],
                        }
                        cached = _usage_delta(
                            total.get("cachedInputTokens"), base.get("cachedInputTokens"))
                        if cached is not None:
                            # Prompt-cache observability: how much of this
                            # turn's input the backend served from its
                            # prefix cache (delta of the thread totals).
                            self.usage["cached_input_tokens"] = cached
                if data.get("method") == "item/completed":
                    item = params["item"]
                    if item.get("type") == "agentMessage":
                        text = item.get("text", "")
                        answer = text if server_wiki else truncate_output(text, 8000)
                if data.get("method") == "turn/completed":
                    turn = params["turn"]
                    if _rate_limited(turn.get("error")):
                        raise CodexRateLimited("Codex rate limited")
                    if turn.get("status") != "completed":
                        raise RuntimeError("Codex turn failed or was interrupted")
                    return answer
        finally:
            # Never leave the previous turn's tool bindings live on a
            # persistent server between turns.
            self.wiki = None
            self.server_log = None
            self.web = None
            self.self_improve = None


class CodexClient:
    def __init__(self, settings, *, runner=run_command, max_tool_calls=20,
                 run_command_enabled=True):
        self.settings = settings
        self.runner = runner
        self.run_command_enabled = run_command_enabled
        self.chat = SimpleNamespace(completions=self)
        self.lock = asyncio.Lock()
        self.runtime = None
        self.max_tool_calls = max_tool_calls
        self._version_key = None
        self._catalog_key = None
        self._catalog_models = None
        # Plan A: one persistent app-server plus a registry of reusable
        # per-scope threads. Turns stay
        # serialized on self.lock, exactly as the per-call path always
        # was, so a single connection never interleaves two turns.
        self._persistent = None
        self._persistent_ctx = None
        self._persistent_work = None
        self._persistent_failures = 0
        self._threads = OrderedDict()

    async def close(self):
        async with self.lock:
            await self._drop_persistent_server("client closing", warn=False)
            if self.runtime is not None:
                self.runtime.cleanup()
                self.runtime = None

    def forget_threads(self, scopes):
        """Drop persistent threads for conversation scopes (e.g. after
        the assistant's local history for them was reset)."""
        wanted = set(scopes)
        for key in [key for key in self._threads if key[0] in wanted]:
            del self._threads[key]

    async def arbitrate(self, *, model, messages):
        async with asyncio.timeout(12):
            return await self.create(model=model, messages=messages, arbiter=True)

    def prepare_runtime(self, *, auth_file=None):
        if self.runtime is None:
            self.runtime = tempfile.TemporaryDirectory(prefix=".discord-codex-", dir=Path.home())
            root = Path(self.runtime.name)
            home = root / "home"
            home.mkdir(mode=0o700)
            auth = auth_file or Path(self.settings.codex_home).expanduser() / "auth.json"
            try:
                shutil.copyfile(auth, home / "auth.json")
                (home / "auth.json").chmod(0o600)
            except OSError:
                self.runtime.cleanup()
                self.runtime = None
                raise RuntimeError("Codex auth.json unavailable; run codex login first") from None
        return Path(self.runtime.name)

    @staticmethod
    def _server_env(root):
        env = {key: os.environ[key] for key in (
            "PATH", "LANG", "SSL_CERT_FILE",
            # The Codex CLI itself needs the host's egress proxy, if any;
            # without it a proxied host cannot reach OpenAI at all.
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        ) if key in os.environ}
        env.update(HOME=str(root), CODEX_HOME=str(root / "home"))
        return env

    @contextmanager
    def _workspace(self, root):
        with tempfile.TemporaryDirectory(prefix="work-", dir=root) as work:
            yield Path(work), self._server_env(root)

    def _binary_key(self):
        try:
            stat_result = os.stat(self.settings.codex_binary)
            return (self.settings.codex_binary, stat_result.st_mtime_ns, stat_result.st_size)
        except OSError:
            return None

    async def _verify_version(self, env):
        # Cache per client lifetime, keyed by the binary's identity
        # (path + mtime + size): an upgraded/replaced binary is re-checked.
        key = self._binary_key()
        if key is not None and key == self._version_key:
            return
        version = await asyncio.create_subprocess_exec(
            self.settings.codex_binary,
            "--version",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(version.communicate(), timeout=10)
            if version.returncode or stdout.decode().strip() != CODEX_VERSION:
                raise RuntimeError(f"Codex provider requires {CODEX_VERSION}")
        finally:
            if version.returncode is None:
                version.kill()
                await version.wait()
        self._version_key = key

    async def rate_limits(self):
        """Current ChatGPT subscription usage windows, via the app-server.

        Returns the raw GetAccountRateLimitsResponse of
        account/rateLimits/read; parsing lives in
        nezubot_discrod.usage_limits. Auth material never leaves the
        runtime home and is never logged.

        When the persistent server is up it serves the request, so the
        common case has exactly one process touching the auth file;
        otherwise a short-lived server is spawned as before.
        """
        async with self.lock:
            if self._persistent is not None and self._persistent.process.returncode is None:
                self._persistent.tool_free = True
                try:
                    return await self._persistent.request("account/rateLimits/read", {})
                except (CodexConnectionError, OSError):
                    await self._drop_persistent_server("connection lost during rate_limits")
            root = self.prepare_runtime()
            auth = root / "home" / "auth.json"
            with self._workspace(root) as (cwd, env):
                await self._verify_version(env)
                async with self.server(cwd, env) as server:
                    server.tool_free = True
                    await server.initialize(refresh_token=_auth_needs_refresh(auth))
                    return await server.request("account/rateLimits/read", {})

    async def warm_server_wiki_auth(self):
        async with self.lock:
            root = self.prepare_runtime()
            auth = root / "home" / "auth.json"
            if _auth_needs_refresh(auth):
                with self._workspace(root) as (cwd, env):
                    await self._verify_version(env)
                    async with self.server(cwd, env) as server:
                        server.tool_free = True
                        await server.initialize(refresh_token=True)
                if _auth_needs_refresh(auth):
                    raise RuntimeError("Codex auth refresh did not produce fresh credentials")
            return auth

    @asynccontextmanager
    async def server(self, cwd, env, catalog=None):
        arguments = app_server_arguments(self.settings.codex_binary)
        if catalog is not None:
            arguments.extend(["-c", f"model_catalog_json={json.dumps(str(catalog))}"])
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=1024 * 1024,
        )
        try:
            yield AppServer(
                process, self.settings.sandbox, self.runner, self.max_tool_calls,
                run_command_enabled=self.run_command_enabled,
            )
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()

    async def _drop_persistent_server(self, reason, *, warn=True):
        ctx, self._persistent_ctx = self._persistent_ctx, None
        self._persistent = None
        self._persistent_work = None
        self._threads.clear()
        if ctx is None:
            return
        if warn:
            logger.warning("Dropping persistent Codex app-server: %s", reason)
        else:
            logger.info("Stopping persistent Codex app-server: %s", reason)
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.exception("Persistent Codex app-server did not stop cleanly")

    async def _ensure_persistent_server(self, model):
        """The one long-lived app-server, started lazily and restarted
        after death. Threads do not survive it: the registry is cleared
        whenever a fresh process comes up, so the next turn for each
        scope re-seeds a new thread from the assistant's local history.
        """
        if self._persistent is not None:
            if self._persistent.process.returncode is None:
                return self._persistent
            logger.warning(
                "Persistent Codex app-server exited (code %s); restarting",
                self._persistent.process.returncode,
            )
            await self._drop_persistent_server("process exited")
        root = self.prepare_runtime()
        work = root / "work-persistent"
        work.mkdir(exist_ok=True)
        env = self._server_env(root)
        await self._verify_version(env)
        catalog = await self.direct_catalog(root, work, env, model)
        ctx = self.server(work, env, catalog)
        server = await ctx.__aenter__()
        try:
            auth = root / "home" / "auth.json"
            try:
                refresh = _auth_needs_refresh(auth) if auth.exists() else False
            except Exception:
                refresh = False
            await server.initialize(refresh_token=refresh)
        except BaseException:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
            raise
        self._persistent = server
        self._persistent_ctx = ctx
        self._persistent_work = work
        self._persistent_failures = 0
        logger.info("Persistent Codex app-server started")
        return server

    def _thread_entry(self, key, now):
        entry = self._threads.get(key)
        if entry is None:
            return None
        if (
            entry.turns >= THREAD_MAX_TURNS
            or now - entry.created_at >= THREAD_MAX_AGE_SECONDS
            or now - entry.last_used_at >= THREAD_MAX_IDLE_SECONDS
        ):
            del self._threads[key]
            return None
        return entry

    @staticmethod
    def _response(text, server):
        return SimpleNamespace(
            usage=server.usage,
            sources=tuple(server.sources),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=text,
                        tool_calls=None,
                    )
                )
            ],
        )

    async def _create_persistent(
        self, *, model, messages, scope, wiki=None, server_log=None, web=None,
        ambient=False, private_read=False, self_improve=None,
    ):
        config = _thread_config(
            model, messages, wiki=wiki, server_log=server_log, web=web,
            ambient=ambient, self_improve=self_improve,
            run_command_enabled=self.run_command_enabled,
        )
        server = await self._ensure_persistent_server(model)
        now = time.monotonic()
        key = (scope, config.config_hash)
        entry = self._thread_entry(key, now)
        if entry is not None:
            # The thread already holds the conversation; send only the
            # newest user message, in the same JSON envelope shape the
            # first turn used, so the request stays a strict extension
            # of the cached prefix.
            input_text = json.dumps([messages[-1]], ensure_ascii=False)
            thread_id = entry.thread_id
            baseline = entry.totals
        else:
            input_text = None
            thread_id = None
            baseline = None
        text = await server.complete(
            model,
            messages,
            str(self._persistent_work),
            wiki=wiki,
            **({"server_log": server_log} if server_log is not None else {}),
            **({"web": web} if web is not None else {}),
            **({"ambient": True} if ambient else {}),
            **({"private_read": True} if private_read else {}),
            **({"self_improve": self_improve} if self_improve is not None else {}),
            thread_id=thread_id,
            input_text=input_text,
            usage_baseline=baseline,
        )
        now = time.monotonic()
        if entry is None:
            entry = _ThreadEntry(
                thread_id=server.thread_id, created_at=now, last_used_at=now,
            )
            self._threads[key] = entry
            while len(self._threads) > THREAD_REGISTRY_LIMIT:
                self._threads.popitem(last=False)
        entry.turns += 1
        entry.last_used_at = now
        entry.totals = server.thread_totals
        self._threads.move_to_end(key)
        self._persistent_failures = 0
        return self._response(text, server)

    async def direct_catalog(self, root, cwd, env, model):
        catalog = root / "direct-models.json"
        if not catalog.exists():
            async with self.server(cwd, env) as server:
                await server.initialize()
                await server.request("model/list", {"includeHidden": True})
                cache = json.loads((root / "home" / "models_cache.json").read_text())
            models = cache.get("models")
            if not isinstance(models, list) or not models or not all(
                isinstance(entry, dict) and isinstance(entry.get("slug"), str)
                for entry in models
            ):
                raise CodexSecurityError("Codex model metadata unavailable")
            for entry in models:
                entry["tool_mode"] = "direct"
            catalog.write_text(json.dumps({"models": models}))
        # Cache the parsed catalog per client lifetime, keyed by the file's
        # identity; a rewritten catalog (new runtime) is re-parsed.
        try:
            stat_result = catalog.stat()
            key = (str(catalog), stat_result.st_mtime_ns, stat_result.st_size)
        except OSError:
            key = None
        if key is None or key != self._catalog_key:
            metadata = json.loads(catalog.read_text())
            self._catalog_models = metadata["models"]
            self._catalog_key = key
        if not any(entry["slug"] == model for entry in self._catalog_models):
            raise RuntimeError("Requested model is missing from the Codex catalog")
        return catalog

    @staticmethod
    def _persistent_eligible(scope, messages, *, arbiter, server_wiki):
        return (
            scope is not None
            and not arbiter
            and not server_wiki
            and isinstance(messages, list)
            and len(messages) >= 1
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "user"
        )

    async def create(
        self, *, model, messages, wiki=None, server_log=None, web=None, ambient=False,
        arbiter=False, server_wiki=False, private_read=False, self_improve=None, scope=None,
        **kwargs
    ):
        async with self.lock, asyncio.timeout(600):
            if self._persistent_eligible(scope, messages, arbiter=arbiter, server_wiki=server_wiki):
                try:
                    return await self._create_persistent(
                        model=model,
                        messages=messages,
                        scope=scope,
                        wiki=wiki,
                        server_log=server_log,
                        web=web,
                        ambient=ambient,
                        private_read=private_read,
                        self_improve=self_improve,
                    )
                except (CodexSecurityError, CodexRateLimited):
                    raise
                except (CodexConnectionError, OSError) as error:
                    # Startup failure or a server that died mid-turn:
                    # answer this turn on the per-call path instead of
                    # failing the reply.
                    logger.warning(
                        "Persistent Codex path failed (%s); "
                        "falling back to a per-call server",
                        type(error).__name__,
                    )
                    await self._drop_persistent_server("connection failure")
                except Exception as error:
                    # A turn-level failure leaves the server in place,
                    # but repeated failures (or a timeout, which can
                    # leave a turn running server-side) force a fresh
                    # process — with a fresh auth read — on a later turn.
                    self._persistent_failures += 1
                    if self._persistent_failures >= 2 or isinstance(error, TimeoutError):
                        await self._drop_persistent_server("repeated turn failures")
                    raise
            return await self._create_legacy(
                model=model,
                messages=messages,
                wiki=wiki,
                server_log=server_log,
                web=web,
                ambient=ambient,
                arbiter=arbiter,
                server_wiki=server_wiki,
                private_read=private_read,
                self_improve=self_improve,
            )

    async def _create_legacy(
        self, *, model, messages, wiki=None, server_log=None, web=None, ambient=False,
        arbiter=False, server_wiki=False, private_read=False, self_improve=None,
    ):
        # Never load the user's config, MCP servers, plugins, hooks or skills.
        root = self.prepare_runtime()
        with self._workspace(root) as (cwd, env):
            await self._verify_version(env)
            catalog = await self.direct_catalog(root, cwd, env, model)
            async with self.server(cwd, env, catalog) as server:
                text = await server.complete(
                    model,
                    messages,
                    str(cwd),
                    wiki=wiki,
                    **({"server_log": server_log} if server_log is not None else {}),
                    **({"web": web} if web is not None else {}),
                    **({"ambient": True} if ambient else {}),
                    **({"arbiter": True} if arbiter else {}),
                    **({"server_wiki": True} if server_wiki else {}),
                    **({"private_read": True} if private_read else {}),
                    **({"self_improve": self_improve} if self_improve is not None else {}),
                )
        return self._response(text, server)
