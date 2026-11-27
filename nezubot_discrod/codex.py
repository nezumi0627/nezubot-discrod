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
from dataclasses import dataclass, field
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
from nezubot_discrod.turn_context import current_turn_id
from nezubot_discrod.weather_tools import WEATHER_PARAMETERS, WEATHER_TOOLS
from nezubot_discrod.web_guard import OutboundGuard
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
# that have been idle long past the backend's cache retention. The
# turn cap was halved from 24 to 12 for v1.20.0's input budget: a
# 24-turn thread re-reads two dozen full evidence payloads on every
# turn, which is where the measured 106k-token tail came from; the
# assistant's compacted replay covers the older exchange when a
# thread rolls over.
THREAD_MAX_TURNS = 12
THREAD_MAX_AGE_SECONDS = 6 * 3600
THREAD_MAX_IDLE_SECONDS = 90 * 60
THREAD_REGISTRY_LIMIT = 256

# Input budget for a reused thread (v1.23.0). A continuing thread
# re-reads its whole context on every turn, so the previous turn's
# input is the size the next turn starts from; when the estimate
# (previous input + previous output + the new payload) crosses this
# budget, the thread is retired before the turn and the turn
# re-seeds a fresh thread from the assistant's compacted local
# history — the same rollover the turn/age/idle caps already
# perform, so no conversational state is managed twice.
#
# The value is set from the 2026-11-26 measurement (per-turn input
# p50 14,538 / p95 70,280 / max 94,807 tokens): it sits far above
# the median on purpose. A hard cap near 12k — the figure floated
# in the GPT architecture review — would retire most threads and
# throw away the 96–98% prompt-cache hit rate that thread reuse
# exists for. The 40,000 budget instead cuts only
# the accumulated tail (roughly the top decile of turns) while
# ordinary threads, whose contexts run at the ~14.5k median, are
# never touched. Override with CODEX_THREAD_INPUT_BUDGET_TOKENS;
# 0 disables the budget gate (the turn/age/idle caps still apply).
THREAD_INPUT_BUDGET_TOKENS = 40_000


def _thread_input_budget_from_env():
    raw = os.getenv("CODEX_THREAD_INPUT_BUDGET_TOKENS", "")
    try:
        value = int(raw)
    except ValueError:
        return THREAD_INPUT_BUDGET_TOKENS
    return value if value >= 0 else THREAD_INPUT_BUDGET_TOKENS


# Concurrent turns on the one persistent app-server (v1.25.0). The
# 2026-11-26 probe proved the server
# executes turns on different threads truly in parallel — two turns
# at once took 53.5s against 96.7s sequentially — so the client now
# admits up to this many turns at once instead of serializing every
# turn on one lock. Turns that would share a thread still serialize
# (the server rejects a second turn on a busy thread), and auth
# operations keep their own exclusive lock. The default stays at 2:
# the measured win is "a second simultaneous speaker no longer waits
# for the first", and how the subscription rate limit answers higher
# concurrency is still unmeasured. Override with
# CODEX_MAX_CONCURRENT_TURNS.
MAX_CONCURRENT_TURNS = 2


def _max_concurrent_turns_from_env():
    raw = os.getenv("CODEX_MAX_CONCURRENT_TURNS", "")
    try:
        value = int(raw)
    except ValueError:
        return MAX_CONCURRENT_TURNS
    return value if value >= 1 else MAX_CONCURRENT_TURNS


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
    # Per-turn usage of the most recent turn on this thread (deltas
    # of `totals`): the input figure is the context size the next
    # turn would start from, which is what the input budget gates on.
    last_input_tokens: int | None = None
    last_output_tokens: int | None = None


def _thread_config(model, messages, *, wiki=None, server_log=None, web=None,
                   weather=None,
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
                       or weather is not None
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
            web=web is not None, weather=weather is not None, ambient=ambient,
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
        and weather is None
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
        *(WEATHER_TOOLS if weather is not None else []),
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


# Per-turn notification queue bound: a turn that lets more than this
# many routed messages pile up undrained fails with the limit error
# the pre-demux buffered-events path used to raise.
_TURN_QUEUE_LIMIT = 1000


@dataclass
class _TurnState:
    """One in-flight turn's private state on a shared AppServer.

    Until v1.23.0 the AppServer object itself held exactly one turn's
    state (tool bindings, tool budget, outbound guard, usage), which
    is what forced every turn through a single client-wide lock even
    though the server executes threads in parallel. The state now
    lives here, one instance per active turn, keyed by thread in
    AppServer._turns. The bindings and the web_guard in particular
    are a security boundary: a tool call is only ever handled with
    the bindings of the turn its threadId names.
    """

    queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=_TURN_QUEUE_LIMIT)
    )
    thread_id: str | None = None
    tool_calls: int = 0
    max_tool_calls: int = 20
    wiki: object = None
    server_log: object = None
    web: object = None
    weather: object = None
    self_improve: object = None
    tool_free: bool = False
    private_read: bool = False
    web_read: bool = False
    web_guard: OutboundGuard = field(default_factory=OutboundGuard)
    usage: dict | None = None
    sources: list = field(default_factory=list)
    thread_totals: dict | None = None
    usage_baseline: dict = field(default_factory=dict)
    error: BaseException | None = None


class AppServer:
    def __init__(self, process, sandbox, runner=run_command, max_tool_calls=20,
                 run_command_enabled=True):
        self.process = process
        self.sandbox = sandbox
        self.runner = runner
        self.run_command_enabled = run_command_enabled
        self.sequence = 0
        self.max_tool_calls = max_tool_calls
        self.initialized = False
        # Compatibility mirrors of the most recently completed turn.
        # Concurrent turns keep their real state in _TurnState; these
        # attributes serve the per-call path and diagnostics and must
        # never be read to serve another turn.
        self.thread_id = None
        self.usage = None
        self.sources = []
        self.thread_totals = None
        self.wiki = None
        self.server_log = None
        self.web = None
        self.weather = None
        self.self_improve = None
        self.tool_free = False
        self.private_read = False
        self.web_read = False
        self.web_guard = OutboundGuard()
        # Demux plumbing: a single reader task owns stdout. Responses
        # are matched to their callers by JSON-RPC id; notifications
        # and server requests are routed to the owning turn's queue
        # by threadId (see _route).
        self._pending = {}
        self._turns = {}
        self._reader_task = None
        self._closed_error = None
        self._init_lock = asyncio.Lock()
        # The reader only reads while somebody can consume what it
        # reads — a pending request, or a registered turn. Otherwise
        # it would race ahead of the conversation it serves and drop
        # traffic whose owner has not registered yet (a real server
        # never sends such traffic, but a scripted stream can, and
        # pausing costs nothing when the connection is idle anyway).
        self._interest = asyncio.Event()

    async def send(self, data):
        self.process.stdin.write((json.dumps(data) + "\n").encode())
        await self.process.stdin.drain()

    def _ensure_reader(self):
        if self._reader_task is None:
            self._reader_task = asyncio.ensure_future(self._read_loop())

    async def _stop_reader(self):
        task, self._reader_task = self._reader_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    async def _read_loop(self):
        error = None
        try:
            while True:
                if not self._pending and not self._turns:
                    self._interest.clear()
                    if not self._pending and not self._turns:
                        await self._interest.wait()
                line = await self.process.stdout.readline()
                if not line:
                    raise CodexConnectionError(
                        "Codex app-server closed unexpectedly")
                self._dispatch(json.loads(line))
                # Yield after every line: a caller whose request this
                # line answers (or whose turn it belongs to) must get
                # to run — registering its next request, or its turn
                # state — before the following line is dispatched. A
                # real pipe enforces this by making readline wait for
                # data; an in-memory stream would otherwise let the
                # reader race the whole conversation ahead of the
                # turns it serves.
                await asyncio.sleep(0)
        except BaseException as exc:  # EOF, malformed JSON, cancellation
            error = exc
        finally:
            if error is None or isinstance(error, asyncio.CancelledError):
                error = CodexConnectionError(
                    "Codex app-server closed unexpectedly")
            self._closed_error = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()
            # The sentinel is queued behind whatever the server had
            # already sent, so a turn drains its real traffic first
            # and only then reports the disconnect.
            for state in self._turns.values():
                try:
                    state.queue.put_nowait(("closed", error))
                except asyncio.QueueFull:
                    if state.error is None:
                        state.error = error

    def _route(self, data):
        """The turn state a server message belongs to, or None.

        Exact routing is by params.threadId. A message carrying no
        threadId at all falls back to the sole active turn when there
        is exactly one — the real server always tags its traffic, so
        in production the fallback never reroutes anything; it keeps
        single-turn drivers (and the scripted test fakes, whose
        notifications predate the tagging) working unchanged. A
        message naming an unknown thread is never rerouted.
        """
        params = data.get("params") or {}
        thread_id = params.get("threadId")
        if thread_id is not None:
            return self._turns.get(thread_id)
        if len(self._turns) == 1:
            return next(iter(self._turns.values()))
        return None

    def _enqueue(self, state, item):
        try:
            state.queue.put_nowait(item)
        except asyncio.QueueFull:
            if state.error is None:
                state.error = RuntimeError("Codex notification limit exceeded")

    def _dispatch(self, data):
        if "method" in data and "id" in data:
            # A server-initiated request (a dynamic tool call, or a
            # host-action approval this client always refuses).
            state = self._route(data)
            if state is not None:
                self._enqueue(state, ("request", data))
            elif data["method"] == "item/tool/call":
                asyncio.ensure_future(self._refuse_tool_call(data))
            else:
                asyncio.ensure_future(self._deny_unowned_request(data))
            return
        if "id" in data:
            future = self._pending.pop(data["id"], None)
            if future is not None and not future.done():
                future.set_result(data)
            return
        if "method" in data:
            state = self._route(data)
            if state is not None:
                self._enqueue(state, ("notification", data))

    async def _deny_unowned_request(self, data):
        """Answer a server request no live turn owns, without touching
        any turn's state: approvals are refused with their decision,
        everything else gets the method-not-allowed error. Unlike the
        owned path this never raises — there is no turn to fail."""
        method = data["method"]
        logger.warning("Denying Codex request with no owning turn: %s", method)
        try:
            if method in {"execCommandApproval", "applyPatchApproval"}:
                await self.send({"id": data["id"], "result": {"decision": "abort"}})
            elif method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
            }:
                await self.send({"id": data["id"], "result": {"decision": "cancel"}})
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
        except (OSError, RuntimeError):
            pass

    async def _refuse_tool_call(self, data):
        """A dynamic tool call naming a thread with no live turn:
        answer it as a failed tool result — never executed, and no
        turn's bindings or guard are touched — exactly as the owning
        turn's own scope check answers an out-of-scope call."""
        logger.warning(
            "Refusing Codex tool call for unknown thread: %s",
            (data.get("params") or {}).get("tool"),
        )
        try:
            await self.send(
                {
                    "id": data["id"],
                    "result": {
                        "success": False,
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": "Tool error: unknown tool or invalid arguments.",
                            }
                        ],
                    },
                }
            )
        except (OSError, RuntimeError):
            pass

    async def request(self, method, params):
        self._ensure_reader()
        if self._closed_error is not None:
            raise self._closed_error
        self.sequence += 1
        request_id = self.sequence
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._interest.set()
        try:
            await self.send({"id": request_id, "method": method, "params": params})
            data = await future
        finally:
            self._pending.pop(request_id, None)
        if "error" in data:
            if _rate_limited(data["error"]):
                raise CodexRateLimited("Codex rate limited")
            raise RuntimeError("Codex app-server returned an error")
        return data["result"]

    async def _handle_request(self, state, data):
        if state.tool_free:
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
        state.tool_calls += 1
        if state.tool_calls > state.max_tool_calls:
            # Budget exhausted: answer the call with an error result
            # instead of failing the whole turn, so the model can wrap
            # up with what it has already gathered.
            await self.send(
                {
                    "id": data["id"],
                    "result": {
                        "success": False,
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": (
                                    "Tool error: the tool-call budget for "
                                    f"this turn is exhausted ({state.max_tool_calls} "
                                    "calls). Stop calling tools and answer "
                                    "now with what you have already gathered."
                                ),
                            }
                        ],
                    },
                }
            )
            return
        name = params.get("tool")
        scoped = (
            state.thread_id is not None
            and params.get("threadId") == state.thread_id
            and not params.get("namespace")
            and isinstance(name, str)
        )
        valid = False
        text = (
            "Tool error: only run_command with a nonempty command string is allowed."
            if state.wiki is None and state.server_log is None and state.web is None
            and state.weather is None
            else "Tool error: unknown tool or invalid arguments."
        )
        tool_started = time.monotonic()
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
        elif scoped and state.self_improve is not None and name == "request_self_improvement":
            try:
                text = await request_improvement(state.self_improve, args)
                valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
        elif scoped and state.web is not None and name in WEB_PARAMETERS:
            try:
                # Outbound payload inspection replaces the old per-turn
                # taint disable: a web call is blocked only when its
                # query/URL itself copies private content this turn has
                # read (see nezubot_discrod.web_guard). Clean follow-up
                # calls — opening search results, paging, comparing
                # sources — stay available all turn.
                blocked = state.web_guard.check_call(name, args)
                if blocked is not None:
                    text = blocked
                else:
                    text = await asyncio.to_thread(state.web.call, name, args)
                    state.web_read = state.web_read or has_web_content(text)
                    valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
        elif scoped and state.weather is not None and name in WEATHER_PARAMETERS:
            # Weather/clock tools hit fixed public APIs with structured
            # arguments and results; they are not web content and do not
            # take part in the web guard accounting above.
            try:
                text = await asyncio.to_thread(state.weather.call, name, args)
                valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
        elif scoped and state.server_log is not None and name in LOG_PARAMETERS:
            state.private_read = True
            try:
                text = await asyncio.to_thread(state.server_log.call, name, args)
                valid = True
            except (ValueError, TypeError) as error:
                text = f"Tool error: {error}"
            state.web_guard.note_private(text)
        elif scoped and state.wiki is not None and name in TOOL_PARAMETERS:
            state.private_read = True
            try:
                text = await asyncio.to_thread(state.wiki.call, name, args)
                valid = True
            except WikiError as error:
                text = f"Tool error: {error}"
            state.web_guard.note_private(text)
        tool_elapsed = time.monotonic() - tool_started
        if tool_elapsed >= 3:
            logger.info(
                "Slow tool call: %s took %.1fs", name, tool_elapsed,
            )
        if valid:
            state.sources = collect_tool_sources(state.sources, name, text)
        if valid and state.tool_calls == state.max_tool_calls - 5:
            text += (
                "\n(Tool budget note: 5 tool calls left in this turn; "
                "start wrapping up.)"
            )
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
        async with self._init_lock:
            if not self.initialized:
                await self.initialize(refresh_token=refresh_token)

    async def complete(
        self, model, messages, cwd, *, wiki=None, server_log=None, web=None,
        weather=None, ambient=False,
        arbiter=False,
        server_wiki=False,
        private_read=False,
        self_improve=None,
        thread_id=None,
        input_text=None,
        usage_baseline=None,
        scope=None,
        result_box=None,
        images=None,
    ):
        # `scope` is accepted (and ignored here) so adapters that forward
        # the client's completion arguments verbatim keep working; thread
        # reuse is orchestrated by CodexClient, not by this class.
        config = _thread_config(
            model, messages, wiki=wiki, server_log=server_log, web=web,
            weather=weather,
            ambient=ambient, arbiter=arbiter, server_wiki=server_wiki,
            self_improve=self_improve,
            run_command_enabled=self.run_command_enabled,
        )
        turn_started = time.monotonic()
        state = _TurnState(max_tool_calls=self.max_tool_calls)
        state.tool_free = config.tool_free
        state.self_improve = config.self_improve
        state.private_read = private_read
        # Fresh outbound-payload guard for this turn, seeded from the
        # conversation being sent: each user message is a turn payload
        # whose non-"message" fields are injected private context
        # (retrieved excerpts, wiki context, recent messages), while
        # the "message" fields are the users' own words and exempt.
        for message in messages or []:
            if isinstance(message, dict) and message.get("role") == "user":
                state.web_guard.seed_from_user_payload(message.get("content"))
        if input_text is not None:
            state.web_guard.seed_from_user_payload(input_text)
        state.usage_baseline = usage_baseline or {}
        state.wiki = wiki
        state.server_log = server_log
        state.web = web
        state.weather = weather
        self._ensure_reader()
        registered = False
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
                state.thread_id = thread["thread"]["id"]
            else:
                # Reused thread: its policy was verified when it was
                # created under the same config hash.
                state.thread_id = thread_id
            if state.thread_id in self._turns:
                # The server rejects a second turn on a thread that
                # already has one ("The thread already has an active
                # turn."); refuse here, before turn/start, so the
                # caller gets a clean client-side error instead. The
                # client's per-thread locks make this unreachable on
                # the persistent path — this is the second layer.
                raise RuntimeError(
                    "Codex thread already has an active turn; "
                    "concurrent turns on one thread are not supported"
                )
            self._turns[state.thread_id] = state
            registered = True
            self._interest.set()
            await self.request(
                "turn/start",
                {
                    "threadId": state.thread_id,
                    # Image items ride after the text item: the
                    # backend reads the turn's attachments from the
                    # same input list (probe-verified 2026-11-27).
                    "input": [
                        {
                            "type": "text",
                            "text": (
                                input_text if input_text is not None
                                else json.dumps(messages, ensure_ascii=False)
                            ),
                        },
                        *(images or []),
                    ],
                    **({"effort": "low"} if state.tool_free else {}),
                },
            )
            answer = ""
            while True:
                kind, data = await state.queue.get()
                if state.error is not None:
                    raise state.error
                if kind == "closed":
                    raise data
                if kind == "request":
                    await self._handle_request(state, data)
                    continue
                params = data.get("params", {})
                if data.get("method") in {"item/started", "item/completed"}:
                    item_kind = params.get("item", {}).get("type")
                    if state.tool_free and item_kind == "dynamicToolCall":
                        raise CodexSecurityError("Tool-free Codex turn cannot call tools")
                    if item_kind not in PASSIVE_ITEM_TYPES | {"dynamicToolCall"}:
                        raise CodexSecurityError("Unexpected built-in tool item; stopping Codex")
                if (data.get("method") == "thread/tokenUsage/updated"
                        and params.get("threadId") == state.thread_id):
                    total = params.get("tokenUsage", {}).get("total", {})
                    state.thread_totals = total
                    base = state.usage_baseline
                    counts = token_usage({
                        "prompt_tokens": _usage_delta(
                            total.get("inputTokens"), base.get("inputTokens")),
                        "completion_tokens": _usage_delta(
                            total.get("outputTokens"), base.get("outputTokens")),
                    })
                    if counts is not None:
                        state.usage = {
                            "prompt_tokens": counts[0],
                            "completion_tokens": counts[1],
                        }
                        cached = _usage_delta(
                            total.get("cachedInputTokens"), base.get("cachedInputTokens"))
                        if cached is not None:
                            # Prompt-cache observability: how much of this
                            # turn's input the backend served from its
                            # prefix cache (delta of the thread totals).
                            state.usage["cached_input_tokens"] = cached
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
                    usage = state.usage or {}
                    logger.info(
                        "Codex turn finished in %.1fs (turn_id=%s, "
                        "tool_calls=%d, input=%s, cached=%s, output=%s)",
                        time.monotonic() - turn_started, current_turn_id(),
                        state.tool_calls,
                        usage.get("prompt_tokens"),
                        usage.get("cached_input_tokens"),
                        usage.get("completion_tokens"),
                    )
                    return answer
        finally:
            if registered:
                # Remove only this turn's own registration: a refused
                # double-turn attempt (which never registers) must
                # not evict the turn legitimately running on the
                # thread, and neither may a late finally from an
                # earlier turn on a reused thread.
                if self._turns.get(state.thread_id) is state:
                    del self._turns[state.thread_id]
                # Compatibility mirrors: the most recent turn's outcome
                # stays readable on the server object, as the per-call
                # path and older callers expect. Concurrent turns never
                # read these; their results travel via result_box.
                self.thread_id = state.thread_id
                self.usage = state.usage
                self.sources = state.sources
                self.thread_totals = state.thread_totals
                self.private_read = state.private_read
                self.web_read = state.web_read
                if result_box is not None:
                    result_box.update(
                        {
                            "thread_id": state.thread_id,
                            "usage": state.usage,
                            "sources": state.sources,
                            "thread_totals": state.thread_totals,
                        }
                    )


class CodexClient:
    def __init__(self, settings, *, runner=run_command, max_tool_calls=20,
                 run_command_enabled=True):
        self.settings = settings
        self.runner = runner
        self.run_command_enabled = run_command_enabled
        self.chat = SimpleNamespace(completions=self)
        self.lock = asyncio.Lock()
        # Admission control for turns (v1.25.0): create() no longer
        # serializes on self.lock — it takes one of this many turn
        # slots instead, and turns proceed in parallel on the shared
        # app-server. self.lock remains as the exclusive lock for the
        # auth-adjacent operations (rate_limits, warm_server_wiki_auth,
        # close), exactly as before.
        self.max_concurrent_turns = _max_concurrent_turns_from_env()
        self._turn_slots = asyncio.Semaphore(self.max_concurrent_turns)
        # Turns currently queued for a turn slot (excluding holders):
        # reported with each turn's lock wait so the admission cost
        # stays as visible as the serialization cost was.
        self._lock_waiters = 0
        # Serializes persistent-server startup and teardown, so two
        # concurrent turns cannot spawn it (or drop it) twice.
        self._server_lock = asyncio.Lock()
        # One lock per persistent-thread registry key: two turns that
        # would land on the same thread serialize on it, while turns
        # for different scopes run in parallel. Entries are dropped
        # opportunistically by forget_threads; the map stays bounded
        # by the distinct (scope, config) pairs actually seen.
        self._thread_locks = {}
        self.thread_input_budget = _thread_input_budget_from_env()
        self.runtime = None
        self.max_tool_calls = max_tool_calls
        self._version_key = None
        self._catalog_key = None
        self._catalog_models = None
        # Plan A: one persistent app-server plus a registry of reusable
        # per-scope threads. Since
        # v1.25.0 the turns on it run concurrently (up to the turn
        # slots); only turns sharing one thread still serialize, on
        # that thread's key lock.
        self._persistent = None
        self._persistent_ctx = None
        self._persistent_work = None
        self._persistent_failures = 0
        self._threads = OrderedDict()

    @asynccontextmanager
    async def _turn_lock(self):
        """A turn slot, with the acquisition wait measured and logged.

        Until v1.23.0 this wrapped the client's single lock and every
        turn waited for the previous one to finish. The wait line
        ("Codex client lock wait") is kept verbatim so the cost stays
        comparable across versions — it now measures the wait for one
        of the concurrency slots (MAX_CONCURRENT_TURNS), which is
        ~0ms unless that many turns are already running. One line per
        create() call: the wait in ms, the turn's correlation id, and
        how many turns were still queued behind this one when it got
        its slot.
        """
        wait_started = time.monotonic()
        self._lock_waiters += 1
        try:
            await self._turn_slots.acquire()
        except BaseException:
            self._lock_waiters -= 1
            raise
        self._lock_waiters -= 1
        logger.info(
            "Codex client lock wait %.1fms (turn_id=%s, waiters=%d)",
            (time.monotonic() - wait_started) * 1000,
            current_turn_id(), self._lock_waiters,
        )
        try:
            yield
        finally:
            self._turn_slots.release()

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
        for key in [key for key in self._thread_locks if key[0] in wanted]:
            if not self._thread_locks[key].locked():
                del self._thread_locks[key]

    def catalog_slugs(self):
        """Slugs of the most recently parsed Codex catalog, or None
        when no catalog has been loaded in this client's lifetime
        yet (the catalog is fetched lazily on the first turn)."""
        if self._catalog_models is None:
            return None
        return {entry["slug"] for entry in self._catalog_models}

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
            "PATH", "LANG", "TZ", "SSL_CERT_FILE",
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
                try:
                    return await self._persistent.request("account/rateLimits/read", {})
                except (CodexConnectionError, OSError):
                    await self._drop_persistent_server("connection lost during rate_limits")
            root = self.prepare_runtime()
            auth = root / "home" / "auth.json"
            with self._workspace(root) as (cwd, env):
                await self._verify_version(env)
                async with self.server(cwd, env) as server:
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
        server = AppServer(
            process, self.settings.sandbox, self.runner, self.max_tool_calls,
            run_command_enabled=self.run_command_enabled,
        )
        try:
            yield server
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            await server._stop_reader()

    async def _drop_persistent_server(self, reason, *, warn=True):
        async with self._server_lock:
            await self._drop_persistent_server_locked(reason, warn=warn)

    async def _drop_persistent_server_locked(self, reason, *, warn=True):
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

        Startup and teardown are serialized on self._server_lock:
        with turns running concurrently, two of them must not spawn
        (or drop) the shared process at the same time.
        """
        async with self._server_lock:
            return await self._ensure_persistent_server_locked(model)

    async def _ensure_persistent_server_locked(self, model):
        if self._persistent is not None:
            if self._persistent.process.returncode is None:
                return self._persistent
            logger.warning(
                "Persistent Codex app-server exited (code %s); restarting",
                self._persistent.process.returncode,
            )
            await self._drop_persistent_server_locked("process exited")
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

    def _estimated_next_input(self, entry, last_message):
        """Estimated input tokens if `entry`'s thread took one more
        turn, or None when the budget gate does not apply (budget
        disabled, or the thread has no measured turn yet)."""
        if self.thread_input_budget <= 0 or entry.last_input_tokens is None:
            return None
        payload = json.dumps([last_message], ensure_ascii=False)
        # Character count as the new payload's token estimate:
        # Japanese text tokenizes at roughly a token per character,
        # so this errs on the safe (early-retire) side. The previous
        # turn's output is included because it joined the thread's
        # context after that turn's input was measured.
        return (
            entry.last_input_tokens
            + (entry.last_output_tokens or 0)
            + len(payload)
        )

    @staticmethod
    def _response(text, usage, sources):
        return SimpleNamespace(
            usage=usage,
            sources=tuple(sources or ()),
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
        weather=None,
        ambient=False, private_read=False, self_improve=None, images=None,
    ):
        config = _thread_config(
            model, messages, wiki=wiki, server_log=server_log, web=web,
            weather=weather,
            ambient=ambient, self_improve=self_improve,
            run_command_enabled=self.run_command_enabled,
        )
        server = await self._ensure_persistent_server(model)
        key = (scope, config.config_hash)
        # Turns for the same registry key would land on the same
        # thread, and the server rejects a second concurrent turn on
        # one thread — so this section (registry lookup, the turn
        # itself, the registry update) runs under the key's own lock.
        # Different keys proceed in parallel up to the turn-slot
        # limit, which is the concurrency this refactor exists for.
        key_lock = self._thread_locks.setdefault(key, asyncio.Lock())
        async with key_lock:
            now = time.monotonic()
            entry = self._thread_entry(key, now)
            if entry is not None:
                estimate = self._estimated_next_input(entry, messages[-1])
                if estimate is not None and estimate > self.thread_input_budget:
                    # Retire the thread before this turn: it re-seeds
                    # below from the full local history, exactly like a
                    # turn-cap rollover. The fresh thread's actual input
                    # lands in this turn's "Codex turn finished" line
                    # under the same turn_id, so before/after sizes can
                    # be compared from the log.
                    logger.info(
                        "Codex thread retired by input budget (turn_id=%s, "
                        "estimated_input=%d, budget=%d, last_input=%s, "
                        "thread_turns=%d)",
                        current_turn_id(), estimate, self.thread_input_budget,
                        entry.last_input_tokens, entry.turns,
                    )
                    del self._threads[key]
                    entry = None
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
            result_box = {}
            text = await server.complete(
                model,
                messages,
                str(self._persistent_work),
                wiki=wiki,
                **({"server_log": server_log} if server_log is not None else {}),
                **({"web": web} if web is not None else {}),
                **({"weather": weather} if weather is not None else {}),
                **({"ambient": True} if ambient else {}),
                **({"private_read": True} if private_read else {}),
                **({"self_improve": self_improve} if self_improve is not None else {}),
                thread_id=thread_id,
                input_text=input_text,
                usage_baseline=baseline,
                result_box=result_box,
                images=images,
            )
            now = time.monotonic()
            if entry is None:
                entry = _ThreadEntry(
                    thread_id=result_box["thread_id"],
                    created_at=now, last_used_at=now,
                )
                self._threads[key] = entry
                while len(self._threads) > THREAD_REGISTRY_LIMIT:
                    self._threads.popitem(last=False)
            entry.turns += 1
            entry.last_used_at = now
            entry.totals = result_box["thread_totals"]
            usage = result_box["usage"] or {}
            if type(usage.get("prompt_tokens")) is int:
                entry.last_input_tokens = usage["prompt_tokens"]
            if type(usage.get("completion_tokens")) is int:
                entry.last_output_tokens = usage["completion_tokens"]
            self._threads.move_to_end(key)
            self._persistent_failures = 0
            return self._response(
                text, result_box["usage"], result_box["sources"])

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
        self, *, model, messages, wiki=None, server_log=None, web=None, weather=None,
        ambient=False,
        arbiter=False, server_wiki=False, private_read=False, self_improve=None, scope=None,
        images=None,
        **kwargs
    ):
        async with self._turn_lock(), asyncio.timeout(600):
            if self._persistent_eligible(scope, messages, arbiter=arbiter, server_wiki=server_wiki):
                try:
                    return await self._create_persistent(
                        model=model,
                        messages=messages,
                        scope=scope,
                        wiki=wiki,
                        server_log=server_log,
                        web=web,
                        weather=weather,
                        ambient=ambient,
                        private_read=private_read,
                        self_improve=self_improve,
                        images=images,
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
                weather=weather,
                ambient=ambient,
                arbiter=arbiter,
                server_wiki=server_wiki,
                private_read=private_read,
                self_improve=self_improve,
                images=images,
            )

    async def _create_legacy(
        self, *, model, messages, wiki=None, server_log=None, web=None, weather=None,
        ambient=False,
        arbiter=False, server_wiki=False, private_read=False, self_improve=None,
        images=None,
    ):
        # Never load the user's config, MCP servers, plugins, hooks or skills.
        root = self.prepare_runtime()
        with self._workspace(root) as (cwd, env):
            await self._verify_version(env)
            catalog = await self.direct_catalog(root, cwd, env, model)
            async with self.server(cwd, env, catalog) as server:
                result_box = {}
                text = await server.complete(
                    model,
                    messages,
                    str(cwd),
                    wiki=wiki,
                    **({"server_log": server_log} if server_log is not None else {}),
                    **({"web": web} if web is not None else {}),
                    **({"weather": weather} if weather is not None else {}),
                    **({"ambient": True} if ambient else {}),
                    **({"arbiter": True} if arbiter else {}),
                    **({"server_wiki": True} if server_wiki else {}),
                    **({"private_read": True} if private_read else {}),
                    **({"self_improve": self_improve} if self_improve is not None else {}),
                    result_box=result_box,
                    images=images,
                )
        return self._response(text, result_box["usage"], result_box["sources"])
