import asyncio
import base64
import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager, contextmanager
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


class CodexSecurityError(RuntimeError):
    pass


class CodexRateLimited(RuntimeError):
    pass


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

    async def send(self, data):
        self.process.stdin.write((json.dumps(data) + "\n").encode())
        await self.process.stdin.drain()

    async def receive(self):
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("Codex app-server closed unexpectedly")
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

    async def complete(
        self, model, messages, cwd, *, wiki=None, server_log=None, web=None, ambient=False,
        arbiter=False,
        server_wiki=False,
        private_read=False,
        self_improve=None,
    ):
        self.tool_free = arbiter or server_wiki
        if (arbiter and server_wiki) or (
            self.tool_free and (wiki is not None or server_log is not None or web is not None
                                or ambient or self_improve is not None)
        ):
            raise CodexSecurityError("Tool-free Codex turn cannot receive conversation tools")
        self.self_improve = None if ambient or self.tool_free else self_improve
        self.private_read = private_read
        self.web_read = False
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
            and not self.tool_free
            and prompt == SELF_MOD_PROMPT
        ):
            raise CodexSecurityError("Only fixed system prompts are supported")
        self.wiki = wiki
        self.server_log = server_log
        self.web = web
        await self.initialize()
        tools = [
            *(TOOLS if self.run_command_enabled else []),
            *(WIKI_TOOLS if wiki is not None else []),
            *(LOG_TOOLS if server_log is not None else []),
            *(WEB_TOOLS if web is not None else []),
            *(SELF_IMPROVEMENT_TOOLS if self.self_improve is not None else []),
        ]
        if self.tool_free:
            tools = []
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
                "baseInstructions": prompt,
                "developerInstructions": (
                    "Do not use any tools. Return only the short decision JSON."
                    if arbiter else "Do not use any tools. Return only the Server Wiki JSON."
                    if server_wiki else CODEX_TOOL_INSTRUCTIONS
                    if wiki is None and server_log is None and web is None
                    else CODEX_WIKI_TOOL_INSTRUCTIONS
                ),
                "dynamicTools": [
                    {
                        "type": "function",
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "inputSchema": tool["function"]["parameters"],
                        "deferLoading": False,
                    }
                    for tool in tools
                ],
            },
        )
        if (
            thread.get("approvalPolicy") != "never"
            or thread.get("sandbox", {}).get("type") != "readOnly"
            or thread.get("sandbox", {}).get("networkAccess", False)
        ):
            raise CodexSecurityError("Codex did not apply the requested security policy")
        self.thread_id = thread["thread"]["id"]
        await self.request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": json.dumps(messages, ensure_ascii=False)}],
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
                counts = token_usage({
                    "prompt_tokens": total.get("inputTokens"),
                    "completion_tokens": total.get("outputTokens"),
                })
                if counts is not None:
                    self.usage = {"prompt_tokens": counts[0], "completion_tokens": counts[1]}
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

    async def close(self):
        async with self.lock:
            if self.runtime is not None:
                self.runtime.cleanup()
                self.runtime = None

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

    @contextmanager
    def _workspace(self, root):
        with tempfile.TemporaryDirectory(prefix="work-", dir=root) as work:
            env = {key: os.environ[key] for key in (
                "PATH", "LANG", "SSL_CERT_FILE",
                # The Codex CLI itself needs the host's egress proxy, if any;
                # without it a proxied host cannot reach OpenAI at all.
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            ) if key in os.environ}
            env.update(HOME=str(root), CODEX_HOME=str(root / "home"))
            yield Path(work), env

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

    async def create(
        self, *, model, messages, wiki=None, server_log=None, web=None, ambient=False,
        arbiter=False, server_wiki=False, private_read=False, self_improve=None, **kwargs
    ):
        async with self.lock, asyncio.timeout(600):
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
            ]
        )
