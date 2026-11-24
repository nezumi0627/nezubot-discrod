import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from openai import AsyncOpenAI

from nezubot_discrod.assistant import Assistant
from nezubot_discrod.codex import CodexClient
from nezubot_discrod.policy import validate_workspace
from nezubot_discrod.prompts import SELF_MOD_PROMPT
from nezubot_discrod.sandbox import run_command, shutdown_sandboxes, truncate_output

CHECK_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 ruff check --no-cache . && "
    "PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider"
)


def github_repository(source: str) -> str:
    for prefix in (
        "https://github.com/",
        "git@github.com:",
        "github.com/",
        "https://git-manager.devin.ai/proxy/github.com/",
    ):
        if source.startswith(prefix):
            source = source[len(prefix) :]
            break
    source = source.removesuffix(".git").rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source):
        raise ValueError("SELF_REPO/origin must identify a GitHub owner/repository")
    if any(part in {".", ".."} for part in source.split("/")):
        raise ValueError("Invalid repository")
    return source


class GitHubClient:
    def __init__(self, token: str):
        self.token = token

    def request(self, method, path, body=None):
        request = Request(
            "https://api.github.com" + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "nezubot-discrod",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read(1024 * 1024)
        except HTTPError as error:
            raise RuntimeError(f"GitHub API failed (HTTP {error.code})") from None
        return json.loads(data) if data else {}

    def default_branch(self, repo):
        return self.request("GET", f"/repos/{repo}")["default_branch"]

    def create_pr(self, repo, branch, base, request, summary):
        return self.request(
            "POST",
            f"/repos/{repo}/pulls",
            {
                "head": branch,
                "base": base,
                "title": f"Bot improvement: {request[:120]}",
                "body": f"Owner request:\n{request[:8000]}\n\nAI summary:\n{summary[:5000]}",
            },
        )

    def merge_pr(self, repo, number, sha):
        result = self.request(
            "PUT",
            f"/repos/{repo}/pulls/{number}/merge",
            {
                "sha": sha,
                "merge_method": "squash",
            },
        )
        if not result.get("merged"):
            raise RuntimeError("GitHub did not merge the PR")

    def close_pr(self, repo, number):
        self.request("PATCH", f"/repos/{repo}/pulls/{number}", {"state": "closed"})

    def delete_branch(self, repo, branch):
        self.request("DELETE", f"/repos/{repo}/git/refs/heads/{quote(branch, safe='')}")


def host_git(args, *, cwd, git_dir=None, token="") -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="/bin/false",
    )
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
            GIT_CONFIG_VALUE_0=f"Authorization: Basic {auth}",
        )
    command = [
        "git",
        "--literal-pathspecs",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "user.name=Discord Sandbox Bot",
        "-c",
        "user.email=bot@localhost",
    ]
    if git_dir is not None:
        command.extend(["--git-dir", str(git_dir), "--work-tree", str(cwd)])
    result = subprocess.run(
        [*command, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Git operation failed ({args[0]}, exit {result.returncode})")
    output = result.stdout.decode(errors="replace")
    return output if "-z" in args else output.strip()


def restart_bot():
    shutdown_sandboxes()
    os.execv(sys.executable, [sys.executable, *sys.argv])


@dataclass
class Proposal:
    repo: str
    number: int
    url: str
    branch: str
    base: str
    sha: str
    stat: str
    merged: bool = False


class ImprovementService:
    def __init__(
        self,
        settings,
        *,
        github=None,
        git=host_git,
        runner=run_command,
        editor=None,
        restart=restart_bot,
    ):
        self.settings = settings
        self.github = github or GitHubClient(settings.github_token)
        self.git = partial(git, token=settings.github_token)
        self.runner = runner
        self.editor = editor or self.edit
        self.restart = restart
        self.checkout = Path(
            settings.self_checkout or Path(__file__).resolve().parents[1]
        ).resolve()
        self.lock = asyncio.Lock()
        self.pending = None

    def authorize(self, user_id):
        if user_id != self.settings.owner_id:
            raise PermissionError("オーナーのみ利用できます。")

    async def git_call(self, args, cwd, git_dir=None):
        return await asyncio.to_thread(self.git, args, cwd=cwd, git_dir=git_dir)

    async def edit(self, request, workspace, sandbox):
        runner = partial(self.runner, workspace=workspace)
        if self.settings.provider == "codex":
            client = CodexClient(
                replace(self.settings, sandbox=sandbox),
                runner=runner,
                max_tool_calls=self.settings.selfmod_iterations * 4,
            )
        else:
            client = AsyncOpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=60,
                max_retries=1,
            )
        try:
            assistant = Assistant(
                client,
                self.settings.model,
                sandbox,
                runner=runner,
                max_iterations=self.settings.selfmod_iterations,
                history_turns=1,
                system_prompt=SELF_MOD_PROMPT,
            )
            return await assistant.reply(0, request)
        finally:
            await client.close()

    async def validate_changes(self, workspace, metadata, base_sha):
        validate_workspace(workspace, "", "")
        raw = await self.git_call(
            ["diff", "--raw", "--no-abbrev", "--no-renames", "--no-ext-diff",
             "--no-textconv", "-z", base_sha], workspace, metadata,
        )
        untracked = await self.git_call(["ls-files", "--others", "-z"], workspace, metadata)
        validate_workspace(workspace, raw, untracked)

    async def propose(self, user_id, request):
        self.authorize(user_id)
        if not request.strip() or len(request) > 8000:
            raise ValueError("依頼は 1〜8000 文字で入力してください。")
        if self.lock.locked() or self.pending is not None:
            raise RuntimeError("自己改善の処理または承認待ちの PR があります。")
        if not self.settings.github_token:
            raise RuntimeError("GITHUB_TOKEN が未設定です。")
        if not 1 <= self.settings.selfmod_iterations <= 100:
            raise ValueError("SELFMOD_ITERATIONS must be between 1 and 100")
        async with self.lock:
            source = self.settings.self_repo or await self.git_call(
                ["remote", "get-url", "origin"],
                self.checkout,
            )
            repo = github_repository(source)
            base = await asyncio.to_thread(self.github.default_branch, repo)
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", base) or base.startswith("-"):
                raise ValueError("Invalid default branch")
            slug = re.sub(r"[^a-z0-9]+", "-", request.lower()).strip("-")[:35] or "improve"
            branch = f"selfmod/{time.time_ns()}-{slug}"
            sandbox = replace(
                self.settings.sandbox,
                image=self.settings.selfmod_image,
                timeout=self.settings.selfmod_timeout,
                memory="512m",
                cpus=1,
            )
            with tempfile.TemporaryDirectory(prefix="discord-selfmod-") as root:
                root = Path(root)
                workspace, metadata = root / "work", root / "git-metadata"
                template = root / "empty-template"
                template.mkdir()
                await self.git_call(
                    [
                        "clone",
                        "--single-branch",
                        "--branch",
                        base,
                        f"--template={template}",
                        f"https://github.com/{repo}.git",
                        str(workspace),
                    ],
                    root,
                )
                shutil.move(str(workspace / ".git"), metadata)
                await self.git_call(["switch", "-c", branch], workspace, metadata)
                base_sha = await self.git_call(["rev-parse", "HEAD"], workspace, metadata)
                summary = await self.editor(request, workspace, sandbox)
                await self.validate_changes(workspace, metadata, base_sha)
                checks = await asyncio.to_thread(
                    self.runner, CHECK_COMMAND, sandbox, workspace=workspace
                )
                if not checks.startswith("Exit code: 0\n"):
                    raise RuntimeError("検証に失敗したため PR は作成しません。\n" + checks)
                await self.validate_changes(workspace, metadata, base_sha)
                changed = await self.git_call(
                    ["diff", "--name-only", "--no-ext-diff", "--no-textconv", "-z"],
                    workspace,
                    metadata,
                )
                untracked = await self.git_call(
                    ["ls-files", "--others", "--exclude-standard", "-z"],
                    workspace,
                    metadata,
                )
                files = sorted(set(filter(None, changed.split("\0") + untracked.split("\0"))))
                if not files:
                    raise RuntimeError("変更がありません。PR は作成しません。")
                if len(files) > 1000 or any(
                    Path(file).name in {".env", "auth.json", "credentials.json"}
                    or (Path(file).name.startswith(".env.") and Path(file).name != ".env.example")
                    for file in files
                ):
                    raise RuntimeError(
                        "変更ファイル数または機密ファイルの安全チェックに失敗しました。"
                    )
                await self.git_call(["add", "--", *files], workspace, metadata)
                stat = await self.git_call(
                    ["diff", "--cached", "--stat", "--no-ext-diff", "--no-textconv"],
                    workspace,
                    metadata,
                )
                await self.git_call(
                    ["commit", "-m", f"Self improvement: {request[:100]}"], workspace, metadata
                )
                sha = await self.git_call(["rev-parse", "HEAD"], workspace, metadata)
                await self.git_call(
                    ["push", "origin", f"HEAD:refs/heads/{branch}"], workspace, metadata
                )
                pr = await asyncio.to_thread(
                    self.github.create_pr, repo, branch, base, request, summary
                )
                self.pending = Proposal(
                    repo,
                    pr["number"],
                    pr["html_url"],
                    branch,
                    base,
                    sha,
                    truncate_output(stat, 1200),
                )
            return self.pending

    async def approve(self, user_id, number):
        self.authorize(user_id)
        async with self.lock:
            proposal = self.require_pending(number)
            current = await self.git_call(["branch", "--show-current"], self.checkout)
            status = await self.git_call(["status", "--porcelain"], self.checkout)
            origin = await self.git_call(["remote", "get-url", "origin"], self.checkout)
            if current != proposal.base or status or github_repository(origin) != proposal.repo:
                raise RuntimeError("稼働 checkout を既定ブランチ・変更なしの状態にしてください。")
            if not proposal.merged:
                await asyncio.to_thread(self.github.merge_pr, proposal.repo, number, proposal.sha)
                proposal.merged = True
            await self.git_call(["pull", "--ff-only", "origin", proposal.base], self.checkout)
            self.restart()
            self.pending = None

    async def reject(self, user_id, number):
        self.authorize(user_id)
        async with self.lock:
            proposal = self.require_pending(number)
            if proposal.merged:
                raise RuntimeError("すでにマージ済みです。更新・再起動を再試行してください。")
            await asyncio.to_thread(self.github.close_pr, proposal.repo, number)
            await asyncio.to_thread(self.github.delete_branch, proposal.repo, proposal.branch)
            self.pending = None

    def require_pending(self, number):
        if self.pending is None or self.pending.number != number:
            raise RuntimeError("この PR は承認待ちではありません。")
        return self.pending
