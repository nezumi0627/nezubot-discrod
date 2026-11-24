import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def data_dir() -> Path:
    """Directory that holds this deployment's data files (DBs, locks, caches).

    Override with the DATA_DIR environment variable; defaults to
    ~/.nezubot-discrod.
    """
    return Path(os.getenv("DATA_DIR") or Path.home() / ".nezubot-discrod")


@dataclass(frozen=True)
class SandboxSettings:
    image: str = "discord-ai-sandbox:local"
    timeout: float = 15
    memory: str = "128m"
    cpus: float = 0.5
    output_limit: int = 12000

    def __post_init__(self):
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("SANDBOX_TIMEOUT must be positive and finite")
        if not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("SANDBOX_CPUS must be positive and finite")
        if not self.image or self.image.startswith("-") or self.output_limit < 100:
            raise ValueError("Invalid sandbox image or output limit")


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    codex_binary: str
    codex_home: str
    sandbox: SandboxSettings = SandboxSettings()

    @classmethod
    def from_env(cls):
        load_dotenv()
        provider = os.getenv("AI_PROVIDER", "openai")
        if provider not in {"openai", "codex"}:
            raise ValueError("AI_PROVIDER must be openai or codex")
        return cls(
            provider,
            os.getenv("AI_API_KEY", ""),
            os.getenv("AI_BASE_URL", "https://opencode.ai/zen/go/v1"),
            os.getenv("AI_MODEL") or ("gpt-6-luna" if provider == "codex" else "deepseek-v4-flash"),
            os.getenv("CODEX_BINARY", "codex"),
            os.getenv("CODEX_HOME", "~/.codex"),
        )


@dataclass(frozen=True)
class Settings:
    discord_token: str
    api_key: str
    base_url: str
    model: str
    sandbox: SandboxSettings
    provider: str = "openai"
    codex_binary: str = "codex"
    codex_home: str = "~/.codex"
    guild_id: int = 111111111111111111
    owner_id: int = 222222222222222222
    github_token: str = ""
    self_repo: str = ""
    self_checkout: str = ""
    selfmod_image: str = "discord-ai-selfmod:local"
    selfmod_iterations: int = 30
    selfmod_timeout: float = 180

    @classmethod
    def from_env(cls):
        model = ModelSettings.from_env()
        token = os.environ.get("DISCORD_TOKEN", "")
        if not token or (model.provider == "openai" and not model.api_key):
            raise ValueError("DISCORD_TOKEN is required; openai also requires AI_API_KEY")
        if not os.getenv("DISCORD_GUILD_ID") or not os.getenv("DISCORD_OWNER_ID"):
            raise ValueError("DISCORD_GUILD_ID and DISCORD_OWNER_ID are required")
        return cls(
            token,
            model.api_key,
            model.base_url,
            model.model,
            SandboxSettings(
                image=os.getenv("SANDBOX_IMAGE", "discord-ai-sandbox:local"),
                timeout=float(os.getenv("SANDBOX_TIMEOUT", "15")),
                memory=os.getenv("SANDBOX_MEMORY", "128m"),
                cpus=float(os.getenv("SANDBOX_CPUS", "0.5")),
            ),
            provider=model.provider,
            codex_binary=model.codex_binary,
            codex_home=model.codex_home,
            guild_id=int(os.getenv("DISCORD_GUILD_ID", "111111111111111111")),
            owner_id=int(os.getenv("DISCORD_OWNER_ID", "222222222222222222")),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            self_repo=os.getenv("SELF_REPO", ""),
            self_checkout=os.getenv("SELF_CHECKOUT", ""),
            selfmod_image=os.getenv("SELFMOD_IMAGE", "discord-ai-selfmod:local"),
            selfmod_iterations=int(os.getenv("SELFMOD_ITERATIONS", "30")),
            selfmod_timeout=float(os.getenv("SELFMOD_TIMEOUT", "180")),
        )
