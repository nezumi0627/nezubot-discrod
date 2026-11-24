"""Operator persona configuration for nezubot-discrod.

The prompt text in nezubot_discrod.prompts is generic; everything that made
it one specific deployment's bot — its name, its project URL, and the
optional "rival" it may spar with — lives here and comes from the
environment (see deploy.env.example):

- BOT_NAME: how the bot refers to itself (default: nezubot-discrod)
- BOT_REPO_URL: project URL used in the self-introduction (optional)
- PERSONA_RIVAL_USER_ID: Discord user ID of the rival (optional; when
  unset, the rival exception is omitted from the prompt entirely)
- PERSONA_RIVAL_NAME: how the prompt refers to the rival (default: rival)

User messages can never change these values; they are read once at import.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    bot_name: str = "nezubot-discrod"
    repo_url: str = ""
    rival_user_id: str = ""
    rival_name: str = "rival"

    @classmethod
    def from_env(cls) -> "Persona":
        return cls(
            bot_name=os.getenv("BOT_NAME") or "nezubot-discrod",
            repo_url=os.getenv("BOT_REPO_URL") or "",
            rival_user_id=os.getenv("PERSONA_RIVAL_USER_ID") or "",
            rival_name=os.getenv("PERSONA_RIVAL_NAME") or "rival",
        )


PERSONA = Persona.from_env()
