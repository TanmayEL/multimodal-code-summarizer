"""
tiny LLM wrapper.
right now this is basically a stub with 2 modes:
- mock mode (default): returns a canned summary (good for tests)
- later: real API calls (OpenAI / Anthropic / local LLM ... etc)
I can build / test the pipeline without depending on network.
"""

from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str = "mock"
    model_name: str = "gpt-like-placeholder"


class LLMClient:
    """
    Very small abstraction layer around whatever LLM I end up using.
    """

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        For now this just returns a dumb string so I dontt spam any API.
        """
        if self.cfg.provider == "mock":
            prefix = prompt[:120].replace("\n", " ")
            return (
                "Summary (mock LLM): this PR changes code related to: "
                f"{prefix} ..."
            )

        #I can use OpenAI / Claude / Llama.cpp here later
        raise NotImplementedError("Real LLM provider not wired yet, still WIP")



