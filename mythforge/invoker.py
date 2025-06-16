from __future__ import annotations

"""LLM invocation utilities."""

from typing import Any, Dict
import os

from .logger import LOGGER

from .model import call_llm


class LLMInvoker:
    """Simple wrapper around :func:`call_llm`."""

    def __init__(self) -> None:
        """Store model configuration for later use."""

        self.config: Dict[str, Any] = {}

    def load_model(self, config: Dict[str, Any]) -> None:
        """Persist ``config`` to be used when invoking the model."""

        self.config = config

    def invoke(self, prompt: str, options: Dict[str, Any] | None = None):
        """Send ``prompt`` to the model and return its output."""

        LOGGER.log(
            "chat_flow",
            {
                "function": "LLMInvoker.invoke",
            },
        )
        opts = options or {}
        LOGGER.log(
            "chat_flow",
            {
                "function": "LLMInvoker.invoke",
                "prompt": prompt,
                "options": opts,
            },
        )
        model_name = (
            self.config.get("model_name")
            or os.environ.get("MODEL_NAME", "gpt-4")
        )
        return call_llm(model_name, prompt, **opts)


LLM_INVOKER = LLMInvoker()
