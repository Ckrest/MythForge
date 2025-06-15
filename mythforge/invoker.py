from __future__ import annotations

"""LLM invocation utilities."""

from typing import Any, Dict

from .model import call_llm


class LLMInvoker:
    """Simple wrapper around :func:`call_llm`."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}

    def load_model(self, config: Dict[str, Any]) -> None:
        """Store ``config`` for later invocations."""

        self.config = config

    def invoke(self, prompt: str, options: Dict[str, Any] | None = None):
        """Invoke the language model with ``prompt`` and ``options``."""

        opts = options or {}
        return call_llm("", prompt, **opts)


LLM_INVOKER = LLMInvoker()
