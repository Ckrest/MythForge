from __future__ import annotations

from typing import Any, Iterable

from ..main import ChatRequest


def prepare(req: ChatRequest, history: list) -> tuple[str, str]:
    """Return basic prompts when no type matches."""

    del history
    return req.global_prompt or "", req.message


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return system and user prompt text."""

    return system_text, user_text


def response(result: Any) -> Any:
    """Return the raw model response."""

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return result
