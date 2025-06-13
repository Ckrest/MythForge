from __future__ import annotations

from typing import Any, Iterable

from ..main import ChatRequest


def prepare(req: ChatRequest, history: list) -> tuple[str, str]:
    """Return prompts for helper calls."""

    del history
    return req.global_prompt or "", req.message


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return system and user prompt text."""

    return system_text, user_text


def response(result: Any) -> str:
    """Return a single parsed model response."""

    from ..call_core import parse_response

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return parse_response(result)
