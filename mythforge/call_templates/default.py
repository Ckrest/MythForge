from __future__ import annotations

from typing import Any, Iterable

from ..call_core import CallData


def prepare(call: CallData, history: list) -> tuple[str, str]:
    """Return basic prompts when no type matches."""

    del history
    return call.global_prompt or "", call.message


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return system and user prompt text."""

    return system_text, user_text


def response(result: Any) -> Any:
    """Return the raw model response."""

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return result
