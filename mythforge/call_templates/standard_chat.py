from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List

from ..call_core import CallData, _default_global_prompt, format_for_model
from .. import memory


def prepare_system_text(call: CallData) -> str:
    """Return the system prompt text for ``call``."""

    if not call.global_prompt:
        call.global_prompt = _default_global_prompt()

    parts = [call.global_prompt]
    goals = memory.MEMORY.goals_data
    if goals.enabled:
        if goals.character:
            parts.append(goals.character)
        if goals.setting:
            parts.append(goals.setting)
        if goals.active_goals:
            joined = ", ".join(str(g) for g in goals.active_goals)
            parts.append(f"Active Goals: {joined}")
        if goals.deactive_goals:
            joined = ", ".join(str(g) for g in goals.deactive_goals)
            parts.append(f"Completed Goals: {joined}")
    return "\n".join(p for p in parts if p)


def prepare_user_text(history: List[Dict[str, Any]]) -> str:
    """Return the user prompt text from ``history``."""

    return "\n".join(m.get("content", "") for m in history)


def prepare(call: CallData, history: List[Dict[str, Any]]) -> tuple[str, str]:
    """Return prompts for a standard chat call."""

    system_text = prepare_system_text(call)
    user_text = prepare_user_text(history)
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return formatted prompts for model input."""

    combined = format_for_model(system_text, user_text, "standard_chat")
    return "", combined


def response(result: Iterable[dict]) -> Iterator[str]:
    """Yield parsed output for streaming responses."""

    from ..call_core import stream_parsed

    return stream_parsed(result)
