from __future__ import annotations

"""Prompt helpers for standard chat interactions."""

from typing import Any, Iterable, Iterator, List, Dict, TYPE_CHECKING

from ..memory import MEMORY_MANAGER
from ..logger import LOGGER
from ..response_parser import ResponseParser

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..call_core import CallData


def prepare(call: "CallData") -> tuple[str, str]:
    """Return prompts for a standard chat call."""

    if not call.global_prompt:
        from ..call_core import _default_global_prompt

        call.global_prompt = MEMORY_MANAGER.global_prompt or _default_global_prompt()

    parts = [call.global_prompt]
    goals = MEMORY_MANAGER.goals_data
    if MEMORY_MANAGER.goals_active:
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
    system_text = "\n".join(p for p in parts if p)

    history = MEMORY_MANAGER.load_history(call.chat_id)
    user_text = "\n".join(m.get("content", "") for m in history)

    LOGGER.log(
        "prepared_prompts",
        {
            "call_type": call.call_type,
            "system_text": system_text,
            "user_text": user_text,
        },
    )
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return ``system_text`` and ``user_text`` unchanged."""

    return system_text, user_text


def response(result: Iterable[dict]) -> Iterator[str]:
    """Yield parsed output for streaming responses."""

    return ResponseParser().load(result).parse()
