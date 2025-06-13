from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List

from ..call_core import CallData, _default_global_prompt, format_for_model
from ..utils import goals_exists, goals_path


def prepare_system_text(call: CallData) -> str:
    """Return the system prompt text for ``call``."""

    if not call.global_prompt:
        call.global_prompt = _default_global_prompt()

    parts = [call.global_prompt]
    if goals_exists(call.chat_id):
        try:
            with open(goals_path(call.chat_id), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                if data.get("character"):
                    parts.append(data["character"])
                if data.get("setting"):
                    parts.append(data["setting"])
                if data.get("in_progress"):
                    goals = ", ".join(str(g) for g in data["in_progress"])
                    parts.append(f"Active Goals: {goals}")
                if data.get("completed"):
                    goals = ", ".join(str(g) for g in data["completed"])
                    parts.append(f"Completed Goals: {goals}")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Failed to load goals for '{call.chat_id}': {exc}")
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
