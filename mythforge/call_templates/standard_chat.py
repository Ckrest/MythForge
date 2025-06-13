from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List

from ..call_core import CallData
from ..utils import goals_exists, goals_path


def prepare(call: CallData, history: List[Dict[str, Any]]) -> tuple[str, str]:
    """Return prompts for a standard chat call."""

    system_parts = [call.global_prompt or ""]
    if goals_exists(call.chat_id):
        try:
            with open(goals_path(call.chat_id), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                if data.get("character"):
                    system_parts.append(data["character"])
                if data.get("setting"):
                    system_parts.append(data["setting"])
        except Exception as exc:
            print(f"Failed to load goals for '{call.chat_id}': {exc}")
    system_text = "\n".join(p for p in system_parts if p)
    user_text = "\n".join(m.get("content", "") for m in history)
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return system and user prompt text."""

    return system_text, user_text


def response(result: Iterable[dict]) -> Iterator[str]:
    """Yield parsed output for streaming responses."""

    from ..call_core import stream_parsed

    return stream_parsed(result)
