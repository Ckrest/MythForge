from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List

from ..main import ChatRequest
from ..utils import goals_exists, goals_path


def prepare(
    req: ChatRequest, history: List[Dict[str, Any]]
) -> tuple[str, str]:
    """Return prompts for a standard chat call."""

    system_parts = [req.global_prompt or ""]
    if goals_exists(req.chat_id):
        try:
            with open(goals_path(req.chat_id), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                if data.get("character"):
                    system_parts.append(data["character"])
                if data.get("setting"):
                    system_parts.append(data["setting"])
        except Exception as exc:
            print(f"Failed to load goals for '{req.chat_id}': {exc}")
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
