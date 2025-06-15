from __future__ import annotations

import json
import os
import re
import threading
import queue
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, TYPE_CHECKING, Callable

from fastapi.responses import StreamingResponse

from .model import (
    GENERATION_CONFIG,
    DEFAULT_N_GPU_LAYERS,
)
from .invoker import LLMInvoker
from .prompt_preparer import PromptPreparer
from .response_parser import ResponseParser
from .memory import MemoryManager, MEMORY_MANAGER
from .logger import LOGGER

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .main import ChatRequest


@dataclass
class CallData:
    """Container for information used when calling the model."""

    chat_id: str
    message: str
    global_prompt: str = ""
    call_type: str = "standard_chat"
    options: Dict[str, Any] = None


def _default_global_prompt() -> str:
    prompts = MEMORY_MANAGER.load_global_prompts()
    if prompts:
        return prompts[0]["content"]
    return ""




# --- Background task queue -------------------------------------------------

_task_queue: queue.Queue[tuple[str, Callable[..., None], tuple]] = (
    queue.Queue()
)
_queued_types: set[str] = set()


def _task_worker() -> None:
    while True:
        call_type, func, args = _task_queue.get()
        try:
            func(*args)
        finally:
            _queued_types.discard(call_type)
            _task_queue.task_done()


threading.Thread(target=_task_worker, daemon=True).start()


def enqueue_task(call_type: str, func: Callable[..., None], *args) -> None:
    if call_type in _queued_types:
        return
    _queued_types.add(call_type)
    _task_queue.put((call_type, func, args))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: str, *, trim: bool = False) -> str:
    """Return ``text`` with special tokens removed."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned.strip() if trim else cleaned




# ---------------------------------------------------------------------------
# Core chat handling
# ---------------------------------------------------------------------------


def _parse_goals_from_response(text: str) -> List[Dict[str, Any]]:
    """Attempt to parse valid goal objects from model output."""

    try:

        # Naive direct parse first
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "goals" not in parsed:
            raise ValueError("Top-level object is not a dict with 'goals'")
        goals = parsed["goals"]
        if not isinstance(goals, list):
            raise ValueError("'goals' is not a list")
    except Exception as e:

        # Fallback regex to extract JSON object manually
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in text")

            parsed = json.loads(match.group())
            goals = parsed.get("goals", [])
        except Exception:
            return []

    filtered = []
    for i, g in enumerate(goals):
        desc = g.get("description", "").strip()
        importance = g.get("importance", None)
        if not desc or not isinstance(importance, int):
            continue
        filtered.append(
            {
                "id": str(i + 1),
                "description": desc,
                "importance": importance,
                "status": "in progress",
            }
        )

    return filtered


def _dedupe_new_goals(
    new: List[Dict[str, str]], existing: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    existing_desc = {g.get("description", "") for g in existing}
    return [g for g in new if g.get("description", "") not in existing_desc]


def _maybe_generate_goals(
    chat_id: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
) -> None:
    goals = memory.load_goals(chat_id)
    if not memory.goals_active:
        return
    character = goals.character
    setting = goals.setting

    state = memory.load_goal_state(chat_id)
    state["messages_since_goal_eval"] = (
        state.get("messages_since_goal_eval", 0) + 1
    )

    refresh = GENERATION_CONFIG.get("goal_refresh_rate", 1)
    LOGGER.log(
        "goal_state_check",
        {
            "chat_id": chat_id,
            "current": state["messages_since_goal_eval"],
            "refresh_rate": refresh,
            "generate": state["messages_since_goal_eval"] >= refresh,
        },
    )
    if state["messages_since_goal_eval"] < refresh:
        memory.save_goal_state(chat_id, state)
        return

    goal_limit = GENERATION_CONFIG.get("goal_limit", 3)

    history = memory.load_history(chat_id)
    user_text = "\n".join(m.get("content", "") for m in history)

    system_parts = [p for p in (global_prompt, character, setting) if p]

    goal_prompt = f"""
You are a reasoning assistant.

Given the character profile and scene context, determine whether the character has any goals they would *naturally and strongly* pursue based on their personality, desires, or immediate situation.

If so, generate up to {goal_limit} specific, actionable goals. For each goal, rate how much the character cares about it on a scale from 1 (barely cares) to 10 (deeply invested).

ONLY respond in the following format:
{{
  "goals": [
    {{
      "description": "...",
      "importance": 7
    }},
    ...
  ]
}}

Do not include any explanation, commentary, or other text. If no goals are currently appropriate, return an empty list.
"""

    system_parts.append(textwrap.dedent(goal_prompt).strip())
    system_text = "\n".join(system_parts)

    LOGGER.log(
        "prepared_prompts",
        {
            "call_type": "goal_generation",
            "system_text": system_text,
            "user_text": user_text,
        },
    )

    from .call_templates import goal_generation

    text = goal_generation.generate_goals(
        system_text,
        user_text,
        {**goal_generation.MODEL_LAUNCH_OVERRIDE},
    )

    goals = _parse_goals_from_response(text)
    if not goals:
        memory.save_goal_state(chat_id, state)
        return

    goals = _dedupe_new_goals(goals, state.get("goals", []))
    if not goals:
        memory.save_goal_state(chat_id, state)
        return

    combined = state.get("goals", []) + goals
    state["goals"] = combined[:goal_limit]
    state["messages_since_goal_eval"] = 0
    memory.save_goal_state(chat_id, state)


# ---------------------------------------------------------------------------


def _finalize_chat(
    reply: str,
    call: CallData,
    memory: MemoryManager = MEMORY_MANAGER,
    prompt: str | None = None,
) -> None:
    """Store assistant reply and queue background work."""

    history = memory.load_history(call.chat_id)
    history.append({"role": "assistant", "content": reply})
    memory.save_history(call.chat_id, history)
    enqueue_task(
        "goal_generation",
        _maybe_generate_goals,
        call.chat_id,
        call.global_prompt,
        memory,
    )


def handle_chat(
    call: CallData,
    memory: MemoryManager = MEMORY_MANAGER,
    stream: bool = False,
    *,
    current_chat_id: str | None = None,
    current_prompt: str | None = None,
):
    """Process ``call`` and return a model reply."""

    from .call_templates import standard_chat, logic_check

    call.options = call.options or {"stream": stream}

    if call.call_type == "logic_check":
        processed = logic_check.logic_check(
            call.global_prompt,
            call.message,
            call.options,
        )
    else:
        processed = standard_chat.prepare_and_chat(call)

    if stream:

        def _generate():
            parts: list[str] = []
            for text in processed:
                parts.append(text)
                yield text

            assistant_reply = "".join(parts).strip()
            _finalize_chat(
                assistant_reply,
                call,
                memory,
                prompt=current_prompt,
            )

        return StreamingResponse(_generate(), media_type="text/plain")

    assistant_reply = (
        processed if isinstance(processed, str) else str(processed)
    )
    _finalize_chat(
        assistant_reply,
        call,
        memory,
        prompt=current_prompt,
    )

    return {"detail": assistant_reply}


class ChatRunner:
    """High level interface for processing chat messages."""

    def __init__(
        self,
        memory: MemoryManager = MEMORY_MANAGER,
    ) -> None:
        self.memory = memory
        self.current_chat_id: str | None = None
        self.current_prompt: str | None = None

    def process_user_message(
        self, chat_id: str, message: str, stream: bool = False
    ):
        call = CallData(chat_id=chat_id, message=message, options={"stream": stream})
        result = handle_chat(
            call,
            self.memory,
            stream=stream,
            current_chat_id=self.current_chat_id,
            current_prompt=self.current_prompt,
        )
        self.current_chat_id = call.chat_id
        self.current_prompt = call.global_prompt or self.current_prompt
        return result
