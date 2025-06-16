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

from .invoker import LLMInvoker
from .prompt_preparer import PromptPreparer
from .response_parser import ResponseParser
from .memory import MemoryManager, MEMORY_MANAGER
from .logger import LOGGER


@dataclass
class CallData:
    """Container for information used when calling the model."""

    chat_name: str
    message: str
    global_prompt: str = ""
    call_type: str = "standard_chat"
    options: Dict[str, Any] = None




# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: str, *, trim: bool = False) -> str:
    """Strip unwanted tokens and optionally trim whitespace."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned.strip() if trim else cleaned


# ---------------------------------------------------------------------------
# Core chat handling
# ---------------------------------------------------------------------------


def _maybe_generate_goals(
    chat_name: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
) -> None:
    """Generate new goals if refresh interval has been met."""
    LOGGER.log(
        "chat_flow",
        {
            "function": "_maybe_generate_goals",
            "chat_name": chat_name,
            "global_prompt": global_prompt,
            "memory_root": memory.root_dir,
        },
    )
    goals = memory.load_goals(chat_name)
    if not memory.goals_active:
        return
    character = goals.character
    setting = goals.setting

    state = memory.load_goal_state(chat_name)
    state["messages_since_goal_eval"] = (
        state.get("messages_since_goal_eval", 0) + 1
    )

    refresh = GENERATION_CONFIG.get("goal_refresh_rate", 1)
    LOGGER.log(
        "goal_state_check",
        {
            "chat_name": chat_name,
            "current": state["messages_since_goal_eval"],
            "refresh_rate": refresh,
            "generate": state["messages_since_goal_eval"] >= refresh,
        },
    )
    if state["messages_since_goal_eval"] < refresh:
        memory.save_goal_state(chat_name, state)
        return

    goal_limit = GENERATION_CONFIG.get("goal_limit", 3)

    history = memory.load_history(chat_name)
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


    combined = state.get("goals", []) + goals
    state["goals"] = combined[:goal_limit]
    state["messages_since_goal_eval"] = 0
    memory.save_goal_state(chat_name, state)


# ---------------------------------------------------------------------------


def _finalize_chat(
    reply: str,
    call: CallData,
    memory: MemoryManager = MEMORY_MANAGER,
    prompt: str | None = None,
) -> None:
    """Finalize the turn and schedule goal evaluation."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "_finalize_chat",
            "chat_name": call.chat_name,
            "reply": reply,
            "prompt": prompt,
            "message": call.message,
            "global_prompt": call.global_prompt,
            "call_type": call.call_type,
            "options": call.options,
            "memory_root": memory.root_dir,
        },
    )

    history = memory.load_history(call.chat_name)
    history.append({"role": "assistant", "content": reply})
    memory.save_history(call.chat_name, history)
    enqueue_task(
        "goal_generation",
        _maybe_generate_goals,
        call.chat_name,
        call.global_prompt,
        memory,
    )


def handle_chat(
    call: CallData,
    memory: MemoryManager = MEMORY_MANAGER,
    stream: bool = False,
    *,
    current_chat_name: str | None = None,
    current_prompt: str | None = None,
):
    """Orchestrate a full chat cycle and persist the result."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "handle_chat",
            "chat_name": current_chat_name or call.chat_name,
            "message": call.message,
            "global_prompt": call.global_prompt,
            "call_type": call.call_type,
            "options": call.options,
            "stream": stream,
            "current_chat_name": current_chat_name,
            "current_prompt": current_prompt,
            "memory_root": memory.root_dir,
        },
    )

    memory.update_paths(
        chat_name=current_chat_name or call.chat_name,
        prompt_name=current_prompt or call.global_prompt,
    )

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


