from __future__ import annotations

"""Prompt helpers for standard chat interactions."""

from typing import Iterator, Dict, Any

from ..response_parser import ResponseParser
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker
from ..logger import LOGGER
from ..memory import MEMORY_MANAGER, MemoryManager


MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "n_ctx": 4096,
    "stream": True,
    "verbose": True,
}


def standard_chat_prepared_system_text(
    chat_name: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
) -> str:
    """Return system prompt with optional goal context."""

    memory.update_paths(chat_name=chat_name)
    parts = [global_prompt]
    if memory.goals_active:
        goals = memory.load_goals(chat_name)
        if goals.character:
            parts.append(goals.character)
        if goals.setting:
            parts.append(goals.setting)
        state = memory.load_goal_state(chat_name)
        active = state.get("goals", [])
        for g in active:
            desc = g.get("description", "")
            if desc:
                parts.append(desc)
    return "\n".join(p for p in parts if p)


def standard_chat_prepared_user_text(
    chat_name: str, message: str, memory: MemoryManager = MEMORY_MANAGER
) -> str:
    """Combine chat history with the current ``message``."""
    memory.update_paths(chat_name=chat_name)
    history = memory.load_chat_history(chat_name)
    parts = [m.get("content", "") for m in history]
    parts.append(message)
    return "\n".join(parts)


def standard_chat(
    chat_name: str,
    message: str,
    global_prompt: str,
    options: Dict[str, Any],
) -> Iterator[str]:
    """Apply the standard template, invoke the model and return a reply."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "prepare_and_chat",
            "chat_name": chat_name,
            "message": message,
            "global_prompt": global_prompt,
            "call_type": "standard_chat",
            "options": options,
        },
    )

    system_text = standard_chat_prepared_system_text(chat_name, global_prompt)
    user_text = standard_chat_prepared_user_text(chat_name, message)
    prepared = PromptPreparer().prepare(system_text, user_text)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)
    return ResponseParser().load(raw).parse()
