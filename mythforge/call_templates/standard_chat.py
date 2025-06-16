from __future__ import annotations

"""Prompt helpers for standard chat interactions."""

from typing import Iterator, Dict, Any, List

from ..response_parser import ResponseParser
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker
from ..logger import LOGGER
from ..memory import MemoryManager, MEMORY_MANAGER



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
    """Compose the system prompt for standard chat."""

    memory.update_paths(chat_name=chat_name)
    goals = memory.load_goals(chat_name)
    parts: List[str] = [global_prompt]

    if memory.goals_active:
        if goals.character:
            parts.append(goals.character)
        if goals.setting:
            parts.append(goals.setting)
        state = memory.load_goal_state(chat_name)
        active = state.get("goals", [])
        if active:
            goal_lines = ["Active goals:"]
            for item in active:
                if isinstance(item, dict):
                    goal_lines.append(f"- {item.get('description', '')}")
                else:
                    goal_lines.append(f"- {item}")
            parts.append("\n".join(goal_lines))

    return "\n".join(p for p in parts if p)


def standard_chat_prepared_user_text(
    chat_name: str,
    message: str,
    memory: MemoryManager = MEMORY_MANAGER,
) -> str:
    """Combine chat history and the new message."""

    history = memory.load_chat_history(chat_name)
    history_text = "\n".join(m.get("content", "") for m in history)
    if history_text:
        return f"{history_text}\n{message}"
    return message

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

