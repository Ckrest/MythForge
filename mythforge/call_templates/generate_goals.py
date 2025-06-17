from __future__ import annotations

import textwrap
import json
from typing import Any, Dict, Iterator, List

from ..response_parser import (
    ResponseParser,
    _parse_goals_from_response,
    _parse_duplicates_from_response,
)
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker
from ..model import GENERATION_CONFIG
from ..memory import MemoryManager, MEMORY_MANAGER
from ..logger import LOGGER
from ..background import schedule_task
from .logic_goblin_duplicate_goals import logic_goblin_duplicate_goals

# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "n_gpu_layers": 0,
    "stream": False,
    "background": True,
    "verbose": True,
}


def generate_goals(
    global_prompt: str, message: str, options: Dict[str, Any]
) -> Iterator[str]:
    """Run the goal-generation template and return parsed output."""

    prepared = PromptPreparer().prepare(global_prompt, message)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)
    return ResponseParser().load(raw).parse()


def clean_text(text: str, *, trim: bool = False) -> str:
    """Strip unwanted tokens and optionally trim whitespace."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned.strip() if trim else cleaned


def logic_goblin_duplicate_goals_call(goals: List[Any]) -> List[List[int]]:
    """Invoke the duplicate-goal goblin on ``goals`` and return duplicates."""

    goals_json = json.dumps(goals, ensure_ascii=False, indent=2)
    try:
        result_iter = logic_goblin_duplicate_goals(
            "",
            "",
            {**MODEL_LAUNCH_OVERRIDE},
            goals=goals_json,
        )
        result_text = "".join(list(result_iter))
        LOGGER.log(
            "chat_flow",
            {
                "function": "logic_goblin_duplicate_goals_call",
                "output": result_text,
            },
        )
        duplicates = _parse_duplicates_from_response(result_text)
        return duplicates
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.log_error(exc)
        return []


def _evaluate_goals(
    chat_name: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
) -> None:
    """Generate new goals when below the configured limit."""
    LOGGER.log(
        "chat_flow",
        {
            "function": "evaluate_goals",
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
    goal_limit = GENERATION_CONFIG.get("goal_limit", 3)
    if len(state.get("goals", [])) >= goal_limit:
        return

    chat_history = memory.load_chat_history(chat_name)
    user_text = "\n".join(m.get("content", "") for m in chat_history)

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

    preparer = PromptPreparer()
    prompt_log = preparer.format_for_logging(system_text, user_text)
    LOGGER.log(
        "prepared_prompts",
        {"call_type": "goal_generation", "prompt": prompt_log},
    )

    try:
        text = generate_goals(
            system_text,
            user_text,
            {**MODEL_LAUNCH_OVERRIDE},
        )
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.log_error(exc)
        state["error"] = str(exc)
        memory.save_goal_state(chat_name, state)
        return

    parsed = ResponseParser().load(text).parse()
    if isinstance(parsed, Iterator):
        parsed = "".join(parsed)
    cleaned = clean_text(str(parsed), trim=True)
    new_goals = _parse_goals_from_response(cleaned)

    state.pop("error", None)
    previous = list(state.get("goals", []))
    combined = state.get("goals", []) + new_goals

    if new_goals and state.get("goals"):
        duplicates = logic_goblin_duplicate_goals_call(combined)
        if duplicates:
            to_remove = {max(a, b) for a, b in duplicates}
            combined = [g for i, g in enumerate(combined) if i not in to_remove]

    state["goals"] = combined[:goal_limit]
    memory.save_goal_state(chat_name, state)

    for goal in new_goals:
        desc = goal.get("description", str(goal))
        memory.add_debug_message(f"new goal: {desc}")

    removed = combined[goal_limit:]
    for goal in removed:
        desc = goal.get("description", str(goal))
        memory.add_debug_message(f"goal removed: {desc}")


def evaluate_goals(
    chat_name: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
    *,
    background: bool = True,
) -> None:
    """Schedule or run goal evaluation."""

    if background:
        schedule_task(_evaluate_goals, chat_name, global_prompt, memory)
        return

    _evaluate_goals(chat_name, global_prompt, memory)
