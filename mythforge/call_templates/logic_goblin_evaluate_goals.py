from __future__ import annotations

"""Prompt helpers for goal evaluation utilities."""

from typing import Any, Dict, Iterator

from ..prompt_preparer import PromptPreparer
from ..response_parser import ResponseParser
from ..invoker import LLMInvoker
from ..logger import LOGGER

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "background": True,
    "n_ctx": 4096,
    "stream": False,
    "max_tokens": 256,
    "verbose": True,
}


# CallType helpers -----------------------------------------------------------

def logic_goblin_evaluate_goals_prepared_user_text(
    global_prompt: str,
    character: str,
    setting: str,
    history: str,
    message: str,
) -> str:
    """Build user prompt text for goal evaluation."""

    parts = [global_prompt, character, setting, history, message]
    return "\n".join(p for p in parts if p)


def logic_goblin_evaluate_goals_prepared_system_text(active_goals: str) -> str:
    """Return system prompt text consisting of active goals."""

    return active_goals


def logic_goblin_evaluate_goals(
    global_prompt: str,
    message: str,
    options: Dict[str, Any],
    *,
    character: str = "",
    setting: str = "",
    history: str = "",
    active_goals: str = "",
) -> Iterator[str]:
    """Send ``message`` through the goal evaluation prompt."""

    user_text = logic_goblin_evaluate_goals_prepared_user_text(
        global_prompt, character, setting, history, message
    )
    system_text = logic_goblin_evaluate_goals_prepared_system_text(active_goals)

    preparer = PromptPreparer()
    prompt_log = preparer.format_for_logging(system_text, user_text)
    LOGGER.log(
        "prepared_prompts",
        {"call_type": "logic_goblin", "prompt": prompt_log},
    )

    prepared = preparer.prepare(system_text, user_text)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)
    return ResponseParser().load(raw).parse()
