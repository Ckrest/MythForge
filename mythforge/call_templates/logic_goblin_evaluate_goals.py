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
    character: str,
    setting: str,
    active_goals: str,
) -> str:
    """Compose the user prompt for goal evaluation."""

    return "\n".join(
        [
            "[CHARACTER CONTEXT]",
            character,
            "",
            "[SETTING]",
            setting,
            "",
            "Goals:",
            active_goals,
            "",
            'Respond by returning the same JSON object with each goal\'s "status" field updated.',
            "",
            "Do not add any explanation or text—return JSON only.",
        ]
    )


def logic_goblin_evaluate_goals_prepared_system_text() -> str:
    """Return the instruction text for logic goal evaluation."""

    return f"""
You are a logic goblin who evaluates character goals with brutal honesty.

Given the character's current situation and goals, your job is to:
- Determine for each goal whether it is:
  - "completed"
  - "in progress"
  - OR "abandoned" (only if it is clearly irrelevant, impossible, or contradicted by recent events).

Only abandon a goal if it is no longer feasible or logical to continue pursuing it. This should be rare—don't be lazy.

Respond by returning the same JSON object with each goal's "status" field updated.

Do not add any explanation or text—return JSON only.
"""


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
        character,
        setting,
        active_goals,
    )
    system_text = logic_goblin_evaluate_goals_prepared_system_text()

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
