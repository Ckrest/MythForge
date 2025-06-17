from __future__ import annotations

"""Prompt helpers for detecting duplicate goals."""

from typing import Any, Dict, Iterator

import json
from ..prompt_preparer import PromptPreparer
from ..response_parser import (
    ResponseParser,
    _parse_duplicates_from_response,
)
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


def logic_goblin_duplicate_goals_prepared_user_text(goals: str) -> str:
    """Compose the user prompt for duplicate goal detection."""

    return "\n".join(
        [
            "Goals:",
            goals,
        ]
    )


def logic_goblin_duplicate_goals_prepared_system_text() -> str:
    """Return the instruction text for duplicate goal detection."""

    return """
You are a logic goblin who detects duplicate or near-duplicate goals.

Your task is to examine a list of character goals and determine if any of them are meaningfully the same—either by intent, action, or outcome.

A goal is a duplicate if:
- It shares the same purpose or outcome.
- It could reasonably be merged with another goal without losing meaning.

Output a JSON object listing any pairs of duplicate goal indices. If none are duplicates, return an empty list.

Only respond in the following format:
{
  "duplicates": [[0, 2], [1, 3]]
}
(Where each pair refers to duplicate goals at those indices in the input list.)

Do not add commentary or explanation.
"""


def logic_goblin_duplicate_goals(
    global_prompt: str,
    message: str,
    options: Dict[str, Any],
    *,
    goals: str = "",
) -> Iterator[str]:
    """Send ``message`` through the duplicate-goal check prompt."""

    user_text = logic_goblin_duplicate_goals_prepared_user_text(goals)
    system_text = logic_goblin_duplicate_goals_prepared_system_text()

    preparer = PromptPreparer()
    prompt_log = preparer.format_for_logging(system_text, user_text)
    LOGGER.log(
        "prepared_prompts",
        {"call_type": "logic_goblin", "prompt": prompt_log},
    )

    prepared = preparer.prepare(system_text, user_text)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)

    parser = ResponseParser().load(raw)
    parsed_iter = parser.parse()
    text = "".join(list(parsed_iter))
    parsed = _parse_duplicates_from_response(text)
    LOGGER.log(
        "chat_flow",
        {"function": "logic_goblin_duplicate_goals", "output": parsed},
    )

    clean_json = json.dumps({"duplicates": parsed}, ensure_ascii=False)
    return iter([clean_json])
