from __future__ import annotations

"""Prompt helpers for logic checking utilities."""

from typing import Any, Dict

from ..logger import LOGGER
from ..prompt_preparer import PromptPreparer
from ..response_parser import ResponseParser
from ..invoker import LLMInvoker

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "background": True,
    "n_ctx": 4096,
    "stream": False,
    "max_tokens": 256,
}




# CallType helpers -----------------------------------------------------------


def prepare(call: "CallData") -> tuple[str, str]:
    """Return prompts for ``call`` without modifications."""
    system_text = call.global_prompt
    user_text = call.message
    LOGGER.log(
        "prepared_prompts",
        {
            "call_type": call.call_type,
            "system_text": system_text,
            "user_text": user_text,
        },
    )
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return ``system_text`` and ``user_text`` unchanged."""

    return system_text, user_text


def response(result: Any) -> str:
    """Return a single parsed model response."""

    return ResponseParser().load(result).parse()
