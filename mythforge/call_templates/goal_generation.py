from __future__ import annotations

from typing import Any, Iterable, Dict

from ..call_core import CallData, _default_global_prompt
from .. import memory
from ..memory import MEMORY_MANAGER
from ..logger import LOGGER
from ..response_parser import ResponseParser

# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "n_gpu_layers": 0,
    "stream": False,
    "background": True,
}


def prepare(call: CallData) -> tuple[str, str]:
    """Return prompts for goal generation calls."""
    if not call.global_prompt:
        call.global_prompt = (
            memory.MEMORY.global_prompt or _default_global_prompt()
        )
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
    """Return system and user prompt text."""

    return system_text, user_text


def response(result: Any) -> str:
    """Return a single parsed model response."""
    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return ResponseParser().load(result).parse()
