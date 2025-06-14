from __future__ import annotations

from typing import Any, Iterable, Dict

from ..call_core import CallData, _default_global_prompt
from .. import memory
from ..utils import log_prepared_prompts

# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "n_gpu_layers": 0,
    "stream": False,
    "background": True,
}


def prepare_system_text(call: CallData) -> str:
    """Return the system prompt for ``call``."""

    if not call.global_prompt:
        call.global_prompt = (
            memory.MEMORY.global_prompt or _default_global_prompt()
        )
    return call.global_prompt


def prepare_user_text(call: CallData) -> str:
    """Return the user prompt for ``call``."""

    return call.message


def prepare(call: CallData) -> tuple[str, str]:
    """Return prompts for goal generation calls."""
    system_text = prepare_system_text(call)
    user_text = prepare_user_text(call)
    log_prepared_prompts(call.call_type, system_text, user_text)
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return system and user prompt text."""

    return system_text, user_text


def response(result: Any) -> str:
    """Return a single parsed model response."""

    from ..call_core import parse_response

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return parse_response(result)
