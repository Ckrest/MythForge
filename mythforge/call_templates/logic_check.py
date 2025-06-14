from __future__ import annotations

"""Prompt helpers for logic checking utilities."""

from typing import Any, Dict

from ..model import _select_model_path
from ..call_core import format_for_model, parse_response
from ..utils import log_server_call, log_prepared_prompts

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "background": True,
    "n_ctx": 4096,
    "stream": False,
    "max_tokens": 256,
}


def send_prompt(system_text: str, user_text: str) -> dict[str, str]:
    """Return raw model output for ``system_text`` and ``user_text``."""
    from llama_cpp import Llama

    prompt = format_for_model(system_text, user_text)
    log_server_call(prompt)

    llm = Llama(
        model_path=_select_model_path(background=True),
        n_ctx=MODEL_LAUNCH_OVERRIDE["n_ctx"],
    )

    result = llm(prompt, max_tokens=MODEL_LAUNCH_OVERRIDE["max_tokens"])
    text = ""
    if isinstance(result, dict):
        choices = result.get("choices", [{}])
        if choices:
            text = str(choices[0].get("text", ""))
    return {"text": text}


def run_logic_check(system_text: str, user_text: str) -> str:
    """Return a parsed model reply for ``system_text`` and ``user_text``."""

    raw = send_prompt(system_text, user_text)
    return parse_response(raw)


# CallType helpers -----------------------------------------------------------


def prepare(call: "CallData") -> tuple[str, str]:
    """Return prompts for ``call`` without modifications."""
    system_text = call.global_prompt
    user_text = call.message
    log_prepared_prompts(call.call_type, system_text, user_text)
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return ``system_text`` and ``user_text`` unchanged."""

    return system_text, user_text


def response(result: Any) -> str:
    """Return a single parsed model response."""

    return parse_response(result)
