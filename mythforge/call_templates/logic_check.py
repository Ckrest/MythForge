from __future__ import annotations

"""Prompt helpers for logic checking utilities."""

from typing import Any, Dict

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

def logic_check(global_prompt: str, message: str, options: Dict[str, Any]):
    """Return parsed logic check result."""

    system, user = PromptPreparer().prepare(global_prompt, message)
    raw = LLMInvoker().invoke(system, options)
    result = ResponseParser().load(raw).parse()
    return result
