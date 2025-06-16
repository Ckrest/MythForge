from __future__ import annotations

"""Prompt helpers for logic checking utilities."""

from typing import Any, Dict, Iterator

from ..prompt_preparer import PromptPreparer
from ..response_parser import ResponseParser
from ..invoker import LLMInvoker

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "background": True,
    "n_ctx": 4096,
    "stream": False,
    "max_tokens": 256,
    "verbose": True,
}


# CallType helpers -----------------------------------------------------------

def logic_check(
    global_prompt: str, message: str, options: Dict[str, Any]
) -> Iterator[str]:
    """Send ``message`` through a logic-checking prompt."""

    prepared = PromptPreparer().prepare(global_prompt, message)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)
    return ResponseParser().load(raw).parse()
