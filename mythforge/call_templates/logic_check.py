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
}




# CallType helpers -----------------------------------------------------------

def logic_check(global_prompt: str, message: str, options: Dict[str, Any]):
    """Send ``message`` through a logic-checking prompt."""

    prepared = PromptPreparer().prepare(global_prompt, message)
    raw = LLMInvoker().invoke(prepared, options)
    parsed = ResponseParser().load(raw).parse()
    if isinstance(parsed, Iterator):
        try:
            first = next(parsed)
        except StopIteration as exc:  # pragma: no cover - best effort
            return str(exc.value)

        def _chain() -> Iterator[str]:
            yield first
            yield from parsed

        return _chain()
    return parsed
