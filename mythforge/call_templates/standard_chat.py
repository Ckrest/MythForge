from __future__ import annotations

"""Prompt helpers for standard chat interactions."""

from typing import Iterator, Dict, Any

from ..response_parser import ResponseParser
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker
from ..logger import LOGGER



MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "n_ctx": 4096,
    "stream": True,
}

def standard_chat(call: "CallData"):
    """Apply the standard template, invoke the model and return a reply."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "prepare_and_chat",
            "chat_name": call.chat_name,
            "message": call.message,
            "global_prompt": call.global_prompt,
            "call_type": call.call_type,
            "options": call.options,
        },
    )

    prepared = PromptPreparer().prepare(call.global_prompt, call.message)
    raw = LLMInvoker().invoke(prepared, call.options)
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

