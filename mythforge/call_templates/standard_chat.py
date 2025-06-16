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
    "verbose": True,
}

def standard_chat(
    chat_name: str,
    message: str,
    global_prompt: str,
    options: Dict[str, Any],
) -> Iterator[str]:
    """Apply the standard template, invoke the model and return a reply."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "prepare_and_chat",
            "chat_name": chat_name,
            "message": message,
            "global_prompt": global_prompt,
            "call_type": "standard_chat",
            "options": options,
        },
    )

    prepared = PromptPreparer().prepare(global_prompt, message)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)
    return ResponseParser().load(raw).parse()

