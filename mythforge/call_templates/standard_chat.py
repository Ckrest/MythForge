from __future__ import annotations

"""Prompt helpers for standard chat interactions."""

from typing import TYPE_CHECKING

from ..response_parser import ResponseParser
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..call_core import CallData


def prepare_and_chat(call: "CallData"):
    """Return parsed chat output for ``call``."""

    system, user = PromptPreparer().prepare(call.global_prompt, call.message)
    raw = LLMInvoker().invoke(system, call.options)
    parsed = ResponseParser().load(raw).parse()
    return parsed
