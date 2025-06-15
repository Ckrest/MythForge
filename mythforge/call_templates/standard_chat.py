from __future__ import annotations

"""Prompt helpers for standard chat interactions."""

from typing import TYPE_CHECKING

from typing import Iterator

from ..response_parser import ResponseParser
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..call_core import CallData


def prepare_and_chat(call: "CallData"):
    """Return parsed chat output for ``call``."""

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


# Alias used by ``main.py`` during startup.
prep_standard_chat = prepare_and_chat
