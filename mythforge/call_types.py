from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, TYPE_CHECKING

from .call_templates import standard_chat, goal_generation, logic_check

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .call_core import CallData


@dataclass
class CallHandler:
    """Container for prompt and response handlers."""

    prepare: Callable[["CallData"], tuple[str, str]]
    prompt: Callable[[str, str], tuple[str, str]]
    response: Callable[[Any], Any]


CALL_HANDLERS: dict[str, CallHandler] = {
    "standard_chat": CallHandler(
        standard_chat.prepare,
        standard_chat.prompt,
        standard_chat.response,
    ),
    "goal_generation": CallHandler(
        goal_generation.prepare,
        goal_generation.prompt,
        goal_generation.response,
    ),
    "logic_check": CallHandler(
        logic_check.prepare,
        logic_check.prompt,
        logic_check.response,
    ),
}
