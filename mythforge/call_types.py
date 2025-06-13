from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, TYPE_CHECKING

from .call_templates import (
    standard_chat,
    helper,
    goal_generation,
    default as default_call,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .main import ChatRequest


@dataclass
class CallHandler:
    """Container for prompt and response handlers."""

    prepare: Callable[["ChatRequest", list], tuple[str, str]]
    prompt: Callable[[str, str], tuple[str, str]]
    response: Callable[[Any], Any]


CALL_HANDLERS: dict[str, CallHandler] = {
    "standard_chat": CallHandler(
        standard_chat.prepare,
        standard_chat.prompt,
        standard_chat.response,
    ),
    "user_message": CallHandler(
        standard_chat.prepare,
        standard_chat.prompt,
        standard_chat.response,
    ),
    "helper": CallHandler(helper.prepare, helper.prompt, helper.response),
    "goal_generation": CallHandler(
        goal_generation.prepare,
        goal_generation.prompt,
        goal_generation.response,
    ),
    "default": CallHandler(
        default_call.prepare,
        default_call.prompt,
        default_call.response,
    ),
}
