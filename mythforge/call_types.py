from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator


@dataclass
class CallHandler:
    """Container for prompt and response handlers."""

    prompt: Callable[[str, str], tuple[str, str]]
    response: Callable[[Any], Any]


# Prompt builders -----------------------------------------------------------


def _fmt(system_text: str, user_text: str, call_type: str) -> tuple[str, str]:
    """Return ``system_text`` and ``user_text`` ignoring ``call_type``."""

    del call_type
    return system_text, user_text


def standard_chat_prompt(system_text: str, user_text: str) -> tuple[str, str]:
    return _fmt(system_text, user_text, "standard_chat")


def helper_prompt(system_text: str, user_text: str) -> tuple[str, str]:
    return _fmt(system_text, user_text, "helper")


def goal_generation_prompt(
    system_text: str, user_text: str
) -> tuple[str, str]:
    return _fmt(system_text, user_text, "goal_generation")


def default_prompt(system_text: str, user_text: str) -> tuple[str, str]:
    return _fmt(system_text, user_text, "default")


# Response handlers --------------------------------------------------------


def standard_chat_response(result: Iterable[dict]) -> Iterator[str]:
    from .call_core import stream_parsed

    return stream_parsed(result)


def helper_response(result: Any) -> str:
    from .call_core import parse_response

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return parse_response(result)


def goal_generation_response(result: Any) -> str:
    from .call_core import parse_response

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return parse_response(result)


def default_response(result: Any) -> Any:
    return result


CALL_HANDLERS: dict[str, CallHandler] = {
    "standard_chat": CallHandler(standard_chat_prompt, standard_chat_response),
    "helper": CallHandler(helper_prompt, helper_response),
    "goal_generation": CallHandler(
        goal_generation_prompt, goal_generation_response
    ),
    "user_message": CallHandler(standard_chat_prompt, standard_chat_response),
    "default": CallHandler(default_prompt, default_response),
}
