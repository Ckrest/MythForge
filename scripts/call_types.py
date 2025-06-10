"""Utilities defining prompt/response handlers for model call types."""

from __future__ import annotations

from typing import Iterable, Iterator, Any

from . import model_call


def standard_chat_prompt(system_text: str, user_text: str) -> str:
    """Return a prompt for a standard chat call."""

    return model_call.format_for_model(system_text, user_text, "standard_chat")


def standard_chat_response(result: Iterable[dict]) -> Iterator[str]:
    """Yield parsed text for a streaming standard chat call."""

    return model_call.stream_parsed(result)


def helper_prompt(system_text: str, user_text: str) -> str:
    """Return a prompt for a helper call."""

    return model_call.format_for_model(system_text, user_text, "helper")


def helper_response(result: Any) -> str:
    """Return parsed text from a helper call."""

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return model_call.parse_response(result)


def goal_generation_prompt(system_text: str, user_text: str) -> str:
    """Return a prompt for goal generation."""

    return model_call.format_for_model(
        system_text, user_text, "goal_generation"
    )


def goal_generation_response(result: Any) -> str:
    """Return parsed text from a goal generation call."""

    if isinstance(result, Iterable):
        result = next(iter(result), {})
    return model_call.parse_response(result)


def default_prompt(system_text: str, user_text: str) -> str:
    """Fallback prompt builder."""

    return model_call.format_for_model(system_text, user_text, "default")


def default_response(result: Any) -> Any:
    """Return ``result`` unchanged."""

    return result
