"""Utilities for formatting prompts and calling the language model."""

from __future__ import annotations

from typing import Dict, Iterable, List

from . import model_launch


def build_prompt(
    messages: List[Dict[str, str]], global_prompt: str | None = None
) -> str:
    """Return a prompt string for the model."""

    parts: List[str] = []
    if global_prompt:
        parts.append(global_prompt.strip())
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def call_model(
    messages: List[Dict[str, str]],
    *,
    global_prompt: str | None = None,
    stream: bool | None = None,
) -> Iterable[Dict[str, object]]:
    """Build a prompt and send it to the model via ``call_llm``."""

    prompt = build_prompt(messages, global_prompt)
    kwargs: Dict[str, object] = model_launch.GENERATION_CONFIG.copy()
    if stream is not None:
        kwargs["stream"] = stream
    else:
        kwargs["stream"] = model_launch.MODEL_SETTINGS.get("stream", False)
    return model_launch.call_llm(prompt, **kwargs)


def call_tagged(
    system_text: str, user_text: str, *, stream: bool | None = None
) -> Iterable[Dict[str, object]]:
    """Call the model using the header-tagged prompt format."""

    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_text}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    kwargs: Dict[str, object] = model_launch.GENERATION_CONFIG.copy()
    if stream is not None:
        kwargs["stream"] = stream
    else:
        kwargs["stream"] = model_launch.MODEL_SETTINGS.get("stream", False)
    return model_launch.call_llm(prompt, **kwargs)
