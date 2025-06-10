"""Utilities for formatting prompts and calling the language model."""

from __future__ import annotations

from typing import Iterable, Iterator, List, TYPE_CHECKING

import json
from fastapi.responses import StreamingResponse


if TYPE_CHECKING:
    from .MythForgeServer import ChatRequest


def clean_text(text: str) -> str:
    """Return ``text`` with tokens removed while preserving spaces."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned


def parse_response(output: dict) -> str:
    """Extract and clean the text portion from a model call result."""

    text = output.get("choices", [{}])[0].get("text", "")
    return clean_text(text)


def stream_parsed(chunks: Iterable[dict]) -> Iterator[str]:
    """Yield cleaned text from a streaming model call."""

    for chunk in chunks:
        yield clean_text(chunk.get("choices", [{}])[0].get("text", ""))


def format_for_model(system_text: str, user_text: str, call_type: str) -> str:
    """Return ``system_text`` and ``user_text`` formatted for the model."""

    return (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_text}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def model_call(system_text: str, user_text: str, call_type: str) -> str:
    """Return a prompt for a model call based on ``call_type``."""

    return format_for_model(system_text, user_text, call_type)


def chat_stream(req: "ChatRequest"):
    """Stream a reply from the model and store it in chat history."""

    from .MythForgeServer import (
        ensure_chat_dir,
        load_item,
        save_item,
        make_model_call,
    )

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})

    chunks = make_model_call(
        req.global_prompt or "",
        req.message,
        "standard_chat",
    )
    parts: List[str] = []

    def generate():
        meta = {"prompt": req.global_prompt or ""}
        yield json.dumps(meta) + "\n"
        for text in stream_parsed(chunks):
            parts.append(text)
            yield text
        assistant_reply = "".join(parts).strip()
        history.append({"role": "assistant", "content": assistant_reply})
        save_item("chat_history", req.chat_id, data=history)

    return StreamingResponse(generate(), media_type="text/plain")


def chat(req: "ChatRequest"):
    """Return a standard model reply and store it in chat history."""

    from .MythForgeServer import (
        ensure_chat_dir,
        load_item,
        save_item,
        make_model_call,
    )

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})
    output = make_model_call(
        req.global_prompt or "",
        req.message,
        "helper",
    )
    if isinstance(output, Iterable):
        output = next(iter(output), {})
    assistant_reply = parse_response(output)
    history.append({"role": "assistant", "content": assistant_reply})
    save_item("chat_history", req.chat_id, data=history)
    return {"detail": assistant_reply}
