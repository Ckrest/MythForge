"""Utilities for formatting prompts and calling the language model."""

from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING

import json
from fastapi.responses import StreamingResponse


from . import model_response

if TYPE_CHECKING:
    from .MythForgeServer import ChatRequest


def format_for_model(system_text: str, user_text: str) -> str:
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


def user_model_call(system_text: str, user_text: str) -> str:
    """Return a prompt for a streaming model call."""

    return format_for_model(system_text, user_text)


def helper_model_call(system_text: str, user_text: str) -> str:
    """Return a prompt for a standard model call."""

    return format_for_model(system_text, user_text)


def chat_stream(req: "ChatRequest"):
    """Stream a reply from the model and store it in chat history."""

    from .MythForgeServer import (
        ensure_chat_dir,
        load_item,
        save_item,
        make_user_model_call,
    )

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})

    chunks = make_user_model_call(req.global_prompt or "", req.message)
    parts: List[str] = []

    def generate():
        meta = {"prompt": req.global_prompt or ""}
        yield json.dumps(meta) + "\n"
        for text in model_response.stream_parsed(chunks):
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
        make_helper_model_call,
    )

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})
    output = make_helper_model_call(req.global_prompt or "", req.message)
    if isinstance(output, Iterable):
        output = next(iter(output), {})
    assistant_reply = model_response.parse_response(output)
    history.append({"role": "assistant", "content": assistant_reply})
    save_item("chat_history", req.chat_id, data=history)
    return {"detail": assistant_reply}
