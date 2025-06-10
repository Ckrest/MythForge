"""Utilities for formatting prompts and calling the language model."""

from __future__ import annotations

from typing import Dict, Iterable, List, TYPE_CHECKING

import json
from fastapi.responses import StreamingResponse

from . import model_launch, model_response

if TYPE_CHECKING:
    from .MythForgeServer import ChatRequest


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


def chat_stream(req: "ChatRequest"):
    """Stream a reply from the model and store it in chat history."""

    from .MythForgeServer import ensure_chat_dir, load_item, save_item

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})

    chunks = call_tagged(req.global_prompt or "", req.message, stream=True)
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

    from .MythForgeServer import ensure_chat_dir, load_item, save_item

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})
    output = call_tagged(req.global_prompt or "", req.message, stream=False)
    if isinstance(output, Iterable):
        output = next(iter(output), {})
    assistant_reply = model_response.parse_response(output)
    history.append({"role": "assistant", "content": assistant_reply})
    save_item("chat_history", req.chat_id, data=history)
    return {"detail": assistant_reply}
