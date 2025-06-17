from __future__ import annotations

import json
from typing import Any, Dict

from fastapi.responses import StreamingResponse

from .memory import MemoryManager, MEMORY_MANAGER
from .logger import LOGGER
from .call_templates.generate_goals import evaluate_goals


# ---------------------------------------------------------------------------
# Core chat handling
# ---------------------------------------------------------------------------
def _finalize_chat(
    reply: str,
    chat_name: str,
    message: str,
    global_prompt: str,
    call_type: str,
    options: Dict[str, Any] | None,
    memory: MemoryManager = MEMORY_MANAGER,
    prompt: str | None = None,
) -> None:
    """Finalize the turn and schedule goal evaluation."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "_finalize_chat",
            "chat_name": chat_name,
            "reply": reply,
            "prompt": prompt,
            "message": message,
            "global_prompt": global_prompt,
            "call_type": call_type,
            "options": options,
            "memory_root": memory.root_dir,
        },
    )

    chat_history = memory.load_chat_history(chat_name)
    chat_history.append({"role": "assistant", "content": reply})
    memory.save_chat_history(chat_name, chat_history)
    evaluate_goals(chat_name, global_prompt, memory)


def handle_chat(
    chat_name: str,
    message: str,
    global_prompt: str,
    call_type: str = "standard_chat",
    options: Dict[str, Any] | None = None,
    memory: MemoryManager = MEMORY_MANAGER,
    stream: bool = False,
    *,
    current_chat_name: str | None = None,
    current_global_prompt: str | None = None,
):
    """Orchestrate a full chat cycle and persist the result."""

    LOGGER.log(
        "chat_flow",
        {
            "function": "handle_chat",
            "chat_name": current_chat_name or chat_name,
            "message": message,
            "global_prompt": global_prompt,
            "call_type": call_type,
            "options": options,
            "stream": stream,
            "current_chat_name": current_chat_name,
            "current_global_prompt": current_global_prompt,
            "memory_root": memory.root_dir,
        },
    )

    from .call_templates import standard_chat, logic_goblin_evaluate_goals

    options = options or {"stream": stream}

    call_map = {
        "logic_check": lambda: logic_goblin_evaluate_goals.logic_goblin_evaluate_goals(
            global_prompt,
            message,
            options,
        ),
        "standard_chat": lambda: standard_chat.standard_chat(
            chat_name, message, global_prompt, options
        ),
    }

    handler = call_map.get(call_type, call_map["standard_chat"])
    processed = handler()

    if stream:

        def _generate():
            meta = {"prompt": current_global_prompt or global_prompt}
            yield json.dumps(meta, ensure_ascii=False) + "\n"

            parts: list[str] = []
            for text in processed:
                parts.append(text)
                yield text

            assistant_reply = "".join(parts).strip()
            _finalize_chat(
                assistant_reply,
                chat_name,
                message,
                global_prompt,
                call_type,
                options,
                memory,
                prompt=current_global_prompt,
            )

        return StreamingResponse(_generate(), media_type="text/plain")

    assistant_reply = "".join(list(processed)).strip()
    _finalize_chat(
        assistant_reply,
        chat_name,
        message,
        global_prompt,
        call_type,
        options,
        memory,
        prompt=current_global_prompt,
    )

    return {"detail": assistant_reply}
