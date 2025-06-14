from __future__ import annotations

import json
import os
import re
import threading
import queue
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, TYPE_CHECKING, Callable

from fastapi.responses import StreamingResponse

from .model import (
    GENERATION_CONFIG,
    DEFAULT_N_GPU_LAYERS,
    call_llm,
)
from .utils import (
    CHATS_DIR,
    chat_file,
    load_global_prompts,
    log_prepared_prompts,
)
from .memory import ChatHistoryService, MemoryManager, MEMORY_MANAGER

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .main import ChatRequest


@dataclass
class CallData:
    """Container for information used when calling the model."""

    chat_id: str
    message: str
    global_prompt: str = ""
    call_type: str = "standard_chat"


def _default_global_prompt() -> str:
    prompts = load_global_prompts()
    if prompts:
        return prompts[0]["content"]
    return ""


def build_call(req: "ChatRequest") -> CallData:
    """Return :class:`CallData` populated from ``req`` and defaults."""

    return CallData(
        chat_id=req.chat_id,
        message=req.message,
    )


# --- Background task queue -------------------------------------------------

_task_queue: queue.Queue[tuple[str, Callable[..., None], tuple]] = (
    queue.Queue()
)
_queued_types: set[str] = set()


def _task_worker() -> None:
    while True:
        call_type, func, args = _task_queue.get()
        try:
            func(*args)
        finally:
            _queued_types.discard(call_type)
            _task_queue.task_done()


threading.Thread(target=_task_worker, daemon=True).start()


def enqueue_task(call_type: str, func: Callable[..., None], *args) -> None:
    if call_type in _queued_types:
        return
    _queued_types.add(call_type)
    _task_queue.put((call_type, func, args))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: str, *, trim: bool = False) -> str:
    """Return ``text`` with special tokens removed."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned.strip() if trim else cleaned


def parse_response(output: Any) -> str:
    """Return ``output`` as plain text."""

    if isinstance(output, dict) and "text" in output:
        return str(output["text"])
    return str(output)


def stream_parsed(chunks: Iterable[Any]) -> Iterator[str]:
    """Yield plain text from streaming model output."""

    for chunk in chunks:
        if isinstance(chunk, dict) and "text" in chunk:
            yield str(chunk["text"])
        else:
            yield str(chunk)


def format_for_model(system_text: str, user_text: str) -> str:
    """Format system and user text using ChatML-style tokens."""

    system_clean = system_text.strip().replace("\n", "/").replace('"', '\\"')
    user_clean = user_text.strip().replace("\n", "/").replace('"', '\\"')

    return (
        f"<|im_start|>system\\{system_clean}<|im_end|>\\"
        f"<|im_start|>user\\{user_clean}<|im_end|>\\"
        f"<|im_start|>assistant\\/"
    )


# ---------------------------------------------------------------------------
# Core chat handling
# ---------------------------------------------------------------------------


def _state_path(chat_id: str) -> str:
    return os.path.join(CHATS_DIR, chat_id, "goal_state.json")


def _load_goal_state(chat_id: str) -> Dict[str, Any]:
    path = _state_path(chat_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"goals": [], "completed_goals": [], "messages_since_goal_eval": 0}


def _save_goal_state(chat_id: str, state: Dict[str, Any]) -> None:
    path = _state_path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _parse_goals_from_response(text: str) -> List[Dict[str, Any]]:
    """Attempt to parse valid goal objects from model output."""

    try:

        # Naive direct parse first
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "goals" not in parsed:
            raise ValueError("Top-level object is not a dict with 'goals'")
        goals = parsed["goals"]
        if not isinstance(goals, list):
            raise ValueError("'goals' is not a list")
    except Exception as e:

        # Fallback regex to extract JSON object manually
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in text")

            parsed = json.loads(match.group())
            goals = parsed.get("goals", [])
        except Exception:
            return []

    filtered = []
    for i, g in enumerate(goals):
        desc = g.get("description", "").strip()
        importance = g.get("importance", None)
        if not desc or not isinstance(importance, int):
            continue
        filtered.append(
            {
                "id": str(i + 1),
                "description": desc,
                "importance": importance,
                "status": "in progress",
            }
        )

    return filtered


def _dedupe_new_goals(
    new: List[Dict[str, str]], existing: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    existing_desc = {g.get("description", "") for g in existing}
    return [g for g in new if g.get("description", "") not in existing_desc]


def _maybe_generate_goals(
    chat_id: str,
    global_prompt: str,
    history_service: ChatHistoryService,
    memory: MemoryManager = MEMORY_MANAGER,
) -> None:
    from .call_types import CALL_HANDLERS

    goals = memory.load_goals(chat_id)
    if not memory.goals_active:
        return
    character = goals.character
    setting = goals.setting

    state = _load_goal_state(chat_id)
    state["messages_since_goal_eval"] = (
        state.get("messages_since_goal_eval", 0) + 1
    )

    refresh = GENERATION_CONFIG.get("goal_refresh_rate", 1)
    if state["messages_since_goal_eval"] < refresh:
        _save_goal_state(chat_id, state)
        return

    goal_limit = GENERATION_CONFIG.get("goal_limit", 3)

    history = history_service.load_history(chat_id)
    user_text = "\n".join(m.get("content", "") for m in history)

    system_parts = [p for p in (global_prompt, character, setting) if p]

    goal_prompt = f"""
You are a reasoning assistant.

Given the character profile and scene context, determine whether the character has any goals they would *naturally and strongly* pursue based on their personality, desires, or immediate situation.

If so, generate up to {goal_limit} specific, actionable goals. For each goal, rate how much the character cares about it on a scale from 1 (barely cares) to 10 (deeply invested).

ONLY respond in the following format:
{{
  "goals": [
    {{
      "description": "...",
      "importance": 7
    }},
    ...
  ]
}}

Do not include any explanation, commentary, or other text. If no goals are currently appropriate, return an empty list.
"""

    system_parts.append(textwrap.dedent(goal_prompt).strip())
    system_text = "\n".join(system_parts)

    log_prepared_prompts("goal_generation", system_text, user_text)

    from .call_types import CALL_HANDLERS
    from .call_templates import goal_generation

    handler = CALL_HANDLERS["goal_generation"]
    system_prompt, user_prompt = handler.prompt(system_text, user_text)
    raw = call_llm(
        system_prompt,
        user_prompt,
        **goal_generation.MODEL_LAUNCH_OVERRIDE,
    )
    text = handler.response(raw)

    goals = _parse_goals_from_response(text)
    if not goals:
        _save_goal_state(chat_id, state)
        return

    goals = _dedupe_new_goals(goals, state.get("goals", []))
    if not goals:
        _save_goal_state(chat_id, state)
        return

    combined = state.get("goals", []) + goals
    state["goals"] = combined[:goal_limit]
    state["messages_since_goal_eval"] = 0
    _save_goal_state(chat_id, state)


# ---------------------------------------------------------------------------


def _finalize_chat(
    reply: str,
    call: CallData,
    history_service: ChatHistoryService,
    memory: MemoryManager = MEMORY_MANAGER,
    prompt: str | None = None,
) -> None:
    """Store assistant reply and queue background work."""

    history_service.append_message(call.chat_id, "assistant", reply)
    enqueue_task(
        "goal_generation",
        _maybe_generate_goals,
        call.chat_id,
        call.global_prompt,
        history_service,
        memory,
    )


def handle_chat(
    call: CallData,
    history_service: ChatHistoryService,
    memory: MemoryManager = MEMORY_MANAGER,
    stream: bool = False,
    *,
    current_chat_id: str | None = None,
    current_prompt: str | None = None,
):
    """Process ``call`` and return a model reply."""

    from .call_types import CALL_HANDLERS

    handler = CALL_HANDLERS.get(call.call_type, CALL_HANDLERS["standard_chat"])

    system_text, user_text = handler.prepare(call)

    if call.chat_id != current_chat_id or system_text != (
        current_prompt or ""
    ):
        current_chat_id = call.chat_id
        current_prompt = system_text

    system_prompt, user_prompt = handler.prompt(system_text, user_text)
    if call.call_type == "standard_chat":
        from .call_templates import standard_chat

        raw = standard_chat.send_prompt(
            system_prompt,
            user_prompt,
            stream=stream,
        )
    elif call.call_type == "logic_check":
        from .call_templates import logic_check

        raw = logic_check.send_prompt(system_prompt, user_prompt)
    else:
        raw = call_llm(
            system_prompt,
            user_prompt,
            stream=stream,
        )
    processed = handler.response(raw)

    if stream:

        def _generate():
            meta = json.dumps(
                {"system_prompt": system_prompt, "prompt": user_prompt},
                ensure_ascii=False,
            )
            yield meta + "\n"

            parts: list[str] = []
            for text in processed:
                print(text, end="", flush=True)
                parts.append(text)
                yield text

            assistant_reply = "".join(parts).strip()
            _finalize_chat(
                assistant_reply,
                call,
                history_service,
                memory,
                prompt=current_prompt,
            )

        return StreamingResponse(_generate(), media_type="text/plain")

    assistant_reply = (
        processed if isinstance(processed, str) else str(processed)
    )
    _finalize_chat(
        assistant_reply,
        call,
        history_service,
        memory,
        prompt=current_prompt,
    )

    return {"detail": assistant_reply}


class ChatRunner:
    """High level interface for processing chat messages."""

    def __init__(
        self,
        history_service: ChatHistoryService,
        memory: MemoryManager = MEMORY_MANAGER,
    ) -> None:
        self.history_service = history_service
        self.memory = memory
        self.current_chat_id: str | None = None
        self.current_prompt: str | None = None

    def process_user_message(
        self, chat_id: str, message: str, stream: bool = False
    ):
        call = CallData(chat_id=chat_id, message=message)
        result = handle_chat(
            call,
            self.history_service,
            self.memory,
            stream=stream,
            current_chat_id=self.current_chat_id,
            current_prompt=self.current_prompt,
        )
        self.current_chat_id = call.chat_id
        self.current_prompt = call.global_prompt or self.current_prompt
        return result
