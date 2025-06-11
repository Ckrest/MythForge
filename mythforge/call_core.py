from __future__ import annotations

import json
import os
import re
import threading
import queue
from typing import Any, Dict, Iterable, Iterator, List, TYPE_CHECKING, Callable

from fastapi.responses import StreamingResponse

from .model import GENERATION_CONFIG, call_llm
from .utils import (
    CHATS_DIR,
    chat_file,
    ensure_chat_dir,
    goals_exists,
    goals_path,
    load_json,
    myth_log,
    save_json,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .main import ChatRequest

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


def _extract_text(data: Any) -> str:
    """Return text content from ``data`` which may be nested."""

    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            item = choices[0]
            if isinstance(item, dict):
                return str(item.get("text", ""))
        return str(data.get("text", ""))
    if isinstance(data, list) and data:
        return _extract_text(data[0])
    if isinstance(data, str):
        return data
    return str(data)


def parse_response(output: Any) -> str:
    """Extract and clean text from a model response chunk."""

    myth_log("pre_parse", raw=str(output))
    text = _extract_text(output)
    return clean_text(text, trim=True)


def stream_parsed(chunks: Iterable[Any]) -> Iterator[str]:
    """Yield cleaned text from streaming model output."""

    for chunk in chunks:
        yield clean_text(_extract_text(chunk))


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


def _find_json_chunk(text: str) -> str | None:
    braces = ("{", "}")
    start = text.find(braces[0])
    end = text.rfind(braces[1])
    if 0 <= start < end:
        chunk = text[start : end + 1]
        try:
            json.loads(chunk)
            return chunk
        except Exception:
            pass

    brackets = ("[", "]")
    start = text.find(brackets[0])
    end = text.rfind(brackets[1])
    if 0 <= start < end:
        chunk = text[start : end + 1]
        try:
            json.loads(chunk)
            return chunk
        except Exception:
            pass
    return None


def _parse_goals_from_response(text: str) -> List[Dict[str, str]]:
    data: Any | None = None
    try:
        data = json.loads(text)
    except Exception:
        chunk = _find_json_chunk(text)
        if chunk is not None:
            try:
                data = json.loads(chunk)
            except Exception:
                data = None

    goals: List[Dict[str, str]] = []
    items: list | None = None
    if isinstance(data, dict) and isinstance(data.get("goals"), list):
        items = data["goals"]
    elif isinstance(data, list):
        items = data

    if items is not None:
        for item in items:
            if isinstance(item, dict):
                desc = str(item.get("description", "")).strip()
                method = str(item.get("method", "")).strip()
                if not desc and isinstance(item.get("text"), str):
                    desc = item["text"].strip()
            else:
                desc = str(item).strip()
                method = ""
            if desc:
                goals.append({"description": desc, "method": method})
        return goals

    for line in text.splitlines():
        line = line.strip(" -*\t")
        if not line:
            continue
        parts = re.split(r"\s*-\s*|:\s*", line, maxsplit=1)
        desc = parts[0].lstrip("0123456789. ").strip()
        method = parts[1].strip() if len(parts) > 1 else ""
        if desc:
            goals.append({"description": desc, "method": method})
    return goals


def _dedupe_new_goals(
    new: List[Dict[str, str]], existing: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    existing_desc = {g.get("description", "") for g in existing}
    return [g for g in new if g.get("description", "") not in existing_desc]


def _maybe_generate_goals(chat_id: str, global_prompt: str) -> None:
    from .call_types import CALL_HANDLERS

    if not goals_exists(chat_id):
        return
    try:
        with open(goals_path(chat_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        character = data.get("character", "")
        setting = data.get("setting", "")
    except Exception:
        return

    state = _load_goal_state(chat_id)
    state["messages_since_goal_eval"] = (
        state.get("messages_since_goal_eval", 0) + 1
    )

    refresh = GENERATION_CONFIG.get("goal_refresh_rate", 1)
    if state["messages_since_goal_eval"] < refresh:
        _save_goal_state(chat_id, state)
        return

    goal_limit = GENERATION_CONFIG.get("goal_limit", 3)

    history = load_json(chat_file(chat_id, "full.json"))
    user_text = "\n".join(m.get("content", "") for m in history)

    system_parts = [p for p in (global_prompt, character, setting) if p]
    system_parts.append(
        "Given the character profile and scene context, determine if the character has any meaningful or natural goals. "
        f"If so, generate up to {goal_limit} specific, actionable goals, each with a brief plan for how the character might pursue it. "
        "If no goals are currently appropriate, return an empty list. "
        'Respond ONLY in JSON format: {"goals": [{"description": "...", "method": "..."}]}.'
    )
    system_text = "\n".join(system_parts)

    from .call_types import CALL_HANDLERS

    handler = CALL_HANDLERS["goal_generation"]
    prompt = handler.prompt(system_text, user_text)
    bg_kwargs = GENERATION_CONFIG.copy()
    bg_kwargs["n_gpu_layers"] = 0
    raw = call_llm(prompt, **bg_kwargs)
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


def handle_chat(req: "ChatRequest", stream: bool = False):
    """Process ``req`` and return a model reply."""

    from .call_types import CALL_HANDLERS

    ensure_chat_dir(req.chat_id)
    history = load_json(chat_file(req.chat_id, "full.json"))
    history.append({"role": "user", "content": req.message})

    call_type = req.call_type or "standard_chat"
    handler = CALL_HANDLERS.get(call_type, CALL_HANDLERS["default"])

    if call_type in ("standard_chat", "user_message"):
        system_parts = [req.global_prompt or ""]
        if goals_exists(req.chat_id):
            try:
                with open(goals_path(req.chat_id), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if data.get("character"):
                        system_parts.append(data["character"])
                    if data.get("setting"):
                        system_parts.append(data["setting"])
            except Exception as e:
                print(f"Failed to load goals for '{req.chat_id}': {e}")
        system_text = "\n".join(p for p in system_parts if p)
        user_text = "\n".join(m["content"] for m in history)
    else:
        system_text = req.global_prompt or ""
        user_text = req.message

    prompt = handler.prompt(system_text, user_text)
    myth_log("model_input", prompt=prompt)
    kwargs = GENERATION_CONFIG.copy()
    kwargs["stream"] = stream
    kwargs["n_gpu_layers"] = 35
    raw = call_llm(prompt, **kwargs)
    processed = handler.response(raw)

    if stream:

        def _generate():
            parts = []
            for text in processed:
                parts.append(text)
                yield text
            assistant_reply = "".join(parts).strip()
            history.append({"role": "assistant", "content": assistant_reply})
            save_json(chat_file(req.chat_id, "full.json"), history)
            enqueue_task(
                "goal_generation",
                _maybe_generate_goals,
                req.chat_id,
                req.global_prompt or "",
            )

        return StreamingResponse(_generate(), media_type="text/plain")

    assistant_reply = (
        processed if isinstance(processed, str) else str(processed)
    )
    history.append({"role": "assistant", "content": assistant_reply})
    save_json(chat_file(req.chat_id, "full.json"), history)
    enqueue_task(
        "goal_generation",
        _maybe_generate_goals,
        req.chat_id,
        req.global_prompt or "",
    )
    return {"detail": assistant_reply}
