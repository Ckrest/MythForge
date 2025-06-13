from __future__ import annotations

import json
import os
import re
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, TYPE_CHECKING, Callable

from fastapi.responses import StreamingResponse

from .model import (
    GENERATION_CONFIG,
    DEFAULT_N_GPU_LAYERS,
    call_llm,
    llm_args,
    warm_up,
    _stop_warm,
)
from .utils import (
    CHATS_DIR,
    chat_file,
    ensure_chat_dir,
    load_json,
    myth_log,
    save_json,
    list_prompt_names,
    get_global_prompt_content,
)
from . import memory

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
    names = list_prompt_names()
    if not names:
        return ""
    content = get_global_prompt_content(names[0])
    return content or ""


def build_call(req: "ChatRequest") -> CallData:
    """Return :class:`CallData` populated from ``req`` and defaults."""

    return CallData(
        chat_id=req.chat_id,
        message=req.message,
    )


_current_chat_id: str | None = None
_current_prompt: str | None = None

# --- Background task queue -------------------------------------------------

_task_queue: queue.Queue[tuple[str, Callable[..., None], tuple]] = queue.Queue()
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


def _strip_model_logs(text: str) -> str:
    """Return ``text`` without lines from model loading logs."""

    noise_prefixes = (
        "llama_model_load:",
        "llama_init_from_gpt_params:",
        "llama_print_timings:",
        "llama_new_context_with_model:",
        "main:",
        "system_info:",
        "ggml_vulkan:",
        "load_tensors:",
        "print_info:",
        "llama_context:",
        "llama_kv_cache_unified:",
    )

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in noise_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines)


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
    text = _strip_model_logs(text)
    return clean_text(text, trim=True)


def stream_parsed(chunks: Iterable[Any]) -> Iterator[str]:
    """Yield cleaned text from streaming model output."""

    for chunk in chunks:
        text = _extract_text(chunk)
        text = _strip_model_logs(text)
        cleaned = clean_text(text)
        if cleaned:
            yield cleaned


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

    goals = memory.MEMORY.goals_data
    if not goals.enabled:
        return
    character = goals.character
    setting = goals.setting

    state = _load_goal_state(chat_id)
    state["messages_since_goal_eval"] = state.get("messages_since_goal_eval", 0) + 1

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
    system_prompt, user_prompt = handler.prompt(system_text, user_text)
    raw = call_llm(
        system_prompt,
        user_prompt,
        **llm_args(background=True),
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


def _finalize_chat(history: list, reply: str, call: CallData) -> None:
    """Update ``history`` and run post-response tasks."""

    history.append({"role": "assistant", "content": reply})
    save_json(chat_file(call.chat_id, "full.json"), history)
    enqueue_task(
        "goal_generation",
        _maybe_generate_goals,
        call.chat_id,
        call.global_prompt,
    )
    warm_up(_current_prompt or "", n_gpu_layers=DEFAULT_N_GPU_LAYERS)


def handle_chat(call: CallData, stream: bool = False):
    """Process ``call`` and return a model reply."""

    from .call_types import CALL_HANDLERS

    global _current_chat_id, _current_prompt
    _stop_warm()

    ensure_chat_dir(call.chat_id)
    history = load_json(chat_file(call.chat_id, "full.json"))
    history.append({"role": "user", "content": call.message})

    handler = CALL_HANDLERS.get(call.call_type, CALL_HANDLERS["default"])

    system_text, user_text = handler.prepare(call, history)

    if call.chat_id != _current_chat_id or system_text != (_current_prompt or ""):
        _current_chat_id = call.chat_id
        _current_prompt = system_text

    system_prompt, user_prompt = handler.prompt(system_text, user_text)
    myth_log("model_input", prompt=user_prompt)
    raw = call_llm(
        system_prompt,
        user_prompt,
        **llm_args(stream=stream),
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
                parts.append(text)
                yield text
            assistant_reply = "".join(parts).strip()
            _finalize_chat(history, assistant_reply, call)

        return StreamingResponse(_generate(), media_type="text/plain")

    assistant_reply = processed if isinstance(processed, str) else str(processed)
    _finalize_chat(history, assistant_reply, call)

    return {"detail": assistant_reply}
