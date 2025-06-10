"""Utilities for formatting prompts and calling the language model."""

from __future__ import annotations

from typing import Iterable, Iterator, List, TYPE_CHECKING, Dict

import os

from .server_log import myth_log

import json
from fastapi.responses import StreamingResponse


if TYPE_CHECKING:
    from .MythForgeServer import ChatRequest


def clean_text(text: str, *, trim: bool = False) -> str:
    """Return ``text`` with tokens removed and optional trimming."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned.strip() if trim else cleaned


def parse_response(output: dict) -> str:
    """Extract and clean the text portion from a model call result."""
    myth_log("pre_parse", raw=str(output))
    text = output.get("choices", [{}])[0].get("text", "")
    return clean_text(text, trim=True)


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
        goals_exists,
        goals_path,
    )

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})

    call_type = req.call_type or "user_message"

    if call_type == "user_message":
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
            except Exception as e:  # pragma: no cover - best effort
                print(f"Failed to load goals for '{req.chat_id}': {e}")
        system_text = "\n".join(p for p in system_parts if p)
        user_text = "\n".join(m["content"] for m in history)
        chunks = make_model_call(system_text, user_text, "standard_chat")
    else:
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
        _maybe_generate_goals(req.chat_id, req.global_prompt or "")

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
    _maybe_generate_goals(req.chat_id, req.global_prompt or "")
    return {"detail": assistant_reply}


def _state_path(chat_id: str) -> str:
    """Return the path to ``chat_id``'s goal state file."""

    from .MythForgeServer import CHATS_DIR

    return os.path.join(CHATS_DIR, chat_id, "goal_state.json")


def _load_goal_state(chat_id: str) -> Dict[str, object]:
    """Return the goal state for ``chat_id``."""

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


def _save_goal_state(chat_id: str, state: Dict[str, object]) -> None:
    """Save ``state`` for ``chat_id``."""

    path = _state_path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _parse_goals_from_response(text: str) -> List[Dict[str, str]]:
    """Return a goal list parsed from ``text`` if possible."""

    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("goals"), list):
            goals = []
            for item in data["goals"]:
                if not isinstance(item, dict):
                    continue
                desc = str(item.get("description", "")).strip()
                method = str(item.get("method", "")).strip()
                if desc:
                    goals.append({"description": desc, "method": method})
            return goals
    except Exception:
        pass
    return []


def _dedupe_new_goals(
    new: List[Dict[str, str]], existing: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Return ``new`` without goals already in ``existing``."""

    existing_desc = {g.get("description", "") for g in existing}
    return [g for g in new if g.get("description", "") not in existing_desc]


def _maybe_generate_goals(chat_id: str, global_prompt: str) -> None:
    """Generate new goals for ``chat_id`` if the refresh rate is met."""

    from .MythForgeServer import (
        load_item,
        make_model_call,
        goals_exists,
        goals_path,
    )
    from . import model_launch

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
    state["messages_since_goal_eval"] = state.get("messages_since_goal_eval", 0) + 1

    refresh = model_launch.MODEL_SETTINGS.get("goal_refresh_rate", 1)
    if state["messages_since_goal_eval"] < refresh:
        _save_goal_state(chat_id, state)
        return

    goal_limit = model_launch.MODEL_SETTINGS.get("goal_limit", 3)

    history = load_item("chat_history", chat_id)
    user_text = "\n".join(m.get("content", "") for m in history)

    system_parts = [p for p in (global_prompt, character, setting) if p]
    system_parts.append(
        "Given the character profile and scene context, determine if the character has any meaningful or natural goals. "
        f"If so, generate up to {goal_limit} specific, actionable goals, each with a brief plan for how the character might pursue it. "
        "If no goals are currently appropriate, return an empty list. "
        'Respond ONLY in JSON format: {"goals": [{"description": "...", "method": "..."}]}.'
    )
    system_text = "\n".join(system_parts)

    output = make_model_call(system_text, user_text, "goal_generation")
    if isinstance(output, Iterable):
        output = next(iter(output), {})
    text = parse_response(output)
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
