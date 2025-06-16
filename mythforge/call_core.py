from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, Iterator, List
import json

from .model import GENERATION_CONFIG

from fastapi.responses import StreamingResponse

from .invoker import LLMInvoker
from .prompt_preparer import PromptPreparer
from .response_parser import ResponseParser, _parse_goals_from_response
from .memory import MemoryManager, MEMORY_MANAGER
from .logger import LOGGER
from .background import schedule_task, has_pending_tasks


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: str, *, trim: bool = False) -> str:
    """Strip unwanted tokens and optionally trim whitespace."""

    cleaned = text.replace("<|eot_id|>", "")
    return cleaned.strip() if trim else cleaned


# ---------------------------------------------------------------------------
# Core chat handling
# ---------------------------------------------------------------------------


def _evaluate_goals(
    chat_name: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
) -> None:
    """Generate new goals when below the configured limit."""
    LOGGER.log(
        "chat_flow",
        {
            "function": "evaluate_goals",
            "chat_name": chat_name,
            "global_prompt": global_prompt,
            "memory_root": memory.root_dir,
        },
    )
    goals = memory.load_goals(chat_name)
    if not memory.goals_active:
        return
    character = goals.character
    setting = goals.setting

    state = memory.load_goal_state(chat_name)
    goal_limit = GENERATION_CONFIG.get("goal_limit", 3)
    if len(state.get("goals", [])) >= goal_limit:
        return

    chat_history = memory.load_chat_history(chat_name)
    user_text = "\n".join(m.get("content", "") for m in chat_history)

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

    LOGGER.log(
        "prepared_prompts",
        {
            "call_type": "goal_generation",
            "system_text": system_text,
            "user_text": user_text,
        },
    )

    from .call_templates import generate_goals

    try:
        text = generate_goals.generate_goals(
            system_text,
            user_text,
            {**generate_goals.MODEL_LAUNCH_OVERRIDE},
        )
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.log_error(exc)
        state["error"] = str(exc)
        memory.save_goal_state(chat_name, state)
        return

    parsed = ResponseParser().load(text).parse()
    if isinstance(parsed, Iterator):
        parsed = "".join(parsed)
    cleaned = clean_text(str(parsed), trim=True)
    new_goals = _parse_goals_from_response(cleaned)

    state.pop("error", None)
    combined = state.get("goals", []) + new_goals
    state["goals"] = combined[:goal_limit]
    memory.save_goal_state(chat_name, state)


def evaluate_goals(
    chat_name: str,
    global_prompt: str,
    memory: MemoryManager = MEMORY_MANAGER,
    *,
    background: bool = True,
) -> None:
    """Schedule or run goal evaluation."""

    if background:
        schedule_task(_evaluate_goals, chat_name, global_prompt, memory)
        return

    _evaluate_goals(chat_name, global_prompt, memory)


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

    from .call_templates import standard_chat, logic_check

    options = options or {"stream": stream}

    call_map = {
        "logic_check": lambda: logic_check.logic_check(global_prompt, message, options),
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
