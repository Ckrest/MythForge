# Goal tracking utilities for Myth Forge
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from airoboros_prompter import format_llama3

# ``CHATS_DIR`` mirrors the constant from ``MythForgeServer``.  Keeping a copy
# here avoids circular imports.
CHATS_DIR = "chats"

STATE_SUFFIX = "_state.json"

# Configurable parameters
HISTORY_WINDOW = int(os.environ.get("MF_HISTORY_WINDOW", 8))
MIN_ACTIVE_GOALS = int(os.environ.get("MF_MIN_ACTIVE_GOALS", 3))

logger = logging.getLogger("goal_tracker")


class GoalModel(BaseModel):
    id: str
    description: str
    method: str
    status: Optional[str] = None


class GoalsListModel(BaseModel):
    goals: List[GoalModel]


class InitialStateModel(BaseModel):
    character_profile: Optional[Dict[str, Any]] = None
    scene_context: Optional[Dict[str, Any]] = None
    goals: List[GoalModel] = []


def _parse_json(text: str, model: Type[BaseModel]) -> Optional[BaseModel]:
    """Parse ``text`` using ``model`` after stripping fences."""
    cleaned = _extract_json(text)
    try:
        data = json.loads(cleaned)
    except Exception as e:
        logger.warning("JSON decoding failed: %s", e)
        logger.debug("Raw text: %s", text)
        return None

    try:
        return model.parse_obj(data)
    except ValidationError as e:
        logger.warning("Schema validation failed: %s", e)
        logger.debug("Raw data: %s", data)
        return None


def _extract_json(text: str) -> str:
    """Return ``text`` with surrounding Markdown code fences removed."""
    cleaned = text.strip()
    # Find the first fence
    if "```" in cleaned:
        start = cleaned.find("```")
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start + 3 : end]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _state_path(chat_id: str) -> str:
    return os.path.join(CHATS_DIR, f"{chat_id}{STATE_SUFFIX}")


def load_state(chat_id: str) -> Dict[str, Any]:
    path = _state_path(chat_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if "messages_since_goal_eval" not in data:
                    data["messages_since_goal_eval"] = 0
                if "completed_goals" not in data:
                    data["completed_goals"] = []
                return data
            except Exception as e:
                logger.warning("Failed to load state '%s': %s", path, e)
    return {
        "character_profile": None,
        "scene_context": None,
        "goals": [],
        "completed_goals": [],
        "messages_since_goal_eval": 0,
    }


def save_state(chat_id: str, state: Dict[str, Any]) -> None:
    path = _state_path(chat_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    logger.debug("Saved state to %s", path)


def generate_initial_state(call_fn, global_prompt: str, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
    """Use the language model to create an initial state for the character."""
    logger.info("Generating initial state")
    instruction = (
        "Analyze the global prompt, the user's first message, and the assistant's first reply. "
        "Return JSON with keys 'character_profile', 'scene_context', and 'goals'. "
        "'character_profile' must contain: name, personality, background, current_location, "
        "known_conflicts, relationships. 'scene_context' must contain: scene_description, "
        "setting_details, known_events. 'goals' is a list of 2-3 short term goals with id, "
        "description and method."
    )
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                f"GLOBAL PROMPT:\n{global_prompt}\n\n"
                f"FIRST USER MESSAGE:\n{user_msg}\n\n"
                f"FIRST ASSISTANT MESSAGE:\n{assistant_msg}"
            ),
        },
    ]
    prompt = format_llama3("", None, messages, "")
    logger.debug("Initial state prompt:\n%s", prompt)
    output = call_fn(prompt, max_tokens=400)
    text = output["choices"][0]["text"].strip()
    logger.debug("Initial state raw output:\n%s", text)
    model = _parse_json(text, InitialStateModel)
    if model is None:
        logger.warning("Failed to parse initial state")
        return {"character_profile": None, "scene_context": None, "goals": []}
    data = model.dict()
    logger.debug("Parsed initial state: %s", data)
    return data


def generate_goals(call_fn, character_profile: Dict[str, Any], scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate new goals using the language model."""
    logger.info("Generating goals")
    instruction = (
        "Based on the character profile and scene context, generate 2 to 3 specific, "
        "actionable goals for the character. Return JSON list of goal objects with id, "
        "description and method."
    )
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": json.dumps({"character_profile": character_profile, "scene_context": scene_context}, ensure_ascii=False),
        },
    ]
    prompt = format_llama3("", None, messages, "")
    logger.debug("Goals prompt:\n%s", prompt)
    output = call_fn(prompt, max_tokens=200)
    text = output["choices"][0]["text"].strip()
    logger.debug("Goals raw output:\n%s", text)
    try:
        data = json.loads(_extract_json(text))
    except Exception as e:
        logger.warning("Failed to parse goals: %s", e)
        return []
    if isinstance(data, list):
        result = []
        for g in data:
            if isinstance(g, dict) and "id" in g and "description" in g:
                result.append(g)
        logger.debug("Parsed goals: %s", result)
        return result
    logger.warning("Goal generation returned invalid format")
    return []


def ensure_initial_state(call_fn, chat_id: str, global_prompt: str, first_user: str, first_assistant: str) -> None:
    logger.info("Generating initial state", extra={"chat_id": chat_id})
    state = load_state(chat_id)
    if state.get("character_profile") is not None:
        return
    if "**goals**" not in global_prompt.lower():
        return
    data = generate_initial_state(call_fn, global_prompt, first_user, first_assistant)
    state.update(data)
    save_state(chat_id, state)


def check_and_generate_goals(call_fn, chat_id: str) -> None:
    logger.info("Checking for missing goals", extra={"chat_id": chat_id})
    state = load_state(chat_id)
    if state.get("character_profile") and state.get("scene_context") and not state.get("goals"):
        goals = generate_goals(call_fn, state["character_profile"], state["scene_context"])
        state["goals"] = goals
        save_state(chat_id, state)
    else:
        logger.debug("Goals already present or character state incomplete", extra={"chat_id": chat_id})


def _load_json(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception as e:
            logger.warning("Failed to load json '%s': %s", path, e)
    return []


def load_and_prepare_state(chat_id: str, history_window: int = HISTORY_WINDOW):
    state = load_state(chat_id)
    trimmed_path = os.path.join(CHATS_DIR, f"{chat_id}_trimmed.json")
    history = _load_json(trimmed_path)
    convo: List[Dict[str, str]] = []
    for m in history[-history_window:]:
        if m.get("type") == "summary":
            convo.append({"role": "system", "content": f"SUMMARY: {m['content']}"})
        else:
            role = "assistant" if m.get("role") == "bot" else m.get("role", "user")
            convo.append({"role": role, "content": m.get("content", "")})
    return state, convo


def build_prompt(convo: List[Dict[str, str]], instruction: str) -> str:
    return format_llama3("", None, convo, instruction)


def parse_and_merge_goals(data: GoalsListModel, state: Dict[str, Any], min_active: int, call_fn) -> None:
    completed = state.get("completed_goals", [])
    active = []
    seen_ids = {g.get("id") for g in completed}
    for g in data.goals:
        status = (g.status or "").lower()
        if status in ("completed", "abandoned"):
            completed.append(g.dict())
            seen_ids.add(g.id)
            continue
        if g.id in seen_ids:
            continue
        active.append(g.dict())
        seen_ids.add(g.id)

    state["completed_goals"] = completed
    state["goals"] = active

    if (
        len(active) < min_active
        and state.get("character_profile")
        and state.get("scene_context")
    ):
        new_goals = generate_goals(call_fn, state["character_profile"], state["scene_context"])
        for g in new_goals:
            gid = g.get("id")
            if not gid or gid in seen_ids:
                continue
            active.append(g)
            seen_ids.add(gid)
            if len(active) >= min_active:
                break
        state["goals"] = active


def _error_path(chat_id: str) -> str:
    """Return the path used for goal evaluation error logs."""
    return os.path.join(CHATS_DIR, f"{chat_id}_goal_eval_error.txt")


def format_goal_eval_response(text: str, chat_id: str) -> Optional[GoalsListModel]:
    """Return a ``GoalsListModel`` parsed from ``text``.

    If ``text`` cannot be parsed or does not conform to the schema, the raw
    value is written to an error file for troubleshooting and ``None`` is
    returned.
    """

    model = _parse_json(text, GoalsListModel)
    if model is not None:
        return model

    path = _error_path(chat_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.warning("Wrote invalid goal evaluation output to %s", path)
    except Exception as e:  # pragma: no cover - file write errors are rare
        logger.warning("Failed to write goal evaluation error file '%s': %s", path, e)
    return None


def evaluate_goals(call_fn, chat_id: str, history_window: int = HISTORY_WINDOW, min_active: int = MIN_ACTIVE_GOALS, max_retries: int = 3) -> None:
    """Evaluate progress on current goals based on recent conversation."""
    state, convo = load_and_prepare_state(chat_id, history_window)
    goals = state.get("goals") or []
    if not goals:
        logger.info("No goals to evaluate", extra={"chat_id": chat_id})
        return

    instruction = (
        "Evaluate the character's goals based solely on the recent messages. "
        "For each goal decide if it is completed, in_progress, needs_tactics, or abandoned. "
        "Add new short term goals only when others are completed or abandoned. "
        "You must respond with ONLY valid JSON exactly matching this schema:\n"
        '{"goals": [{"id": "<id>", "description": "<desc>", "method": "<method>", "status": "<status>"}]}'
    )

    prompt = build_prompt(convo, instruction)
    logger.debug("Goal evaluation prompt:\n%s", prompt)

    # Trim context if it exceeds threshold (approx token count)
    while len(prompt.split()) > 3500 and convo:
        convo.pop(0)
        prompt = build_prompt(convo, instruction)
    if len(prompt.split()) > 3500:
        logger.warning("Context truncated for goal evaluation", extra={"chat_id": chat_id})

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            output = call_fn(prompt, max_tokens=300)
            text = output["choices"][0]["text"].strip()
            logger.debug("Goal evaluation raw output:\n%s", text)
            model = format_goal_eval_response(text, chat_id)
            if model is None:
                raise ValueError("invalid json")
            parse_and_merge_goals(model, state, min_active, call_fn)
            state["messages_since_goal_eval"] = 0
            save_state(chat_id, state)
            return
        except Exception as e:
            logger.warning("Goal evaluation attempt %s failed: %s", attempt + 1, e, extra={"chat_id": chat_id})
            time.sleep(backoff)
            backoff *= 2

    # If we get here, all retries failed
    logger.warning("Goal evaluation failed after retries", extra={"chat_id": chat_id})
    save_state(chat_id, state)


def record_user_message(chat_id: str) -> None:
    path = _state_path(chat_id)
    if not os.path.exists(path):
        return
    state = load_state(chat_id)
    state["messages_since_goal_eval"] = state.get("messages_since_goal_eval", 0) + 1
    save_state(chat_id, state)


def record_assistant_message(chat_id: str) -> bool:
    """Increment message counter and return True if goal evaluation is due."""
    path = _state_path(chat_id)
    if not os.path.exists(path):
        return False
    state = load_state(chat_id)
    state["messages_since_goal_eval"] = state.get("messages_since_goal_eval", 0) + 1
    save_state(chat_id, state)
    return bool(state.get("goals")) and state["messages_since_goal_eval"] >= 4


def state_as_prompt_fragment(state: Dict[str, Any]) -> str:
    lines = []
    cp = state.get("character_profile")
    sc = state.get("scene_context")
    gs = state.get("goals", [])
    if cp:
        lines.append("[CHARACTER PROFILE]")
        for k, v in cp.items():
            lines.append(f"- {k}: {v}")
    if sc:
        lines.append("\n[SCENE CONTEXT]")
        for k, v in sc.items():
            lines.append(f"- {k}: {v}")
    if gs:
        lines.append("\n[CURRENT GOALS]")
        for g in gs:
            status = g.get("status")
            desc = f"- {g.get('id')}: {g.get('description')} ({g.get('method')})"
            if status:
                desc += f" [{status}]"
            lines.append(desc)
    return "\n".join(lines)

