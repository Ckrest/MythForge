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

STATE_NAME = "state.json"

# Configurable parameters
HISTORY_WINDOW = int(os.environ.get("MF_HISTORY_WINDOW", 8))
MIN_ACTIVE_GOALS = int(os.environ.get("MF_MIN_ACTIVE_GOALS", 3))
DEBUG_MODE = os.environ.get("MF_DEBUG", "0") in ("1", "true", "yes")
MAX_GOALS = 3

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
    return os.path.join(CHATS_DIR, chat_id, STATE_NAME)


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


def save_state(chat_id: str, state: Dict[str, Any]) -> bool:
    """Persist ``state`` to disk if it differs from the existing file.

    Returns ``True`` when the file was written.
    """

    path = _state_path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing: Optional[Dict[str, Any]] = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None

    if existing == state:
        logger.debug("State unchanged; not writing %s", path)
        return False

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.debug("Saved state to %s", path)
    log_event("state_saved", {"path": path})
    return True




def extract_scene_context(text: str) -> str:
    """Return a basic scene context extracted from ``text``."""
    return text.strip()


def extract_character_profile(text: str) -> str:
    """Return a basic character profile extracted from ``text``."""
    return text.strip()


def init_state_from_prompt(chat_id: str, global_prompt: str, first_user: str) -> None:
    """Initialize state using ``global_prompt`` and ``first_user`` if unset."""
    logger.info("Initializing state from prompt", extra={"chat_id": chat_id})
    state = load_state(chat_id)
    if state.get("character_profile") is None:
        state["character_profile"] = global_prompt.strip()
    if state.get("scene_context") is None:
        state["scene_context"] = first_user.strip()
    if save_state(chat_id, state):
        log_event("state_saved", {"path": _state_path(chat_id)})


def ensure_initial_state(call_fn, chat_id: str, global_prompt: str, first_user: str, first_assistant: str) -> None:
    """Populate ``scene_context`` and ``character_profile`` from the first exchange."""
    logger.info("Ensuring initial state", extra={"chat_id": chat_id})
    state = load_state(chat_id)
    if state.get("scene_context") is None:
        state["scene_context"] = extract_scene_context(first_user)
    if state.get("character_profile") is None:
        state["character_profile"] = extract_character_profile(first_assistant)
    if save_state(chat_id, state):
        log_event("state_saved", {"path": _state_path(chat_id)})


def parse_goals_from_response(
    text: str, goals: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Parse a goal list from LLM text response, robust to various formats.

    If ``goals`` is provided, new items are appended to it. A new list is
    created otherwise and returned.
    """
    if goals is None:
        goals = []

    # 1. Try strict JSON via Pydantic model
    model = _parse_json(text, GoalsListModel)
    if model is not None:
        goals.extend(g.dict() for g in model.goals)
        return goals

    # 2. Try manual JSON extraction
    extracted = _extract_json(text)
    try:
        data = json.loads(extracted)
        if isinstance(data, dict) and "goals" in data:
            data = data["goals"]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    g = {
                        "id": item.get("id") or f"g{len(goals)+1}",
                        "description": item.get("description") or item.get("text", ""),
                        "method": item.get("method", ""),
                    }
                    if item.get("status") is not None:
                        g["status"] = item["status"]
                    if g["description"]:
                        goals.append(g)
                elif isinstance(item, str):
                    goals.append({"id": f"g{len(goals)+1}", "description": item.strip(), "method": ""})
            return goals
    except json.JSONDecodeError:
        pass

    # 3. Fallback: numbered/bulleted lists
    numbered = re.findall(r'^\s*\d+[\.)]\s*(.+)$', text, flags=re.MULTILINE)
    bullets = re.findall(r'^\s*[-\*•]\s*(.+)$', text, flags=re.MULTILINE)
    lines = numbered if numbered else bullets
    if not lines:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        goals.append({"id": f"g{len(goals)+1}", "description": line.strip(), "method": ""})
    return goals


def _merge_goals(
    existing: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
    remove_missing: bool = False,
) -> tuple[list[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Return ``existing`` merged with ``incoming`` keyed by ``id``.

    Each goal in the result contains the keys ``id``, ``description``,
    ``method`` and ``status``.  ``incoming`` values overwrite existing ones only
    when they are not ``None``.  When ``remove_missing`` is ``True`` any goal
    not present in ``incoming`` is dropped from the result and reported in the
    ``removed`` list.
    """

    by_id = {g.get("id"): g.copy() for g in existing if g.get("id")}
    added: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []
    processed: set[str] = set()

    result: List[Dict[str, Any]] = []
    for g in incoming:
        gid = g.get("id")
        if not gid:
            continue
        processed.add(gid)
        base = by_id.pop(gid, {}).copy()
        if not base:
            base["id"] = gid
        changed = False
        for field in ("description", "method", "status"):
            val = g.get(field)
            if val is not None:
                if base.get(field) != val:
                    base[field] = val
                    changed = True
            elif field not in base:
                base[field] = None
        if gid not in {e.get("id") for e in existing}:
            added.append(base.copy())
        elif changed:
            updated.append(base.copy())
        result.append(base)

    removed = []
    if remove_missing:
        removed = list(by_id.values())
    else:
        result.extend(by_id.values())

    diff = {"added": added, "updated": updated, "removed": removed}
    return result, diff


def _apply_goal_update(
    chat_id: str,
    state: Dict[str, Any],
    new_goals: List[Dict[str, Any]],
    original_goals: List[Dict[str, Any]],
    remove_missing: bool = False,
) -> bool:
    """Merge ``new_goals`` into ``state`` based on ``original_goals``.

    ``original_goals`` should represent the goals as they were loaded from disk
    before any in-memory modifications.  This ensures comparisons against the
    persisted state rather than any mutated version currently in ``state``.
    """

    merged, diff = _merge_goals(original_goals, new_goals, remove_missing)
    if diff["added"] or diff["updated"] or diff["removed"]:
        state["goals"] = merged
        logger.debug(
            "Goal update diff added=%s updated=%s removed=%s",
            diff["added"],
            diff["updated"],
            diff["removed"],
            extra={"chat_id": chat_id},
        )
        log_event(
            "goals_changed",
            {
                "chat_id": chat_id,
                "added": diff["added"],
                "updated": diff["updated"],
                "removed": diff["removed"],
            },
        )
        return True
    return False


def check_and_generate_goals(call_fn, chat_id: str) -> None:
    """Generate goals if none exist using current state."""
    logger.info("Checking for missing goals", extra={"chat_id": chat_id})
    state = load_state(chat_id)
    original_goals = state.get("goals", []).copy()
    if len(state.get("goals", [])) >= MAX_GOALS:
        return
    if not state.get("character_profile") or not state.get("scene_context"):
        logger.debug("Character state incomplete", extra={"chat_id": chat_id})
        return
    instruction = (
        "Based on the character profile and scene context, generate 2 to 3 specific, "
        "actionable goals for the character. Return JSON list of goal objects with id, "
        "description and method."
    )
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": json.dumps({"character_profile": state["character_profile"], "scene_context": state["scene_context"]}, ensure_ascii=False),
        },
    ]
    prompt = format_llama3("", None, messages, "")
    logger.debug("Goal generation prompt:\n%s", prompt, extra={"chat_id": chat_id})
    logger.info("LLM goal prompt", extra={"chat_id": chat_id, "prompt": prompt})
    logger.debug("Final LLM prompt:\n%s", prompt)
    log_event("llm_final_prompt", {"prompt": prompt})

    for attempt in range(1, 3):
        output = call_fn(prompt, max_tokens=200)
        text = output["choices"][0]["text"].strip()
        log_event("goals_llm_raw", {"raw": text})
        logger.debug(
            "Goal generation attempt %d raw output:\n%s",
            attempt,
            text,
            extra={"chat_id": chat_id},
        )
        goals = parse_goals_from_response(text)
        if goals:
            changed = _apply_goal_update(chat_id, state, goals, original_goals)
            if changed:
                state["messages_since_goal_eval"] = 0
                if save_state(chat_id, state):
                    log_event("state_saved", {"path": _state_path(chat_id)})
                logger.info(
                    "Goals updated: %s",
                    goals,
                    extra={"chat_id": chat_id},
                )
                return
            else:
                return
        else:
            log_event("goals_parse_failed", {"raw": text})
            logger.warning(
                "Goal parsing returned no results on attempt %d",
                attempt,
                extra={"chat_id": chat_id},
            )

    logger.warning(
        "Goal generation failed after 2 attempts",
        extra={"chat_id": chat_id}
    )


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
    trimmed_path = os.path.join(CHATS_DIR, chat_id, "trimmed.json")
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






def format_goal_eval_response(text: str, chat_id: str) -> Optional[GoalsListModel]:
    """Return a ``GoalsListModel`` parsed from ``text``.

    If ``text`` cannot be parsed or does not conform to the schema, the raw
    value is logged for troubleshooting and ``None`` is returned.
    """

    model = _parse_json(text, GoalsListModel)
    if model is not None:
        return model

    logger.warning("Invalid goal evaluation output", extra={"chat_id": chat_id})
    log_event("goal_eval_invalid_output", {"raw": text, "chat_id": chat_id})
    return None


def evaluate_and_update_goals(
    call_fn,
    chat_id: str,
    history_window: int = HISTORY_WINDOW,
    min_active: int = MIN_ACTIVE_GOALS,
    max_retries: int = 3,
) -> None:
    """Evaluate progress on goals and replenish if needed."""

    state, convo = load_and_prepare_state(chat_id, history_window)
    original_goals = state.get("goals", []).copy()
    goals = state.get("goals") or []
    if not goals:
        logger.info("No goals to evaluate", extra={"chat_id": chat_id})
        return

    goals_json = json.dumps(goals, ensure_ascii=False)
    instruction = (
        "Given the recent conversation, update the status of each goal in this list:\n"
        f"{goals_json}\n"
        "Return ONLY JSON like:\n"
        '{"goals": [{"id": "g1", "description": "...", "method": "...", "status": "completed"}]}\n'
        "Status must be one of: in_progress, completed, abandoned."
    )

    prompt = build_prompt(convo, instruction)
    logger.debug("Goal evaluation prompt:\n%s", prompt)

    while len(prompt.split()) > 3500 and convo:
        convo.pop(0)
        prompt = build_prompt(convo, instruction)
    if len(prompt.split()) > 3500:
        logger.warning("Context truncated for goal evaluation", extra={"chat_id": chat_id})

    logger.debug("Final LLM prompt:\n%s", prompt)
    log_event("llm_final_prompt", {"prompt": prompt})

    backoff = 1.0
    for attempt in range(max_retries):
        output = call_fn(prompt, max_tokens=300)
        text = output["choices"][0]["text"].strip()
        logger.debug("Goal evaluation raw output:\n%s", text)
        model = format_goal_eval_response(text, chat_id)
        if model is None:
            logger.warning(
                "Goal evaluation attempt %s returned invalid json", attempt + 1, extra={"chat_id": chat_id}
            )
            time.sleep(backoff)
            backoff *= 2
            continue

        completed = state.get("completed_goals", [])
        existing = {g.get("id"): g for g in state.get("goals", [])}
        active: List[Dict[str, Any]] = []
        seen_ids = {g.get("id") for g in completed}
        seen_desc = {g.get("description", "").lower() for g in completed}
        processed: set[str] = set()

        for g in model.goals:
            orig = existing.pop(g.id, None)
            goal_data = orig.copy() if orig is not None else {"id": g.id}

            for field in ("description", "method"):
                val = getattr(g, field, None)
                if val is not None:
                    goal_data[field] = val
                elif field not in goal_data:
                    goal_data[field] = ""

            status = (
                g.status
                if g.status is not None
                else goal_data.get("status")
                or "in_progress"
            ).lower()
            goal_data["status"] = status

            desc_key = goal_data.get("description", "").lower()

            if status in ("completed", "abandoned"):
                if goal_data["id"] not in seen_ids:
                    completed.append(goal_data)
                    seen_ids.add(goal_data["id"])
                    seen_desc.add(desc_key)
                processed.add(goal_data["id"])
                continue

            if goal_data["id"] in seen_ids or desc_key in seen_desc or goal_data["id"] in processed:
                continue

            active.append(goal_data)
            seen_ids.add(goal_data["id"])
            seen_desc.add(desc_key)
            processed.add(goal_data["id"])

        for gid, g in existing.items():
            desc_key = g.get("description", "").lower()
            if gid in seen_ids or desc_key in seen_desc:
                continue
            active.append(g)
            seen_ids.add(gid)
            seen_desc.add(desc_key)

        changed = False
        old_completed = state.get("completed_goals", [])
        if completed != old_completed:
            state["completed_goals"] = completed
            changed = True

        active_changed = _apply_goal_update(chat_id, state, active, original_goals, remove_missing=True)
        changed = changed or active_changed

        state["messages_since_goal_eval"] = 0

        in_progress_count = sum(
            1 for g in active if (g.get("status") or "in_progress").lower() == "in_progress"
        )

        if (
            in_progress_count < min_active
            and state.get("character_profile")
            and state.get("scene_context")
        ):
            if save_state(chat_id, state):
                log_event("state_saved", {"path": _state_path(chat_id)})
            check_and_generate_goals(call_fn, chat_id)
            return

        if changed or active_changed:
            logger.debug("Goals updated: %s", state.get("goals"))
        else:
            logger.debug("Goals evaluated with no changes", extra={"chat_id": chat_id})

        if save_state(chat_id, state):
            log_event("state_saved", {"path": _state_path(chat_id)})
        return

    logger.warning("Goal evaluation failed after retries", extra={"chat_id": chat_id})
    if save_state(chat_id, state):
        log_event("state_saved", {"path": _state_path(chat_id)})


def record_user_message(chat_id: str) -> None:
    """Increment the message counter for ``chat_id``."""
    state = load_state(chat_id)
    state["messages_since_goal_eval"] = state.get("messages_since_goal_eval", 0) + 1
    if save_state(chat_id, state):
        log_event("state_saved", {"path": _state_path(chat_id)})


def record_assistant_message(chat_id: str) -> bool:
    """Increment the counter and return True when goal evaluation should run."""
    state = load_state(chat_id)
    state["messages_since_goal_eval"] = state.get("messages_since_goal_eval", 0) + 1
    if save_state(chat_id, state):
        log_event("state_saved", {"path": _state_path(chat_id)})
    return bool(state.get("goals")) and state["messages_since_goal_eval"] >= 4


def state_as_prompt_fragment(state: Dict[str, Any]) -> str:
    """Return a short prompt fragment describing the current state.

    ``scene_context`` is included while ``character_profile`` is only used to
    determine if the overall state is empty.  When both values are missing an
    empty string is returned.  Otherwise the scene context is emitted with a
    ``[missing]`` placeholder when absent.
    """

    scene_context = state.get("scene_context")
    character_profile = state.get("character_profile")

    if not scene_context and not character_profile:
        return ""

    parts = [f"Scene Context:\n{scene_context if scene_context else '[missing]'}"]

    return "\n\n".join(parts)

# Apply automatic logging to all functions in this module
import sys
from server_log import patch_module_functions, log_event
patch_module_functions(sys.modules[__name__], "goals system")

