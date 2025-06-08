# Goal tracking utilities for Myth Forge
import json
import os
from typing import Dict, Any, List

from airoboros_prompter import format_llama3

# ``CHATS_DIR`` mirrors the constant from ``MythForgeServer``.  Keeping a copy
# here avoids circular imports.
CHATS_DIR = "chats"

STATE_SUFFIX = "_state.json"


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
                print(f"Failed to load state '{path}': {e}")
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
    print(f"[goal_tracker] Saved state to {path}")


def generate_initial_state(call_fn, global_prompt: str, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
    """Use the language model to create an initial state for the character."""
    print("[goal_tracker] Generating initial state ...")
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
    print("[goal_tracker] Initial state prompt:\n" + prompt)
    output = call_fn(prompt, max_tokens=400)
    text = output["choices"][0]["text"].strip()
    print("[goal_tracker] Initial state raw output:\n" + text)
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"Failed to parse initial state: {e}")
        data = {"character_profile": None, "scene_context": None, "goals": []}
    print("[goal_tracker] Parsed initial state:", data)
    return data


def generate_goals(call_fn, character_profile: Dict[str, Any], scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate new goals using the language model."""
    print("[goal_tracker] Generating goals ...")
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
    print("[goal_tracker] Goals prompt:\n" + prompt)
    output = call_fn(prompt, max_tokens=200)
    text = output["choices"][0]["text"].strip()
    print("[goal_tracker] Goals raw output:\n" + text)
    try:
        goals = json.loads(text)
    except Exception as e:
        print(f"Failed to parse goals: {e}")
        goals = []
    print("[goal_tracker] Parsed goals:", goals)
    return goals


def ensure_initial_state(call_fn, chat_id: str, global_prompt: str, first_user: str, first_assistant: str) -> None:
    state = load_state(chat_id)
    if state.get("character_profile") is not None:
        return
    if "**goals**" not in global_prompt.lower():
        return
    data = generate_initial_state(call_fn, global_prompt, first_user, first_assistant)
    state.update(data)
    save_state(chat_id, state)


def check_and_generate_goals(call_fn, chat_id: str) -> None:
    state = load_state(chat_id)
    if state.get("character_profile") and state.get("scene_context") and not state.get("goals"):
        goals = generate_goals(call_fn, state["character_profile"], state["scene_context"])
        state["goals"] = goals
        save_state(chat_id, state)
    else:
        print("[goal_tracker] Goals already present or character state incomplete")


def _load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                print(f"Failed to load json '{path}': {e}")
    return []


def evaluate_goals(call_fn, chat_id: str) -> None:
    """Evaluate progress on current goals based on recent conversation."""
    state = load_state(chat_id)
    goals = state.get("goals") or []
    if not goals:
        print("[goal_tracker] No goals to evaluate")
        return

    trimmed_path = os.path.join(CHATS_DIR, f"{chat_id}_trimmed.json")
    history = _load_json(trimmed_path)
    convo: List[Dict[str, str]] = []
    for m in history[-8:]:  # include recent context
        if m.get("type") == "summary":
            convo.append({"role": "system", "content": f"SUMMARY: {m['content']}"})
        else:
            role = "assistant" if m.get("role") == "bot" else m.get("role", "user")
            convo.append({"role": role, "content": m.get("content", "")})

    instruction = (
        "Evaluate the character's current goals based on the recent conversation. "
        "For each goal decide if it is completed, still in progress, needs new tactics or should be abandoned. "
        "If goals are completed or abandoned you may add new short term goals. "
        "Return JSON with a 'goals' list where each goal has id, description, method and status."
    )

    prompt = format_llama3("", None, convo, instruction)
    print("[goal_tracker] Goal evaluation prompt:\n" + prompt)
    output = call_fn(prompt, max_tokens=300)
    text = output["choices"][0]["text"].strip()
    print("[goal_tracker] Goal evaluation raw output:\n" + text)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("goals"), list):
            completed = state.get("completed_goals", [])
            active = []
            for g in data["goals"]:
                status = str(g.get("status", "")).lower()
                gid = g.get("id")
                if status in ("completed", "abandoned"):
                    completed.append(g)
                    continue
                if gid and any(c.get("id") == gid for c in completed):
                    continue
                active.append(g)

            state["completed_goals"] = completed
            state["goals"] = active

            if (
                len(active) < 3
                and state.get("character_profile")
                and state.get("scene_context")
            ):
                new_goals = generate_goals(
                    call_fn, state["character_profile"], state["scene_context"]
                )

                def duplicate(goal):
                    gid = goal.get("id")
                    return any(a.get("id") == gid for a in active) or any(
                        c.get("id") == gid for c in completed
                    )

                for g in new_goals:
                    if duplicate(g):
                        continue
                    active.append(g)
                    if len(active) >= 3:
                        break
                state["goals"] = active
        else:
            print("[goal_tracker] Invalid goal evaluation format")
    except Exception as e:
        print(f"Failed to parse goal evaluation: {e}")
    state["messages_since_goal_eval"] = 0
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

