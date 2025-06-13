"""Shared in-memory state for Myth Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import model
import json
import os

from .utils import load_global_prompts, goals_path


@dataclass
class GoalsData:
    """Container for goal related information."""

    character: str = ""
    setting: str = ""
    active_goals: List[Any] = field(default_factory=list)
    deactive_goals: List[Any] = field(default_factory=list)
    enabled: bool = False


@dataclass
class ServerMemory:
    """Overall in-memory server state."""

    model: Dict[str, Any] = field(default_factory=dict)
    goals: Dict[str, Any] = field(default_factory=dict)
    global_prompt: str = ""
    goals_data: GoalsData = field(default_factory=GoalsData)


MEMORY = ServerMemory()


def initialize() -> None:
    """Populate :data:`MEMORY` with defaults."""

    MEMORY.model = model.MODEL_SETTINGS.copy()
    MEMORY.goals = {
        "goal_refresh_rate": model.MODEL_SETTINGS.get("goal_refresh_rate", 1),
        "goal_limit": model.MODEL_SETTINGS.get("goal_limit", 3),
        "goal_impulse": model.MODEL_SETTINGS.get("goal_impulse", 2),
        "new_goal_bias": model.MODEL_SETTINGS.get("new_goal_bias", 2),
    }
    prompts = load_global_prompts()
    MEMORY.global_prompt = prompts[0]["content"] if prompts else ""


def update_model_settings(settings: Dict[str, Any]) -> None:
    """Update model settings portion of memory."""

    MEMORY.model.update(settings)


def set_global_prompt(prompt: str) -> None:
    """Update the cached global prompt."""

    MEMORY.global_prompt = prompt


def set_goals_enabled(enabled: bool) -> None:
    """Toggle whether goals are active."""

    MEMORY.goals_data.enabled = enabled


def update_goals(data: Dict[str, Any]) -> None:
    """Update all goal related fields from ``data``."""

    MEMORY.goals_data.character = str(data.get("character", ""))
    MEMORY.goals_data.setting = str(data.get("setting", ""))
    MEMORY.goals_data.active_goals = list(data.get("active_goals", []))
    MEMORY.goals_data.deactive_goals = list(data.get("deactive_goals", []))


def load_goals(chat_id: str) -> None:
    """Populate goal related memory from ``chat_id``'s file."""

    path = goals_path(chat_id)
    if not os.path.exists(path):
        MEMORY.goals_data = GoalsData(enabled=False)
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        update_goals(
            {
                "character": data.get("character", ""),
                "setting": data.get("setting", ""),
                "active_goals": data.get("in_progress", []),
                "deactive_goals": data.get("completed", []),
            }
        )
        MEMORY.goals_data.enabled = True
    except Exception:
        MEMORY.goals_data = GoalsData(enabled=False)
