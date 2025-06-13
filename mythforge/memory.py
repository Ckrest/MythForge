"""Shared in-memory state and file utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import model
from .utils import (
    CHATS_DIR,
    chat_file,
    ensure_chat_dir,
    load_json,
    save_json,
    load_global_prompts,
    goals_path,
)


@dataclass
class GoalsData:
    """Container for goal related information."""

    character: str = ""
    setting: str = ""
    active_goals: List[Any] = field(default_factory=list)
    deactive_goals: List[Any] = field(default_factory=list)
    enabled: bool = False


class ChatHistoryService:
    """Handle loading and saving chat histories."""

    def load_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Return history list for ``chat_id``."""

        return load_json(chat_file(chat_id, "full.json"))

    def _save(self, chat_id: str, history: List[Dict[str, Any]]) -> None:
        ensure_chat_dir(chat_id)
        save_json(chat_file(chat_id, "full.json"), history)

    def save_history(
        self, chat_id: str, history: List[Dict[str, Any]]
    ) -> None:
        """Persist ``history`` for ``chat_id``."""

        self._save(chat_id, history)

    def append_message(self, chat_id: str, role: str, content: str) -> None:
        history = self.load_history(chat_id)
        history.append({"role": role, "content": content})
        self._save(chat_id, history)

    def edit_message(self, chat_id: str, index: int, content: str) -> None:
        history = self.load_history(chat_id)
        if 0 <= index < len(history):
            history[index]["content"] = content
            self._save(chat_id, history)

    def delete_message(self, chat_id: str, index: int) -> None:
        history = self.load_history(chat_id)
        if 0 <= index < len(history):
            history.pop(index)
            self._save(chat_id, history)

    def list_chats(self) -> List[str]:
        os.makedirs(CHATS_DIR, exist_ok=True)
        return [
            d
            for d in os.listdir(CHATS_DIR)
            if os.path.isdir(os.path.join(CHATS_DIR, d))
        ]


class MemoryManager:
    """Centralized state for prompts, goals and model settings."""

    def __init__(self) -> None:
        self.model_settings: Dict[str, Any] = model.MODEL_SETTINGS.copy()
        self.global_prompt: str = ""
        self.goals_data: GoalsData = GoalsData()

    # ------------------------------------------------------------------
    # Goal helpers
    # ------------------------------------------------------------------
    def update_goals(self, data: Dict[str, Any]) -> None:
        self.goals_data.character = str(data.get("character", ""))
        self.goals_data.setting = str(data.get("setting", ""))
        self.goals_data.active_goals = list(data.get("active_goals", []))
        self.goals_data.deactive_goals = list(data.get("deactive_goals", []))

    def toggle_goals(self, enabled: bool) -> None:
        self.goals_data.enabled = enabled

    def load_goals(self, chat_id: str) -> None:
        path = goals_path(chat_id)
        if not os.path.exists(path):
            self.goals_data = GoalsData(enabled=False)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.update_goals(
                {
                    "character": data.get("character", ""),
                    "setting": data.get("setting", ""),
                    "active_goals": data.get("in_progress", []),
                    "deactive_goals": data.get("completed", []),
                }
            )
            self.goals_data.enabled = True
        except Exception:
            self.goals_data = GoalsData(enabled=False)

    def save_goals(self, chat_id: str, data: Dict[str, Any]) -> None:
        ensure_chat_dir(chat_id)
        obj = {
            "character": data.get("character", ""),
            "setting": data.get("setting", ""),
            "in_progress": data.get("in_progress", []),
            "completed": data.get("completed", []),
        }
        with open(goals_path(chat_id), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        self.toggle_goals(True)
        self.update_goals(
            {
                "character": obj.get("character", ""),
                "setting": obj.get("setting", ""),
                "active_goals": obj.get("in_progress", []),
                "deactive_goals": obj.get("completed", []),
            }
        )


MEMORY_MANAGER = MemoryManager()
MEMORY = MEMORY_MANAGER


def set_global_prompt(prompt: str) -> None:
    """Compatibility wrapper to set the global prompt."""

    MEMORY_MANAGER.global_prompt = prompt


def update_model_settings(settings: Dict[str, Any]) -> None:
    """Compatibility wrapper for updating model settings."""

    MEMORY_MANAGER.model_settings.update(settings)


def initialize(manager: MemoryManager = MEMORY_MANAGER) -> None:
    """Populate ``manager`` with default values."""

    manager.model_settings = model.MODEL_SETTINGS.copy()
    manager.global_prompt = ""
    prompts = load_global_prompts()
    if prompts:
        manager.global_prompt = prompts[0]["content"]
