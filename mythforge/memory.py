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
    get_global_prompt_content,
    save_global_prompt,
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

    def load_history(self, chat_name: str) -> List[Dict[str, Any]]:
        """Return history list for ``chat_name``."""
        MEMORY_MANAGER.update_paths(chat_name=chat_name)
        return load_json(chat_file(chat_name, "full.json"))

    def _save(self, chat_name: str, history: List[Dict[str, Any]]) -> None:
        ensure_chat_dir(chat_name)
        MEMORY_MANAGER.update_paths(chat_name=chat_name)
        save_json(chat_file(chat_name, "full.json"), history)

    def save_history(
        self, chat_name: str, history: List[Dict[str, Any]]
    ) -> None:
        """Persist ``history`` for ``chat_name``."""

        self._save(chat_name, history)

    def append_message(self, chat_name: str, role: str, content: str) -> None:
        """Add ``content`` to ``chat_name`` if not blank."""

        if not content.strip():
            return

        MEMORY_MANAGER.update_paths(chat_name=chat_name)
        history = self.load_history(chat_name)
        history.append({"role": role, "content": content})
        self._save(chat_name, history)

    def edit_message(self, chat_name: str, index: int, content: str) -> None:
        MEMORY_MANAGER.update_paths(chat_name=chat_name)
        history = self.load_history(chat_name)
        if 0 <= index < len(history):
            history[index]["content"] = content
            self._save(chat_name, history)

    def delete_message(self, chat_name: str, index: int) -> None:
        MEMORY_MANAGER.update_paths(chat_name=chat_name)
        history = self.load_history(chat_name)
        if 0 <= index < len(history):
            history.pop(index)
            self._save(chat_name, history)

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
        self.chat_name: str = ""
        self.global_prompt_name: str = ""
        self.goals_active: bool = False

    def update_paths(
        self, chat_name: str | None = None, prompt_name: str | None = None
    ) -> None:
        """Update the stored chat and prompt names."""

        if chat_name is not None:
            self.chat_name = chat_name
        if prompt_name is not None:
            self.global_prompt_name = prompt_name

    # ------------------------------------------------------------------
    # Goal helpers
    # ------------------------------------------------------------------
    def toggle_goals(self, enabled: bool) -> None:
        self.goals_active = enabled

    def load_goals(self, chat_name: str | None = None) -> GoalsData:
        name = chat_name or self.chat_name
        self.update_paths(chat_name=name)
        path = goals_path(name)
        if not os.path.exists(path):
            self.toggle_goals(False)
            return GoalsData()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.toggle_goals(False)
            return GoalsData()
        self.toggle_goals(True)
        return GoalsData(
            character=str(data.get("character", "")),
            setting=str(data.get("setting", "")),
            active_goals=list(data.get("in_progress", [])),
            deactive_goals=list(data.get("completed", [])),
        )

    def save_goals(self, chat_name: str, data: Dict[str, Any]) -> None:
        ensure_chat_dir(chat_name)
        self.update_paths(chat_name=chat_name)
        obj = {
            "character": data.get("character", ""),
            "setting": data.get("setting", ""),
            "in_progress": data.get("in_progress", []),
            "completed": data.get("completed", []),
        }
        with open(goals_path(chat_name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        self.toggle_goals(True)

    @property
    def global_prompt(self) -> str:
        return self.get_global_prompt()

    def get_global_prompt(self, prompt_name: str | None = None) -> str:
        name = prompt_name or self.global_prompt_name
        if not name:
            return ""
        return get_global_prompt_content(name) or ""

    @property
    def goals_data(self) -> GoalsData:
        return self.load_goals(self.chat_name)


MEMORY_MANAGER = MemoryManager()
MEMORY = MEMORY_MANAGER


def set_global_prompt(prompt: str) -> None:
    """Set ``prompt`` as the active global prompt."""

    name = "current_prompt"
    save_global_prompt({"name": name, "content": prompt})
    MEMORY_MANAGER.update_paths(prompt_name=name)


def update_model_settings(settings: Dict[str, Any]) -> None:
    """Compatibility wrapper for updating model settings."""

    MEMORY_MANAGER.model_settings.update(settings)


def initialize(manager: MemoryManager = MEMORY_MANAGER) -> None:
    """Populate ``manager`` with default values."""

    manager.model_settings = model.MODEL_SETTINGS.copy()
    manager.goals_active = False
    manager.update_paths(chat_name="", prompt_name="")
    prompts = load_global_prompts()
    if prompts:
        manager.update_paths(prompt_name=prompts[0]["name"])
