"""Centralized file-backed storage and helper functions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List



def _load_json(path: str) -> list[Any] | dict[str, Any] | list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except Exception as e:  # pragma: no cover - best effort
            try:
                from .logger import LOGGER
            except Exception:
                LOGGER = None
            if LOGGER is not None:
                LOGGER.log_error(e)
    return []


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class GoalsData:
    """Container for goal related information."""

    character: str = ""
    setting: str = ""
    active_goals: List[Any] = field(default_factory=list)
    deactive_goals: List[Any] = field(default_factory=list)
    enabled: bool = False


class MemoryManager:
    """Centralized state for prompts, goals and model settings."""

    def __init__(self, root_dir: str | None = None) -> None:
        self.root_dir = root_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.chats_dir = os.path.join(self.root_dir, "chats")
        self.prompts_dir = os.path.join(self.root_dir, "global_prompts")
        self.logs_dir = os.path.join(self.root_dir, "server_logs")
        self.settings_path = os.path.join(self.root_dir, "model_settings.json")

        self.model_settings: Dict[str, Any] = self.load_settings()
        self.chat_name: str = ""
        self.global_prompt_name: str = ""
        self.goals_active: bool = False

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _chat_file(self, chat_id: str, filename: str) -> str:
        return os.path.join(self.chats_dir, chat_id, filename)

    def _ensure_chat_dir(self, chat_id: str) -> str:
        path = os.path.join(self.chats_dir, chat_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _prompt_path(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        return os.path.join(self.prompts_dir, f"{safe}.json")

    def _goals_path(self, chat_id: str) -> str:
        return self._chat_file(chat_id, "goals.json")

    # ------------------------------------------------------------------
    # Basic JSON file helpers
    # ------------------------------------------------------------------
    def _read_json(self, path: str) -> Any:
        return _load_json(path)

    def _write_json(self, path: str, data: Any) -> None:
        _save_json(path, data)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def load_history(self, chat_id: str) -> List[Dict[str, Any]]:
        self.update_paths(chat_name=chat_id)
        return list(self._read_json(self._chat_file(chat_id, "full.json")))

    def save_history(self, chat_id: str, history: List[Dict[str, Any]]) -> None:
        self._ensure_chat_dir(chat_id)
        self.update_paths(chat_name=chat_id)
        self._write_json(self._chat_file(chat_id, "full.json"), history)


    def list_chats(self) -> List[str]:
        os.makedirs(self.chats_dir, exist_ok=True)
        return [
            d
            for d in os.listdir(self.chats_dir)
            if os.path.isdir(os.path.join(self.chats_dir, d))
        ]

    def delete_chat(self, chat_id: str) -> None:
        chat_dir = os.path.join(self.chats_dir, chat_id)
        if os.path.isdir(chat_dir):
            for fname in os.listdir(chat_dir):
                os.remove(os.path.join(chat_dir, fname))
            os.rmdir(chat_dir)

    def rename_chat(self, old_id: str, new_id: str) -> None:
        old_dir = os.path.join(self.chats_dir, old_id)
        new_dir = os.path.join(self.chats_dir, new_id)
        if os.path.isdir(old_dir) and not os.path.exists(new_dir):
            os.rename(old_dir, new_dir)

    # ------------------------------------------------------------------
    # Goal state helpers
    # ------------------------------------------------------------------
    def load_goal_state(self, chat_id: str) -> Dict[str, Any]:
        path = self._chat_file(chat_id, "goal_state.json")
        if os.path.exists(path):
            data = self._read_json(path)
            if isinstance(data, dict):
                return data
        return {
            "goals": [],
            "completed_goals": [],
            "messages_since_goal_eval": 0,
        }

    def save_goal_state(self, chat_id: str, state: Dict[str, Any]) -> None:
        self._write_json(self._chat_file(chat_id, "goal_state.json"), state)

    def disable_goals(self, chat_id: str) -> None:
        path = self._goals_path(chat_id)
        disabled = self._chat_file(chat_id, "goals_disabled.json")
        if os.path.exists(path):
            os.rename(path, disabled)
        self.load_goals(chat_id)

    def enable_goals(self, chat_id: str) -> None:
        path = self._goals_path(chat_id)
        disabled = self._chat_file(chat_id, "goals_disabled.json")
        if os.path.exists(disabled):
            os.rename(disabled, path)
        self.load_goals(chat_id)

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------
    def load_settings(self) -> Dict[str, Any]:
        data = self._read_json(self.settings_path)
        return data if isinstance(data, dict) else {}

    def save_settings(self, full_payload: Dict[str, Any]) -> None:
        """Persist ``full_payload`` and update :attr:`model_settings`."""

        self._write_json(self.settings_path, full_payload)
        self.model_settings.update(full_payload)

    def update_settings(
        self, delta: Dict[str, Any], save: bool = True
    ) -> Dict[str, Any]:
        """Merge ``delta`` into :attr:`model_settings` and optionally save."""

        from . import model

        self.model_settings.update(delta)
        model.MODEL_SETTINGS.update(delta)
        for key in ("temp", "top_k", "top_p", "min_p", "repeat_penalty"):
            if key in delta:
                model.GENERATION_CONFIG[key] = model.MODEL_SETTINGS[key]
        if "max_tokens" in delta:
            model.DEFAULT_MAX_TOKENS = model.MODEL_SETTINGS.get(
                "max_tokens", model.DEFAULT_MAX_TOKENS
            )
        if save:
            self.save_settings(self.model_settings)
        return self.model_settings

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def get_log_path(self, event_type: str) -> str:
        """Return path for the ``event_type`` log file."""

        return os.path.join(self.logs_dir, f"{event_type}.json")

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
        path = self._goals_path(name)
        if not os.path.exists(path):
            self.toggle_goals(False)
            return GoalsData()
        try:
            data = self._read_json(path)
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
        self._ensure_chat_dir(chat_name)
        self.update_paths(chat_name=chat_name)
        obj = {
            "character": data.get("character", ""),
            "setting": data.get("setting", ""),
            "in_progress": data.get("in_progress", []),
            "completed": data.get("completed", []),
        }
        self._write_json(self._goals_path(chat_name), obj)
        self.toggle_goals(True)

    @property
    def global_prompt(self) -> str:
        return self.get_global_prompt()

    def get_global_prompt(self, prompt_name: str | None = None) -> str:
        name = prompt_name or self.global_prompt_name
        if not name:
            return ""
        path = self._prompt_path(name)
        data = self._read_json(path)
        return str(data.get("content", "")) if isinstance(data, dict) else ""

    def set_global_prompt(self, name: str, content: str) -> None:
        os.makedirs(self.prompts_dir, exist_ok=True)
        self._write_json(self._prompt_path(name), {"name": name, "content": content})
        self.update_paths(prompt_name=name)

    def delete_global_prompt(self, name: str) -> None:
        path = self._prompt_path(name)
        if os.path.exists(path):
            os.remove(path)

    def rename_global_prompt(self, old: str, new: str) -> None:
        old_path = self._prompt_path(old)
        new_path = self._prompt_path(new)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)

    def load_global_prompts(self) -> List[dict[str, str]]:
        os.makedirs(self.prompts_dir, exist_ok=True)
        prompts: List[dict[str, str]] = []
        for fname in sorted(os.listdir(self.prompts_dir)):
            if not fname.lower().endswith(".json"):
                continue
            data = self._read_json(os.path.join(self.prompts_dir, fname))
            if isinstance(data, dict) and "name" in data and "content" in data:
                prompts.append({"name": data["name"], "content": data["content"]})
        return prompts

    def list_prompt_names(self) -> List[str]:
        return [p["name"] for p in self.load_global_prompts()]

    @property
    def goals_data(self) -> GoalsData:
        return self.load_goals(self.chat_name)


MEMORY_MANAGER = MemoryManager()
MEMORY = MEMORY_MANAGER


def set_global_prompt(prompt: str) -> None:
    """Set ``prompt`` as the active global prompt."""
    MEMORY_MANAGER.set_global_prompt("current_prompt", prompt)


def initialize(manager: MemoryManager = MEMORY_MANAGER) -> None:
    """Populate ``manager`` with default values."""
    from . import model

    manager.model_settings = manager.load_settings() or model.MODEL_SETTINGS.copy()
    manager.goals_active = False
    manager.update_paths(chat_name="", prompt_name="")
    prompts = manager.load_global_prompts()
    if prompts:
        manager.update_paths(prompt_name=prompts[0]["name"])
