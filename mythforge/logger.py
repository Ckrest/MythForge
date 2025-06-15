from __future__ import annotations

"""Simple JSON file logger used across the project."""

from typing import Any
import os

from .memory import MEMORY_MANAGER, _load_json, _save_json


class LoggerManager:
    """Persist log events under ``root_dir``."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def _path(self, name: str) -> str:
        return os.path.join(self.root_dir, f"{name}.json")

    def log(self, event_type: str, payload: Any) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        path = self._path(event_type)
        data = _load_json(path)
        if not isinstance(data, list):
            data = []
        data.insert(0, payload)
        _save_json(path, data)

    def log_error(self, err: Exception) -> None:
        self.log("errors", {"error": str(err)})


LOGGER = LoggerManager(MEMORY_MANAGER.logs_dir)
