from __future__ import annotations

"""Simple JSON file logger used across the project."""

from typing import Any

from .memory import MEMORY_MANAGER, _load_json, _save_json


class LoggerManager:
    """Persist log events using :class:`MemoryManager`."""

    def __init__(self, memory_manager=MEMORY_MANAGER) -> None:
        """Configure paths and storage for logging."""

        self.memory_manager = memory_manager

    def _path(self, event_type: str) -> str:
        """Resolve path to the ``event_type`` log file."""

        return self.memory_manager.get_log_path(event_type)

    def log(self, event_type: str, payload: Any) -> None:
        """Append a new log entry to ``event_type``."""
        path = self._path(event_type)
        data = _load_json(path)
        if not isinstance(data, list):
            data = []
        data.insert(0, payload)
        _save_json(path, data)

    def log_error(self, err: Exception) -> None:
        """Record exception information in the error log."""

        self.log("errors", {"error": str(err)})


LOGGER = LoggerManager(MEMORY_MANAGER)
