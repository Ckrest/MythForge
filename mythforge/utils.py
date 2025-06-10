"""Utility helpers for Myth Forge."""

from __future__ import annotations

import inspect as _inspect
import json
import os
from datetime import datetime
from typing import Any, List

LOG_DIR = "server_logs"
CHATS_DIR = "chats"


def myth_log(*args, **kwargs) -> None:
    """Log caller name with supplied arguments."""

    now = datetime.utcnow()
    caller = "unknown"
    frame = _inspect.currentframe()
    if frame and frame.f_back:
        caller = frame.f_back.f_code.co_name

    entry = {
        "timestamp": now.isoformat(),
        "caller": caller,
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in kwargs.items()},
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{now.date()}.log")

    lines = [f"[{entry['timestamp']}] {entry['caller']}"]
    lines.extend(entry["args"])
    for k, v in entry["kwargs"].items():
        lines.append(f"{k}: {v}")
    lines.append("")

    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def load_json(path: str) -> List[Any]:
    """Return JSON data from ``path`` or an empty list."""

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception as e:  # pragma: no cover - best effort
            print(f"Failed to load JSON from '{path}': {e}")
    return []


def save_json(path: str, data: Any) -> None:
    """Write ``data`` to ``path`` as JSON."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_text_file(path: str) -> str:
    """Return text loaded from ``path`` if it exists."""

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:  # pragma: no cover - best effort
            print(f"Failed to read text from '{path}': {e}")
    return ""


def write_text_file(path: str, text: str) -> None:
    """Write ``text`` to ``path`` creating parents as needed."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def chat_file(chat_id: str, filename: str) -> str:
    """Return the path for ``filename`` within ``chat_id``'s directory."""

    return os.path.join(CHATS_DIR, chat_id, filename)


def ensure_chat_dir(chat_id: str) -> str:
    """Create and return the directory path for ``chat_id``."""

    path = os.path.join(CHATS_DIR, chat_id)
    os.makedirs(path, exist_ok=True)
    return path


def goals_path(chat_id: str) -> str:
    """Return the path to ``chat_id``'s goals JSON file."""

    return chat_file(chat_id, "goals.json")


def goals_exists(chat_id: str) -> bool:
    """Return ``True`` if goals are enabled for ``chat_id``."""

    path = goals_path(chat_id)
    exists = os.path.exists(path)
    myth_log("goals_check", chat_id=chat_id, exists=exists)
    return exists
