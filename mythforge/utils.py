"""Utility helpers for Myth Forge."""

from __future__ import annotations

import inspect as _inspect
import json
import os
from datetime import datetime
from typing import Any, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT_DIR, "server_logs")
CHATS_DIR = os.path.join(ROOT_DIR, "chats")
GLOBAL_PROMPTS_DIR = os.path.join(ROOT_DIR, "global_prompts")
VERBOSE_MODE = False


def myth_log(*args, **kwargs) -> None:
    """Log the caller name with the provided arguments."""

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


def _prompt_path(name: str) -> str:
    """Return the filesystem path for a prompt ``name``."""

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return os.path.join(GLOBAL_PROMPTS_DIR, f"{safe}.json")


def load_global_prompts() -> List[dict[str, str]]:
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    prompts: List[dict[str, str]] = []
    for fname in sorted(os.listdir(GLOBAL_PROMPTS_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(GLOBAL_PROMPTS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load prompt '{fname}': {e}")
            continue
        if isinstance(data, dict) and "name" in data and "content" in data:
            prompts.append({"name": data["name"], "content": data["content"]})
        else:
            print(f"Ignoring invalid global prompt file: {fname}")
    return prompts


def list_prompt_names() -> List[str]:
    """Return only the names of available global prompts."""

    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    names: List[str] = []
    for fname in sorted(os.listdir(GLOBAL_PROMPTS_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(GLOBAL_PROMPTS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load prompt '{fname}': {e}")
            continue
        if isinstance(data, dict) and "name" in data:
            names.append(data["name"])
    return names


def get_global_prompt_content(name: str) -> str | None:
    """Return the content string for a prompt ``name`` if it exists."""

    path = _prompt_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("content") if isinstance(data, dict) else None
    except Exception as e:
        print(f"Failed to load prompt '{name}': {e}")
        return None


def save_global_prompt(prompt: dict[str, str]) -> None:
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    path = _prompt_path(prompt["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"name": prompt["name"], "content": prompt["content"]},
            f,
            indent=2,
            ensure_ascii=False,
        )


def delete_global_prompt(name: str) -> None:
    path = _prompt_path(name)
    if os.path.exists(path):
        os.remove(path)
