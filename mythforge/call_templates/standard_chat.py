from __future__ import annotations

"""Prompt helpers and interactive model utilities for standard chats."""

import subprocess
from typing import Any, Dict, Iterable, Iterator, List

from typing import TYPE_CHECKING

from ..model import model_launch
from .. import memory
from ..utils import myth_log
from ..call_core import format_for_model

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..call_core import CallData


_chat_process: subprocess.Popen | None = None


def chat_running() -> bool:
    """Return ``True`` if the chat model subprocess is active."""

    return _chat_process is not None and _chat_process.poll() is None


# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {"stream": False}


def prep_standard_chat() -> None:
    """Launch the standard chat model if it is not running."""

    global _chat_process
    if chat_running():
        return

    cmd = model_launch(**MODEL_LAUNCH_OVERRIDE)
    myth_log(" ".join(cmd))
    _chat_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def send_prompt(system_text: str, user_text: str, *, stream: bool = False):
    """Send prompts to the interactive model process."""

    prep_standard_chat()
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None

    _chat_process.stdin.write(system_text + user_text + "\n")
    _chat_process.stdin.flush()

    if stream:

        def _stream() -> Iterator[dict[str, str]]:
            for line in _chat_process.stdout:
                yield {"text": line.rstrip()}

        return _stream()

    line = _chat_process.stdout.readline()
    return {"text": line.rstrip()}


def chat(chat_id: str, user_text: str) -> str:
    """Return a model reply for ``user_text`` in ``chat_id``."""

    if not chat_running():
        prep_standard_chat()

    formatted = format_for_model(chat_id, user_text)
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None
    _chat_process.stdin.write(formatted + "\n")
    _chat_process.stdin.flush()
    output: list[str] = []
    for line in _chat_process.stdout:
        if line.strip() == "<|endoftext|>":
            break
        output.append(line)
    return "".join(output).strip()


def send_cli_command(command: str, *, stream: bool = False):
    """Send ``command`` directly to the interactive CLI process."""

    prep_standard_chat()
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None

    _chat_process.stdin.write(command + "\n")
    _chat_process.stdin.flush()

    if stream:

        def _stream() -> Iterator[dict[str, str]]:
            for line in _chat_process.stdout:
                yield {"text": line.rstrip()}

        return _stream()

    line = _chat_process.stdout.readline()
    return {"text": line.rstrip()}


def prepare_system_text(call: CallData) -> str:
    """Return the system prompt text for ``call``."""

    if not call.global_prompt:
        from ..call_core import _default_global_prompt

        call.global_prompt = (
            memory.MEMORY.global_prompt or _default_global_prompt()
        )

    parts = [call.global_prompt]
    goals = memory.MEMORY.goals_data
    if memory.MEMORY.goals_active:
        if goals.character:
            parts.append(goals.character)
        if goals.setting:
            parts.append(goals.setting)
        if goals.active_goals:
            joined = ", ".join(str(g) for g in goals.active_goals)
            parts.append(f"Active Goals: {joined}")
        if goals.deactive_goals:
            joined = ", ".join(str(g) for g in goals.deactive_goals)
            parts.append(f"Completed Goals: {joined}")
    return "\n".join(p for p in parts if p)


def prepare_user_text(history: List[Dict[str, Any]]) -> str:
    """Return the user prompt text from ``history``."""

    return "\n".join(m.get("content", "") for m in history)


def prepare(call: CallData) -> tuple[str, str]:
    """Return prompts for a standard chat call."""

    from ..call_core import format_for_model
    from ..memory import ChatHistoryService

    history_service = ChatHistoryService()
    history = history_service.load_history(call.chat_id)

    system_text = prepare_system_text(call)
    user_text = prepare_user_text(history)
    combined = format_for_model(system_text, user_text)
    return "", combined


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return ``system_text`` and ``user_text`` for the model."""

    return system_text, user_text


def response(result: Iterable[dict]) -> Iterator[str]:
    """Yield parsed output for streaming responses."""

    from ..call_core import stream_parsed

    return stream_parsed(result)
