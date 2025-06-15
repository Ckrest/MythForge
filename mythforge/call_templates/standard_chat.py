from __future__ import annotations

"""Prompt helpers and interactive model utilities for standard chats."""

import subprocess
import threading
from typing import Any, Dict, Iterable, Iterator, List

from typing import TYPE_CHECKING

from ..model import model_launch
from .. import memory
from ..call_core import format_for_model
from ..memory import MEMORY_MANAGER
from ..logger import LOGGER

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..call_core import CallData


_chat_process: subprocess.Popen | None = None
_inactivity_timer: threading.Timer | None = None

INACTIVITY_TIMEOUT_SECONDS = 20 * 60


def chat_running() -> bool:
    """Return ``True`` if the chat model subprocess is active."""

    return _chat_process is not None and _chat_process.poll() is None


def _terminate_chat() -> None:
    """Terminate the running chat subprocess."""

    global _chat_process, _inactivity_timer
    if _chat_process is None:
        return
    try:
        _chat_process.terminate()
        _chat_process.wait(timeout=5)
    except Exception:
        _chat_process.kill()
    _chat_process = None
    _inactivity_timer = None


def _reset_timer() -> None:
    """Restart the inactivity timer."""

    global _inactivity_timer
    if _inactivity_timer is not None:
        _inactivity_timer.cancel()
    _inactivity_timer = threading.Timer(
        INACTIVITY_TIMEOUT_SECONDS, _terminate_chat
    )
    _inactivity_timer.daemon = True
    _inactivity_timer.start()


def _stream_output(stop_token: str) -> Iterator[dict[str, str]]:
    """Yield CLI output lines once the model is ready."""

    assert _chat_process is not None
    assert _chat_process.stdout is not None

    start_marker = "== Running in interactive mode. =="
    end_marker = "<|im_start|>assistant"
    capturing = False

    for line in _chat_process.stdout:
        text = line.rstrip("\n")
        if not capturing:
            if text == start_marker:
                capturing = True
            continue
        if text in (stop_token, end_marker):
            return
        print(line, end="", flush=True)
        yield {"text": text}


# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {"stream": True}


def prep_standard_chat() -> None:
    """Launch the standard chat model if it is not running."""

    global _chat_process
    if chat_running():
        _reset_timer()
        return

    cmd = model_launch(**MODEL_LAUNCH_OVERRIDE)
    _chat_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _reset_timer()


def send_prompt(system_text: str, user_text: str, *, stream: bool = False):
    """Send prompts to the interactive model process."""

    prep_standard_chat()
    _reset_timer()
    formatted_prompt = format_for_model(system_text, user_text)
    LOGGER.log("model_calls", formatted_prompt)
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None

    _chat_process.stdin.write(formatted_prompt + "\n")
    _chat_process.stdin.flush()

    if stream:
        return _stream_output("<|im_end|>")
    output: list[str] = []
    for chunk in _stream_output("<|im_end|>"):
        output.append(chunk["text"])
    return {"text": "".join(output).strip()}


def chat(chat_id: str, user_text: str) -> str:
    """Return a model reply for ``user_text`` in ``chat_id``.

    This now logs the ``user_text`` so every input sent to the model is
    captured in the server logs.
    """

    if not chat_running():
        prep_standard_chat()
    else:
        _reset_timer()
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None
    LOGGER.log("model_calls", user_text)
    _chat_process.stdin.write(user_text + "\n")
    _chat_process.stdin.flush()
    output: list[str] = []
    for chunk in _stream_output("<|im_end|>"):
        output.append(chunk["text"])
    return "".join(output).strip()


def send_cli_command(command: str, *, stream: bool = False):
    """Send ``command`` directly to the interactive CLI process."""

    prep_standard_chat()
    _reset_timer()
    formatted_prompt = command
    LOGGER.log("model_calls", formatted_prompt)
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None

    _chat_process.stdin.write(formatted_prompt + "\n")
    _chat_process.stdin.flush()

    if stream:
        return _stream_output("<|im_end|>")

    output: list[str] = []
    for chunk in _stream_output("<|im_end|>"):
        output.append(chunk["text"])
    return {"text": "".join(output).strip()}


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
    history = MEMORY_MANAGER.load_history(call.chat_id)

    system_text = prepare_system_text(call)
    user_text = prepare_user_text(history)
    LOGGER.log(
        "prepared_prompts",
        {
            "call_type": call.call_type,
            "system_text": system_text,
            "user_text": user_text,
        },
    )
    return system_text, user_text


def prompt(system_text: str, user_text: str) -> tuple[str, str]:
    """Return ``system_text`` and ``user_text`` for the model."""

    return system_text, user_text


def response(result: Iterable[dict]) -> Iterator[str]:
    """Yield parsed output for streaming responses."""

    from ..call_core import stream_parsed

    return stream_parsed(result)
