from __future__ import annotations

"""Prompt helpers and interactive model utilities for standard chats."""

import subprocess
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List

from typing import TYPE_CHECKING

from ..model import model_launch, MODEL_LAUNCH_PARAMS
from .. import memory
from ..utils import myth_log

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..call_core import CallData


_chat_process: subprocess.Popen | None = None
_last_used: float = 0.0
_lock = threading.Lock()
_TIMEOUT = 20 * 60  # 20 minutes


def chat_running() -> bool:
    """Return ``True`` if the chat model subprocess is active."""

    with _lock:
        return _chat_process is not None and _chat_process.poll() is None


# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "stream": True,  # this is just a placeholder. default is also true
}


def _watchdog() -> None:
    """Monitor the running model process and terminate on timeout."""

    global _chat_process
    while True:
        time.sleep(120)
        with _lock:
            if not _chat_process:
                return
            if time.time() - _last_used < _TIMEOUT:
                continue
            proc = _chat_process
            _chat_process = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        return


def prep_standard_chat() -> None:
    """Start the standard chat model in interactive mode if needed."""

    global _chat_process, _last_used
    with _lock:
        alive = _chat_process and _chat_process.poll() is None
        if alive:
            _last_used = time.time()
            return

        args = model_launch(**MODEL_LAUNCH_OVERRIDE)
        myth_log(" ".join(args))
        popen_args = MODEL_LAUNCH_PARAMS.copy()
        popen_args["stdin"] = subprocess.PIPE
        popen_args["stdout"] = subprocess.PIPE
        _chat_process = subprocess.Popen(args, **popen_args)
        _last_used = time.time()
        threading.Thread(target=_watchdog, daemon=True).start()


def send_prompt(system_text: str, user_text: str, *, stream: bool = False):
    """Send prompts to the interactive model process."""

    prep_standard_chat()
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None
    from ..call_core import format_for_model

    with _lock:
        _chat_process.stdin.write(
            format_for_model(system_text, user_text) + "\n"
        )
        _chat_process.stdin.flush()
        global _last_used
        _last_used = time.time()

    if stream:

        def _stream() -> Iterator[dict[str, str]]:
            for line in _chat_process.stdout:
                yield {"text": line.rstrip()}

        return _stream()

    line = _chat_process.stdout.readline()
    return {"text": line.rstrip()}


def send_cli_command(command: str, *, stream: bool = False):
    """Send ``command`` directly to the interactive CLI process."""

    prep_standard_chat()
    assert _chat_process is not None
    assert _chat_process.stdin is not None
    assert _chat_process.stdout is not None

    with _lock:
        _chat_process.stdin.write(command + "\n")
        _chat_process.stdin.flush()
        global _last_used
        _last_used = time.time()

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
    if goals.enabled:
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
