from __future__ import annotations

"""Simple background task queue for goal evaluation."""

import threading
from queue import Queue, Empty
from typing import Callable, Any, Tuple

from .model import shutdown_background_llama
from .logger import LOGGER


_TASK_QUEUE: "Queue[Tuple[Callable[..., Any], tuple, dict]]" = Queue()
_WORKER: threading.Thread | None = None


def _worker() -> None:
    """Process tasks until the queue is empty."""

    while True:
        try:
            func, args, kwargs = _TASK_QUEUE.get(timeout=5)
        except Empty:
            break
        try:
            func(*args, **kwargs)
        finally:
            _TASK_QUEUE.task_done()
    shutdown_background_llama()
    LOGGER.log("background", {"event": "worker_exit"})


def _start_worker() -> None:
    """Ensure the background worker thread is running."""

    global _WORKER
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(target=_worker, daemon=True)
        _WORKER.start()


def schedule_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Queue ``func`` for execution in the background."""

    _TASK_QUEUE.put((func, args, kwargs))
    _start_worker()


def has_pending_tasks() -> bool:
    """Return ``True`` if tasks are queued or running."""

    return not _TASK_QUEUE.empty() or (_WORKER is not None and _WORKER.is_alive())
