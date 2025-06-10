"""Simple logging utilities for Myth Forge server."""

from __future__ import annotations

import inspect as _inspect
import json
import os
from datetime import datetime

LOG_DIR = "server_logs"


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
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
