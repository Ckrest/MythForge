"""Utility stubs for logging and patching."""

from types import ModuleType
from typing import Callable


def patch_module_functions(module: ModuleType, _name: str) -> None:
    """Dummy patcher that leaves ``module`` untouched."""

    return None


def log_event(event: str, data: dict) -> None:
    """Print a log message for ``event`` with ``data``."""

    print(f"{event}: {data}")
