"""Utility stubs for logging."""


def log_event(event: str, data: dict) -> None:
    """Print a log message for ``event`` with ``data``."""

    print(f"{event}: {data}")
