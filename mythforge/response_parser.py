from __future__ import annotations

"""Utilities for parsing model responses."""

from typing import Any, Iterable, Iterator

from .logger import LOGGER


class ResponseParser:
    """Parse text or streaming model output."""

    def __init__(self) -> None:
        self.raw: Any = None

    def load(self, raw: Any) -> "ResponseParser":
        """Store ``raw`` output to be parsed."""

        LOGGER.log(
            "chat_flow",
            {
                "function": "ResponseParser.load",
                "raw": str(raw),
            },
        )
        self.raw = raw
        return self

    def parse(self) -> Any:
        """Return parsed output from :meth:`load`."""

        LOGGER.log(
            "chat_flow",
            {
                "function": "ResponseParser.parse",
                "raw": str(self.raw),
            },
        )

        if isinstance(self.raw, Iterable) and not isinstance(
            self.raw, (str, bytes, dict)
        ):

            def _iter() -> Iterator[str]:
                for chunk in self.raw:
                    if isinstance(chunk, dict) and "text" in chunk:
                        yield str(chunk["text"])
                    else:
                        yield str(chunk)

            return _iter()
        if isinstance(self.raw, dict) and "text" in self.raw:
            return str(self.raw["text"])
        return str(self.raw)
