from __future__ import annotations

"""Utilities for parsing model responses."""

from typing import Any, Iterable, Iterator

from .logger import LOGGER


class ResponseParser:
    """Parse text or streaming model output."""

    def __init__(self) -> None:
        """Initialize empty parser state."""

        self.raw: Any = None

    def load(self, raw: Any) -> "ResponseParser":
        """Load ``raw`` model output for later parsing."""

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
        """Extract structured data from ``self.raw``."""

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

def _parse_goals_from_response(text: str) -> List[Dict[str, Any]]:
    """Extract individual goal entries from ``text``."""

    try:

        # Naive direct parse first
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "goals" not in parsed:
            raise ValueError("Top-level object is not a dict with 'goals'")
        goals = parsed["goals"]
        if not isinstance(goals, list):
            raise ValueError("'goals' is not a list")
    except Exception as e:

        # Fallback regex to extract JSON object manually
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in text")

            parsed = json.loads(match.group())
            goals = parsed.get("goals", [])
        except Exception:
            return []

    filtered = []
    for i, g in enumerate(goals):
        desc = g.get("description", "").strip()
        importance = g.get("importance", None)
        if not desc or not isinstance(importance, int):
            continue
        filtered.append(
            {
                "id": str(i + 1),
                "description": desc,
                "importance": importance,
                "status": "in progress",
            }
        )
    return filtered