from __future__ import annotations

"""Utilities for parsing model responses."""

from typing import Any, Iterable, Iterator, List, Dict

import json
import re

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

    def parse(self) -> Iterator[str]:
        """Yield text fragments extracted from ``self.raw``."""

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
                    if isinstance(chunk, dict):
                        if "text" in chunk:
                            yield str(chunk["text"])
                        elif "choices" in chunk:
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            if "content" in delta:
                                yield str(delta["content"])
                            else:
                                message = choice.get("message", {})
                                yield str(message.get("content", ""))
                        else:
                            yield str(chunk)
                    else:
                        yield str(chunk)

            return _iter()

        if isinstance(self.raw, dict):
            if "text" in self.raw:
                text = str(self.raw["text"])
            elif "choices" in self.raw:
                choice = self.raw.get("choices", [{}])[0]
                message = choice.get("message", {})
                text = str(message.get("content", ""))
            else:
                text = str(self.raw)
        else:
            text = str(self.raw)

        def _single() -> Iterator[str]:
            yield text

        return _single()


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


def _parse_goal_status_from_response(text: str) -> List[Dict[str, Any]]:
    """Extract goal status updates from ``text``."""

    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
        except Exception:
            return []

    goals = parsed.get("goals", []) if isinstance(parsed, dict) else []
    if not isinstance(goals, list):
        return []

    results: List[Dict[str, Any]] = []
    for g in goals:
        desc = str(g.get("description", "")).strip()
        status = str(g.get("status", "")).strip()
        if desc and status:
            results.append({"description": desc, "status": status})

    return results


def _parse_duplicates_from_response(text: str) -> List[List[int]]:
    """Extract duplicate goal index pairs from ``text``."""

    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
        except Exception:
            return []

    dup_list = parsed.get("duplicates", []) if isinstance(parsed, dict) else []
    results: List[List[int]] = []
    if isinstance(dup_list, list):
        for pair in dup_list:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(isinstance(i, int) for i in pair)
            ):
                results.append([int(pair[0]), int(pair[1])])

    return results
