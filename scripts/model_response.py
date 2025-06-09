"""Utilities for parsing and cleaning model responses."""

from __future__ import annotations

from typing import Iterable, Iterator


def clean_text(text: str) -> str:
    """Return ``text`` stripped of whitespace and known tokens."""

    cleaned = text.replace("<|eot_id|>", "").strip()
    return cleaned


def parse_response(output: dict) -> str:
    """Extract and clean the text portion from a model call result."""

    text = output.get("choices", [{}])[0].get("text", "")
    return clean_text(text)


def stream_parsed(chunks: Iterable[dict]) -> Iterator[str]:
    """Yield cleaned text from a streaming model call."""

    for chunk in chunks:
        yield clean_text(chunk.get("choices", [{}])[0].get("text", ""))
