#!/usr/bin/env python3
"""Airoboros prompt formatter.

This module converts chat history and metadata into the simplified
Airoboros format described in the project instructions.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional


def _strip_junk(text: str) -> str:
    """Remove LM Studio tokens and metadata lines."""
    cleaned = re.sub(r"<\|start_header_id\>|<\|end_header_id\>|<\|eot_id\>", "", text)
    cleaned = re.sub(r"^Cutting Knowledge Date.*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^Today Date.*\n?", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def format_airoboros(
    global_prompt: str,
    random_injection: str,
    summaries: Optional[List[str]],
    history: List[Dict[str, str]],
    instruction: str,
    assistant_name: str = "assistant",
) -> str:
    """Return the conversation formatted for Airoboros."""

    lines = [
        "BEGININPUT",
        "BEGINCONTEXT",
        "[SYSTEM] This is law:",
        f"{global_prompt}",
    ]

    if random_injection:
        lines.append(f"format: {random_injection}")

    lines.append("ENDCONTEXT")

    if summaries:
        lines.append(summaries[-1])

    for msg in history:
        role = msg.get("role", "user")
        if role == "assistant":
            role = assistant_name
        content = _strip_junk(msg.get("content", ""))
        lines.append(f"{role}: {content}")

    lines.append("ENDINPUT")

    lines.extend([
        "",
        "BEGININSTRUCTION",
        instruction.strip(),
        "ENDINSTRUCTION",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format a chat session in Airoboros style")
    parser.add_argument("json_file", help="Path to JSON file containing the session data")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = format_airoboros(
        data.get("global_prompt", ""),
        data.get("random_injection", ""),
        data.get("summaries"),
        data.get("history", []),
        data.get("instruction", ""),
        data.get("assistant_name", "assistant"),
    )

    print(result)


if __name__ == "__main__":
    main()
