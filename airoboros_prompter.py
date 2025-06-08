#!/usr/bin/env python3
"""Prompt formatter for LLaMA3 style header-token conversations.

The previous version of this module generated prompts using the
Airoboros ``BEGININPUT``/``BEGININSTRUCTION`` style.  It has been
refactored to output the header-token format required by models such as
LLaMA3.  Each message is wrapped with ``<|start_header_id|>`` and
``<|end_header_id|>`` tokens followed by a blank line and the message
content.  Messages are separated with ``<|eot_id|>`` tokens and the final
prompt ends with an ``assistant`` header so the model knows to respond.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional


def _strip_junk(text: str) -> str:
    """Remove LM Studio tokens and metadata lines."""
    cleaned = re.sub(
        r"<\|start_header_id\>|<\|end_header_id\>|<\|eot_id\>|<\|begin_of_text\|>",
        "",
        text,
    )
    cleaned = re.sub(r"^Cutting Knowledge Date.*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^Today Date.*\n?", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def format_llama3(
    global_prompt: str,
    summaries: Optional[List[str]],
    history: List[Dict[str, str]],
    instruction: str,
    assistant_name: str = "assistant",
) -> str:
    """Return the conversation formatted for the LLaMA3 header-token style."""

    messages: List[Dict[str, str]] = []

    # Remove any stray header tokens from user provided text to avoid
    # duplicating them in the final prompt.
    global_prompt = _strip_junk(global_prompt)
    instruction = _strip_junk(instruction)

    system_parts: List[str] = []
    if global_prompt:
        system_parts.append(global_prompt.strip())
    if summaries:
        system_parts.append(summaries[-1])
    if system_parts:
        messages.append({"role": "system", "content": "\n".join(system_parts)})

    for msg in history:
        role = msg.get("role", "user")
        if role == "assistant":
            role = assistant_name
        content = _strip_junk(msg.get("content", ""))
        if role == assistant_name:
            role = "assistant"
        messages.append({"role": role, "content": content})

    if instruction.strip():
        messages.append({"role": "user", "content": instruction.strip()})

    lines: List[str] = []
    for m in messages:
        lines.append(f"<|start_header_id|>{m['role']}<|end_header_id|>")
        lines.append("")
        lines.append(m["content"])
        lines.append("")
        lines.append("<|eot_id|>")

    lines.append("<|start_header_id|>assistant<|end_header_id|>")
    lines.append("")

    prompt = "\n".join(lines)
    # Strip any stray ``<|begin_of_text|>`` tokens at the start
    prompt = re.sub(r"^(<\|begin_of_text\|>)+", "", prompt)
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format a chat session in LLaMA3 header-token style"
    )
    parser.add_argument("json_file", help="Path to JSON file containing the session data")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = format_llama3(
        data.get("global_prompt", ""),
        data.get("summaries"),
        data.get("history", []),
        data.get("instruction", ""),
        data.get("assistant_name", "assistant"),
    )

    print(result)


if __name__ == "__main__":
    main()
