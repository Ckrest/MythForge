#!/usr/bin/env python3
"""LMStudio prompt builder.

This script mimics LM Studio's prompt wrapping logic using the
jinja-like template provided in the user description. It accepts an
input object containing chat ``messages`` and optional ``tools`` and
produces the formatted prompt string along with the generation
configuration used by LM Studio.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# Generation settings used by LM Studio
GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "n_batch": 512,
    "stop": ["<|start_header_id|>", "<|eot_id|>"],
}


def format_prompt(
    input_obj: Dict[str, Any],
    bos_token: str = "<s>",
) -> str:
    """Format a prompt using the LM Studio wrapping rules.

    Parameters
    ----------
    input_obj : dict
        Object containing ``messages`` (list of dicts) and optional keys
        ``tools``, ``custom_tools``, ``strftime_now`` and
        ``add_generation_prompt``.
    bos_token : str, optional
        Token inserted at the start of the prompt.

    Returns
    -------
    str
        The fully formatted prompt string.
    """

    messages: List[Dict[str, Any]] = input_obj.get("messages", [])
    tools = input_obj.get("tools")
    custom_tools = input_obj.get("custom_tools")
    strftime_now = input_obj.get("strftime_now")
    add_generation_prompt = bool(input_obj.get("add_generation_prompt", False))
    tools_in_user_message = input_obj.get("tools_in_user_message", True)

    if custom_tools is not None:
        tools = custom_tools

    # determine date string
    if input_obj.get("date_string"):
        date_string = input_obj["date_string"]
    elif strftime_now:
        date_string = strftime_now("%d %b %Y")
    else:
        date_string = datetime.now().strftime("%d %b %Y")

    prompt_parts: List[str] = [bos_token]

    # Extract system message if present
    system_message = ""
    if messages and messages[0].get("role") == "system":
        system_message = messages[0].get("content", "").strip()
        messages = messages[1:]

    # System header
    prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
    if tools is not None:
        prompt_parts.append("Environment: ipython\n")
    prompt_parts.append("Cutting Knowledge Date: December 2023\n")
    prompt_parts.append(f"Today Date: {date_string}\n\n")

    if tools is not None and not tools_in_user_message:
        prompt_parts.append(
            "You have access to the following functions. To call a function, "
            "please respond with JSON for a function call.Respond in the "
            "format {\"name\":function name,\"parameters\":{…}}. Do not use variables.\n\n"
        )
        for t in tools:
            prompt_parts.append(json.dumps(t, indent=4))
            prompt_parts.append("\n\n")

    if system_message:
        prompt_parts.append(system_message)
    prompt_parts.append("<|eot_id|>")

    # Tool-calling user header
    if tools_in_user_message and tools is not None:
        if not messages:
            raise ValueError("No user message for tools!")
        first_msg = messages[0]
        if isinstance(first_msg.get("content"), str):
            first_user_message = first_msg.get("content", "").strip()
        else:
            first_user_message = first_msg.get("content", [{"text": ""}])[0].get("text", "").strip()
        messages = messages[1:]
        prompt_parts.append("<|start_header_id|>user<|end_header_id|>\n\n")
        prompt_parts.append(first_user_message)
        prompt_parts.append("<|eot_id|>")

    # Remaining messages
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            prompt_parts.append(message.get("content", "").strip())
            prompt_parts.append("<|eot_id|>")
        elif role == "user":
            prompt_parts.append("<|start_header_id|>user<|end_header_id|>\n\n")
            prompt_parts.append(message.get("content", "").strip())
            prompt_parts.append("<|eot_id|>")
        elif role == "system":
            prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
            prompt_parts.append(message.get("content", "").strip())
            prompt_parts.append("<|eot_id|>")
        elif message.get("tool_calls"):
            call = message["tool_calls"][0]["function"]
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            prompt_parts.append(
                json.dumps({"name": call["name"], "parameters": call["arguments"]})
            )
            prompt_parts.append("<|eot_id|>")
        elif role == "ipython" or "tool_results" in role:
            prompt_parts.append("<|start_header_id|>ipython<|end_header_id|>\n\n")
            content = message.get("content")
            if isinstance(content, (list, dict)):
                prompt_parts.append(json.dumps(content))
            else:
                prompt_parts.append(str(content))
            prompt_parts.append("<|eot_id|>")

    if add_generation_prompt:
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(prompt_parts)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: lmstudio_prompter.py <input_json_file>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt = format_prompt(data)
    print(prompt)
    print()
    print(json.dumps(GENERATION_CONFIG, indent=2))

