from __future__ import annotations

"""Prompt preparation utilities."""

from typing import Any

from .logger import LOGGER

from .logger import LOGGER


class PromptPreparer:
    """Combine system and user text into a single prompt string."""

    def __init__(self) -> None:
        """Load template data for prompt assembly."""

        self.template: str = ""

    def load_template(self, name: str) -> str:
        """Read the template named ``name`` from storage."""

        from .memory import MEMORY_MANAGER

        self.template = MEMORY_MANAGER.get_global_prompt(name)
        return self.template

    def prepare(self, system_text: str, user_text: str) -> str:
        """Inject texts into ``self.template`` and return the final prompt."""

        LOGGER.log(
            "chat_flow",
            {
                "function": "PromptPreparer.prepare",
                "system_text": system_text,
                "user_text": user_text,
            },
        )

        system_clean = (
            system_text.replace("\n", " ").replace('"', '\\"').strip()
        )
        user_clean = user_text.replace("\n", " ").replace('"', '\\"').strip()
        prompt = (
            f"<|im_start|>{system_clean}<|im_end|>"
            f"<|im_start|>user {user_clean}<|im_end|>"
            f"<|im_start|>assistant"
        )
        return f'--prompt "{prompt}"'
