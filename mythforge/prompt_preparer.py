from __future__ import annotations

"""Prompt preparation utilities."""

from typing import Any

from .logger import LOGGER


class PromptPreparer:
    """Combine system and user text into a single prompt string."""

    def __init__(self) -> None:
        """No template state is stored by default."""

        self.template: str = ""

    def prepare(
        self,
        system_text: str | None = None,
        user_text: str = "",
        *,
        name: str | None = None,
    ) -> list[dict[str, str]]:
        """Return a full prompt for ``system_text`` and ``user_text``.

        If ``name`` is provided, ``system_text`` will be loaded from the
        :class:`MemoryManager` using that prompt name.
        """

        if name is not None:
            from .memory import MEMORY_MANAGER

            system_text = MEMORY_MANAGER.get_global_prompt(name)
        system_text = system_text or ""

        LOGGER.log(
            "chat_flow",
            {
                "function": "PromptPreparer.prepare",
                "system_text": system_text,
                "user_text": user_text,
            },
        )

        system_text_clean = (
            system_text.replace("\n", " ").replace('"', '\\"').strip()
        )

        user_text_clean = (
            user_text.replace("\n", " ").replace('"', '\\"').strip()
        )

        prompt_full: list[dict[str, str]] = []

        if system_text_clean:
            prompt_full.append({"role": "system", "content": system_text_clean})

        prompt_full.append({"role": "user", "content": user_text_clean})

        prompt_full.append({"role": "assistant", "content": ""})

        return prompt_full

    def format_for_logging(self, system_text: str, user_text: str) -> str:
        """Return ``system_text`` and ``user_text`` as a single log string."""

        parts = [
            "[system text]",
            "",
            system_text or "",
            "",
            "[user text]",
            "",
            user_text or "",
            "",
            "[assistant]",
        ]

        return "\n".join(parts)
