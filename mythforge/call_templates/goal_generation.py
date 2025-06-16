from __future__ import annotations

from typing import Any, Dict, Iterator

from ..response_parser import ResponseParser
from ..prompt_preparer import PromptPreparer
from ..invoker import LLMInvoker

# -----------------------------------
# Model launch parameters / arguments ORERRIDE
# -----------------------------------

MODEL_LAUNCH_OVERRIDE: Dict[str, Any] = {
    "n_gpu_layers": 0,
    "stream": False,
    "background": True,
}


def generate_goals(global_prompt: str, message: str, options: Dict[str, Any]):
    """Run the goal-generation template and return parsed output."""

    prepared = PromptPreparer().prepare(global_prompt, message)
    raw = LLMInvoker().invoke(prepared, options)
    parsed = ResponseParser().load(raw).parse()
    if isinstance(parsed, Iterator):
        try:
            first = next(parsed)
        except StopIteration as exc:  # pragma: no cover - best effort
            return str(exc.value)

        def _chain() -> Iterator[str]:
            yield first
            yield from parsed

        return _chain()
    return parsed
