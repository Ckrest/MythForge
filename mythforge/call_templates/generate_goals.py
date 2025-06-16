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
    "verbose": True,
}


def generate_goals(
    global_prompt: str, message: str, options: Dict[str, Any]
) -> Iterator[str]:
    """Run the goal-generation template and return parsed output."""

    prepared = PromptPreparer().prepare(global_prompt, message)
    opts = {**MODEL_LAUNCH_OVERRIDE, **options}
    raw = LLMInvoker().invoke(prepared, opts)
    return ResponseParser().load(raw).parse()
