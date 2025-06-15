from __future__ import annotations

from typing import Any, Dict

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
    """Return parsed goal generation result."""

    system, user = PromptPreparer().prepare(global_prompt, message)
    raw = LLMInvoker().invoke(system, options)
    result = ResponseParser().load(raw).parse()
    return result
