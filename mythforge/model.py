"""Utilities for loading the language model and configuration."""

from __future__ import annotations

import os
from typing import Dict, Iterator, List

from llama_cpp import Llama

from .memory import MEMORY_MANAGER

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


MODELS_DIR = os.path.join(ROOT_DIR, "models")


load_model_settings = MEMORY_MANAGER.load_settings
save_model_settings = MEMORY_MANAGER.save_settings


MODEL_SETTINGS = load_model_settings()

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
DEFAULT_CTX_SIZE = MODEL_SETTINGS.get("n_ctx", 4096)
DEFAULT_N_BATCH = MODEL_SETTINGS.get("n_batch", 512)
DEFAULT_N_THREADS = MODEL_SETTINGS.get("n_threads", os.cpu_count() or 1)
DEFAULT_N_GPU_LAYERS = MODEL_SETTINGS.get("n_gpu_layers", 35)
DEFAULT_MAX_TOKENS = MODEL_SETTINGS.get("max_tokens", 250)
SUMMARIZE_THRESHOLD = MODEL_SETTINGS.get("summarize_threshold", 20)
SUMMARIZE_BATCH = MODEL_SETTINGS.get("summarize_batch", 12)

# Generation settings
GENERATION_CONFIG = {
    "temperature": MODEL_SETTINGS.get("temperature", 0.8),
    "top_k": MODEL_SETTINGS.get("top_k", 40),
    "top_p": MODEL_SETTINGS.get("top_p", 0.95),
    "min_p": MODEL_SETTINGS.get("min_p", 0.05),
    "repeat_penalty": MODEL_SETTINGS.get("repeat_penalty", 1.1),
}


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def discover_model_path() -> str:
    """Search ``MODELS_DIR`` for a supported model file."""

    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory '{MODELS_DIR}' not found")

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            if fname.lower().endswith(".gguf"):
                return os.path.join(root, fname)

    raise FileNotFoundError(f"No .gguf model files found under '{MODELS_DIR}'")


def _select_model_path(background: bool = False) -> str:
    """Choose an explicit or discovered model path."""

    key = "background_model" if background else "primary_model"
    name = MODEL_SETTINGS.get(key, "")
    if name:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return path
    return discover_model_path()


_LLAMA: Llama | None = None


def _get_llama(background: bool = False) -> Llama:
    """Instantiate and cache the Llama backend."""

    global _LLAMA
    if _LLAMA is None:
        _LLAMA = Llama(
            model_path=_select_model_path(background),
            n_ctx=DEFAULT_CTX_SIZE,
            n_gpu_layers=DEFAULT_N_GPU_LAYERS,
            n_batch=DEFAULT_N_BATCH,
            n_threads=DEFAULT_N_THREADS,
        )
    return _LLAMA


MODEL_LAUNCH_ARGS: Dict[str, object] = {
    "background": False,
    "stream": True,
    "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
    **GENERATION_CONFIG,
}


def call_llm(prompt: str | list[dict[str, str]], **overrides):
    """Route ``prompt`` through ``llama_cpp`` and return the response."""

    params = MODEL_LAUNCH_ARGS.copy()
    params.update(overrides)

    background = params.pop("background", False)
    stream = params.pop("stream", True)
    params.pop("n_gpu_layers", None)

    llm = _get_llama(background)

    if stream:

        def _stream() -> Iterator[dict[str, str]]:
            if isinstance(prompt, list):
                for chunk in llm.create_chat_completion(messages=prompt, stream=True, **params):
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    text = delta.get("content", "")
                    yield {"text": text}
            else:
                for chunk in llm.create_completion(prompt, stream=True, **params):
                    text = chunk["choices"][0]["text"]
                    yield {"text": text}

        return _stream()

    if isinstance(prompt, list):
        result = llm.create_chat_completion(messages=prompt, stream=False, **params)
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        return {"text": text}

    result = llm.create_completion(prompt, stream=False, **params)
    return {"text": result["choices"][0]["text"]}


call_llm._patched = True


__all__ = [
    "GENERATION_CONFIG",
    "DEFAULT_N_GPU_LAYERS",
    "call_llm",
]
