"""Utilities for loading the language model and configuration."""

from __future__ import annotations

import inspect
import json
import os
from typing import Dict

from llama_cpp import Llama, llama_print_system_info


MODELS_DIR = "models"
MODEL_SETTINGS_PATH = "model_settings.json"


def load_model_settings(path: str = MODEL_SETTINGS_PATH) -> Dict[str, object]:
    """Load the model settings from ``path`` if it exists."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                print(f"Failed to load model settings '{path}': {e}")
    return {}


MODEL_SETTINGS = load_model_settings()

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
DEFAULT_CTX_SIZE = MODEL_SETTINGS.get("n_ctx", 4096)
DEFAULT_N_BATCH = MODEL_SETTINGS.get("n_batch", 512)
DEFAULT_N_THREADS = MODEL_SETTINGS.get("n_threads", os.cpu_count() or 1)
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
    "stop": MODEL_SETTINGS.get("stop", ["<|eot_id|>"]),
}


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def discover_model_path() -> str:
    """Return the path to the first ``.gguf`` model file under ``MODELS_DIR``."""

    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory '{MODELS_DIR}' not found")

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            if fname.lower().endswith(".gguf"):
                return os.path.join(root, fname)

    raise FileNotFoundError(f"No .gguf model files found under '{MODELS_DIR}'")


_model_config = {
    "model_path": MODEL_SETTINGS.get("model_path") or discover_model_path(),
    "n_ctx": MODEL_SETTINGS.get("n_ctx", DEFAULT_CTX_SIZE),
    "n_batch": MODEL_SETTINGS.get("n_batch", DEFAULT_N_BATCH),
    "n_threads": MODEL_SETTINGS.get("n_threads") or DEFAULT_N_THREADS,
    "prompt_template": MODEL_SETTINGS.get("prompt_template", ""),
}

for key in (
    "f16_kv",
    "use_mmap",
    "use_mlock",
    "n_gpu_layers",
    "main_memory_kv",
):
    if key in MODEL_SETTINGS and MODEL_SETTINGS[key] is not None:
        _model_config[key] = MODEL_SETTINGS[key]

# Default to full GPU utilization when not specified
if "n_gpu_layers" not in _model_config:
    _model_config["n_gpu_layers"] = -1

try:
    sig = inspect.signature(Llama)
    if "add_bos" in sig.parameters:
        _model_config["add_bos"] = False
    elif "add_bos_token" in sig.parameters:
        _model_config["add_bos_token"] = False
except Exception:
    pass

llm = Llama(**_model_config)


def show_system_info() -> None:
    """Print llama.cpp system info for debugging GPU usage."""

    try:
        info = llama_print_system_info()
        print(info)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Failed to get system info: {exc}")


show_system_info()

try:
    _CALL_KWARGS = set(inspect.signature(llm.__call__).parameters)
except Exception:  # pragma: no cover
    _CALL_KWARGS = set()


def call_llm(prompt: str, **kwargs):
    """Call the ``llm`` object with ``prompt`` and return the result."""

    if _CALL_KWARGS:
        filtered = {k: v for k, v in kwargs.items() if k in _CALL_KWARGS}
    else:  # Fallback if inspection failed
        filtered = kwargs

    if filtered.get("stream"):

        def _stream():
            for chunk in llm(prompt, **filtered):
                yield chunk

        return _stream()

    return llm(prompt, **filtered)


call_llm._patched = True
