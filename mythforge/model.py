"""Utilities for loading the language model and configuration."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, Iterator

from .utils import myth_log


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
    """Return path to the first ``.gguf`` model under ``MODELS_DIR``."""

    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory '{MODELS_DIR}' not found")

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            if fname.lower().endswith(".gguf"):
                return os.path.join(root, fname)

    raise FileNotFoundError(f"No .gguf model files found under '{MODELS_DIR}'")


LLAMA_CLI = os.path.join("dependencies", "llama-cli.exe")


def _cli_args(**kwargs) -> list[str]:
    args = []
    for key, value in kwargs.items():
        if key == "stream":
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(option)
        else:
            args.extend([option, str(value)])
    return args


def call_llm(prompt: str, **kwargs):
    """Return output from :data:`LLAMA_CLI` for ``prompt`` with logging."""

    cmd = [LLAMA_CLI, "--prompt", prompt]
    cmd.extend(_cli_args(**kwargs))
    myth_log("call_llm_start", cmd=" ".join(cmd))

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except Exception as exc:  # pragma: no cover - best effort
        myth_log("call_llm_error", error=str(exc))
        raise RuntimeError(f"Failed to start process: {exc}") from exc

    stream = kwargs.get("stream", False)

    if stream:

        def _stream() -> Iterator[dict[str, str]]:
            assert process.stdout is not None
            for line in process.stdout:
                yield {"text": line.rstrip()}
            myth_log("call_llm_exit", code=process.wait())

        return _stream()

    output, _ = process.communicate()
    myth_log("call_llm_exit", code=process.returncode)
    if process.returncode != 0:
        raise RuntimeError(f"Subprocess exited with code {process.returncode}")
    return {"text": output.rstrip()}


call_llm._patched = True
