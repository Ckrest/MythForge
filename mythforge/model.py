"""Utilities for loading the language model and configuration."""

from __future__ import annotations

import json
import os
import subprocess
import threading
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
DEFAULT_N_GPU_LAYERS = MODEL_SETTINGS.get("n_gpu_layers", 35)
DEFAULT_MAX_TOKENS = MODEL_SETTINGS.get("max_tokens", 250)
SUMMARIZE_THRESHOLD = MODEL_SETTINGS.get("summarize_threshold", 20)
SUMMARIZE_BATCH = MODEL_SETTINGS.get("summarize_batch", 12)

# Generation settings
GENERATION_CONFIG = {
    "temp": MODEL_SETTINGS.get("temp", 0.8),
    "top_k": MODEL_SETTINGS.get("top_k", 40),
    "top_p": MODEL_SETTINGS.get("top_p", 0.95),
    "min_p": MODEL_SETTINGS.get("min_p", 0.05),
    "repeat_penalty": MODEL_SETTINGS.get("repeat_penalty", 1.1),
    "no_interactive": True,
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

# Currently running subprocess if any
CURRENT_PROCESS: subprocess.Popen | None = None
_PROC_LOCK = threading.Lock()


def abort_current_generation() -> None:
    """Terminate any active LLM process."""

    global CURRENT_PROCESS
    with _PROC_LOCK:
        proc = CURRENT_PROCESS
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        CURRENT_PROCESS = None


def _cli_args(**kwargs) -> list[str]:
    """Return CLI arguments mapping Python names to binary flags."""

    option_map = {}
    args = []
    for key, value in kwargs.items():
        if key == "stream":
            continue
        opt_key = option_map.get(key, key)
        option = f"--{opt_key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                args.extend([option, str(item)])
        else:
            args.extend([option, str(value)])
    return args


def call_llm(prompt: str, **kwargs):
    """Return output from :data:`LLAMA_CLI` for ``prompt`` with logging."""

    cmd = [LLAMA_CLI, "--prompt", prompt]
    cmd.extend(_cli_args(**kwargs))
    if "--single-turn" not in cmd:
        cmd.insert(1, "--single-turn")
    try:
        if "model" not in kwargs:
            model_path = discover_model_path()
            cmd.extend(["--model", model_path])
    except Exception as exc:
        myth_log("call_llm_error", error=str(exc))
        raise

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
        with _PROC_LOCK:
            abort_current_generation()
            globals()["CURRENT_PROCESS"] = process

        def _stream() -> Iterator[dict[str, str]]:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    yield {"text": line.rstrip()}
            finally:
                myth_log("call_llm_exit", code=process.wait())
                abort_current_generation()

        return _stream()

    output, _ = process.communicate()
    myth_log("call_llm_exit", code=process.returncode)
    if process.returncode != 0:
        raise RuntimeError(f"Subprocess exited with code {process.returncode}")
    return {"text": output.rstrip()}


call_llm._patched = True
