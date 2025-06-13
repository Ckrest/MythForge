"""Utilities for loading the language model and configuration."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Dict, Iterator

from .utils import ROOT_DIR, myth_log


MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_SETTINGS_PATH = os.path.join(ROOT_DIR, "model_settings.json")


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
}


def llm_args(
    *, stream: bool = False, background: bool = False
) -> dict[str, object]:
    """Return argument mapping for :func:`call_llm`."""

    args = GENERATION_CONFIG.copy()
    args["stream"] = stream
    args["n_gpu_layers"] = 0 if background else DEFAULT_N_GPU_LAYERS
    return args


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


def _select_model_path(background: bool = False) -> str:
    """Return configured model path or fallback to discovery."""

    key = "background_model" if background else "primary_model"
    name = MODEL_SETTINGS.get(key, "")
    if name:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return path
    return discover_model_path()


def _default_cli() -> str:
    name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
    return os.path.join(ROOT_DIR, "dependencies", name)


LLAMA_CLI = os.path.abspath(MODEL_SETTINGS.get("llama_cli", _default_cli()))


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


def call_llm(system_prompt: str, user_prompt: str, **kwargs):
    """Return output from :data:`LLAMA_CLI` for the given prompts."""

    cmd = [
        LLAMA_CLI,
        "--chat-template",
        "",
        "--prompt",
        user_prompt,
    ]
    cmd.append("--no-warmup")
    cmd.append("--no-conversation")
    cmd.extend(_cli_args(**kwargs))
    if "--single-turn" not in cmd:
        cmd.insert(1, "--single-turn")
    try:
        if "model" not in kwargs:
            background = kwargs.get("n_gpu_layers", DEFAULT_N_GPU_LAYERS) == 0
            model_path = _select_model_path(background)
            cmd.extend(["--model", model_path])
    except Exception as exc:
        myth_log("call_llm_error", error=str(exc))
        raise

    myth_log("call_llm_start", cmd=" ".join(cmd))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
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

# ---------------------------------------------------------------------------
# Model warm-up utilities
# ---------------------------------------------------------------------------

_warm_process: subprocess.Popen | None = None


def _stop_warm() -> None:
    """Terminate the warm-up process if running."""

    global _warm_process
    if _warm_process and _warm_process.poll() is None:
        _warm_process.terminate()
        try:
            _warm_process.wait(timeout=5)
        except Exception:
            _warm_process.kill()
    _warm_process = None


__all__ = [
    "GENERATION_CONFIG",
    "DEFAULT_N_GPU_LAYERS",
    "call_llm",
    "_stop_warm",
]
