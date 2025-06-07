#!/usr/bin/env python3
"""Minimal FastAPI server using the LM Studio prompt wrapper."""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lmstudio_prompter import format_prompt, GENERATION_CONFIG

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    custom_tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: bool = False


app = FastAPI()
app.state.model = None


@app.on_event("startup")
def load_model() -> None:
    """Load the Llama model from ``models`` or ``MODEL_PATH`` env variable."""
    path = os.environ.get("MODEL_PATH")

    # If MODEL_PATH is not set, search for the first ``*.gguf`` file in
    # the ``models`` directory.
    if path is None:
        models_dir = "models"
        if os.path.isdir(models_dir):
            for file_name in os.listdir(models_dir):
                if file_name.lower().endswith(".gguf"):
                    path = os.path.join(models_dir, file_name)
                    break
        if path is None:
            path = os.path.join(models_dir, "model.gguf")

    if Llama is None:
        raise RuntimeError("llama_cpp is required to load a GGUF model")

    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")

    app.state.model = Llama(path)


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    input_obj = req.dict()
    prompt = format_prompt(input_obj)
    try:
        result = app.state.model(prompt, **GENERATION_CONFIG)
        text = result["choices"][0]["text"]
    except Exception as exc:  # pragma: no cover - depends on llama_cpp
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"prompt": prompt, "response": text}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
