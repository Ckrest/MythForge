from __future__ import annotations

import json

import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .utils import (
    ROOT_DIR,
    CHATS_DIR,
    myth_log,
    load_json,
    save_json,
    chat_file,
    ensure_chat_dir,
    goals_path,
    load_global_prompts,
    list_prompt_names,
    get_global_prompt_content,
    save_global_prompt,
    delete_global_prompt,
    _prompt_path,
)
from . import model, memory
from .call_core import handle_chat, build_call

app = FastAPI(title="Myth Forge Server")


@app.on_event("startup")
def _startup() -> None:
    """Load the model in the background."""

    model.warm_up(n_gpu_layers=model.DEFAULT_N_GPU_LAYERS)
    memory.initialize()


@app.on_event("shutdown")
def _shutdown() -> None:
    """Stop any background model process."""

    model._stop_warm()


# --- Configuration ---------------------------------------------------------


class ChatRequest(BaseModel):
    """Request model for chat-related endpoints."""

    chat_id: str
    message: str


# --- Load/Save operations --------------------------------------------------


def load_item(kind: str, name: str | None = None):
    """Load a JSON item based on ``kind`` and ``name``."""
    if kind == "chat_history" and name:
        return load_json(chat_file(name, "full.json"))
    if kind == "chats":
        os.makedirs(CHATS_DIR, exist_ok=True)
        return [
            d
            for d in os.listdir(CHATS_DIR)
            if os.path.isdir(os.path.join(CHATS_DIR, d))
        ]
    if kind == "prompts":
        if name:
            content = get_global_prompt_content(name)
            if content is None:
                raise HTTPException(status_code=404, detail="Prompt not found")
            return {"name": name, "content": content}
        return load_global_prompts()
    if kind == "prompt_names":
        return list_prompt_names()
    if kind == "settings":
        return model.MODEL_SETTINGS
    raise HTTPException(status_code=400, detail="Invalid load request")


def save_item(
    kind: str,
    name: str | None = None,
    *,
    data: object | None = None,
    delete: bool = False,
    new_name: str | None = None,
):
    """Handle saving, deleting, or renaming items."""
    if kind == "chat_history" and name:
        if delete:
            chat_dir = os.path.join(CHATS_DIR, name)
            if not os.path.isdir(chat_dir):
                raise HTTPException(status_code=404, detail="Chat not found")
            for fname in os.listdir(chat_dir):
                os.remove(os.path.join(chat_dir, fname))
            os.rmdir(chat_dir)
            return
        if new_name:
            old_dir = os.path.join(CHATS_DIR, name)
            new_dir = os.path.join(CHATS_DIR, new_name)
            if not os.path.isdir(old_dir):
                raise HTTPException(status_code=404, detail="Chat not found")
            if os.path.exists(new_dir):
                raise HTTPException(status_code=400, detail="Chat name already exists")
            os.rename(old_dir, new_dir)
            return
        ensure_chat_dir(name)
        save_json(chat_file(name, "full.json"), data or [])
        return

    if kind == "prompts" and name:
        if delete:
            delete_global_prompt(name)
        elif new_name:
            old_path = _prompt_path(name)
            new_path = _prompt_path(new_name)
            if not os.path.exists(old_path):
                raise HTTPException(status_code=404, detail="Prompt not found")
            if os.path.exists(new_path):
                raise HTTPException(
                    status_code=400, detail="Prompt name already exists"
                )
            os.rename(old_path, new_path)
        else:
            if data is None:
                raise HTTPException(status_code=400, detail="No prompt data provided")
            save_global_prompt({"name": name, "content": str(data)})
        prompts = load_global_prompts()
        memory.set_global_prompt(prompts[0]["content"] if prompts else "")
        return

    if kind == "settings" and isinstance(data, dict):
        model.MODEL_SETTINGS.update(data)
        save_json(model.MODEL_SETTINGS_PATH, model.MODEL_SETTINGS)
        for key in (
            "temp",
            "top_k",
            "top_p",
            "min_p",
            "repeat_penalty",
        ):
            if key in model.MODEL_SETTINGS:
                model.GENERATION_CONFIG[key] = model.MODEL_SETTINGS[key]
        model.DEFAULT_MAX_TOKENS = model.MODEL_SETTINGS.get(
            "max_tokens", model.DEFAULT_MAX_TOKENS
        )
        memory.update_model_settings(model.MODEL_SETTINGS)
        return

    raise HTTPException(status_code=400, detail="Invalid save request")


# --- Prompt Endpoints -----------------------------------------------------


@app.get("/prompts")
def list_prompts(names_only: int = 0):
    if names_only:
        return {"prompts": load_item("prompt_names")}
    return {"prompts": load_item("prompts")}


@app.get("/prompts/{name}")
def get_prompt(name: str):
    return load_item("prompts", name)


@app.post("/prompts")
def create_prompt(item: Dict[str, str]):
    save_item("prompts", item["name"], data=item.get("content", ""))
    return {"detail": "Created"}


@app.put("/prompts/{name}")
def update_prompt(name: str, item: Dict[str, str]):
    if item.get("name") != name:
        raise HTTPException(status_code=400, detail="Name mismatch")
    save_item("prompts", name, data=item.get("content", ""))
    return {"detail": "Updated"}


@app.put("/prompts/{name}/rename")
def rename_prompt(name: str, data: Dict[str, str]):
    new_name = data.get("new_name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name required")

    if new_name == name:
        return {"detail": f"Renamed prompt '{name}'"}

    save_item("prompts", name, new_name=new_name)
    return {"detail": f"Renamed prompt '{name}'"}


@app.delete("/prompts/{name}")
def remove_prompt(name: str):
    save_item("prompts", name, delete=True)
    return {"detail": f"Deleted prompt '{name}'"}


# --- Settings Endpoints ---------------------------------------------------


@app.get("/settings")
def get_settings():
    return load_item("settings")


@app.put("/settings")
def update_settings(data: Dict[str, object]):
    save_item("settings", data=data)
    return {"detail": "Updated", "settings": load_item("settings")}


# --- Response Prompt Status -----------------------------------------------


@app.get("/response_prompt_status")
def response_prompt_status():
    return {"pending": 0}


# --- Standard Chat Endpoints ---------------------------------------------


@app.get("/chats")
def list_chats():
    return {"chats": load_item("chats")}


@app.post("/chats/{chat_id}")
def create_chat(chat_id: str):
    """Create an empty chat directory for ``chat_id``."""
    chat_dir = os.path.join(CHATS_DIR, chat_id)
    if os.path.exists(chat_dir):
        raise HTTPException(status_code=400, detail="Chat already exists")
    save_item("chat_history", chat_id, data=[])
    memory.set_goals_enabled(False)
    return {"detail": f"Created chat '{chat_id}'", "chat_id": chat_id}


@app.get("/history/{chat_id}")
def get_history(chat_id: str):
    try:
        history = load_item("chat_history", chat_id)
    except HTTPException as exc:
        raise exc
    memory.load_goals(chat_id)
    return {"chat_id": chat_id, "history": history}


@app.put("/history/{chat_id}/{index}")
def edit_message(chat_id: str, index: int, data: Dict[str, str]):
    """Update the content of a message at ``index`` in ``chat_id``."""
    try:
        full = load_item("chat_history", chat_id)
    except HTTPException as exc:
        raise exc
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full[index]["content"] = data.get("content", "")
    save_item("chat_history", chat_id, data=full)
    return {"detail": "Updated"}


@app.delete("/history/{chat_id}/{index}")
def delete_message(chat_id: str, index: int):
    """Remove the message at ``index`` from ``chat_id``."""
    try:
        full = load_item("chat_history", chat_id)
    except HTTPException as exc:
        raise exc
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full.pop(index)
    save_item("chat_history", chat_id, data=full)
    return {"detail": "Deleted"}


@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str):
    save_item("chat_history", chat_id, delete=True)
    return {"detail": f"Deleted chat '{chat_id}'"}


@app.put("/chat/{chat_id}")
def rename_chat(chat_id: str, data: Dict[str, str]):
    """Rename an existing chat ``chat_id`` to ``data['new_id']``."""

    new_id = data.get("new_id", "").strip()
    if not new_id:
        raise HTTPException(status_code=400, detail="New chat id required")

    if new_id == chat_id:
        return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}

    save_item("chat_history", chat_id, new_name=new_id)
    return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}


@app.get("/chat/{chat_id}/goals")
def get_goals(chat_id: str):
    """Return goals data for ``chat_id`` including progress lists."""
    memory.load_goals(chat_id)
    goals = memory.MEMORY.goals_data
    return {
        "exists": goals.enabled,
        "character": goals.character,
        "setting": goals.setting,
        "in_progress": goals.active_goals,
        "completed": goals.deactive_goals,
    }


@app.put("/chat/{chat_id}/goals")
def save_goals(chat_id: str, data: Dict[str, object]):
    """Save goals for ``chat_id`` including progress tracking."""

    ensure_chat_dir(chat_id)
    obj = {
        "character": data.get("character", ""),
        "setting": data.get("setting", ""),
        "in_progress": data.get("in_progress", []),
        "completed": data.get("completed", []),
    }
    with open(goals_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    memory.set_goals_enabled(True)
    memory.update_goals(
        {
            "character": obj["character"],
            "setting": obj["setting"],
            "active_goals": obj["in_progress"],
            "deactive_goals": obj["completed"],
        }
    )
    return {"detail": "Saved"}


@app.post("/chat/{chat_id}/goals/disable")
def disable_goals(chat_id: str):
    """Disable goals for ``chat_id`` by renaming the JSON file."""

    path = goals_path(chat_id)
    disabled = chat_file(chat_id, "goals_disabled.json")
    if os.path.exists(path):
        os.rename(path, disabled)
    memory.load_goals(chat_id)
    return {"detail": "Disabled"}


@app.post("/chat/{chat_id}/goals/enable")
def enable_goals(chat_id: str):
    """Re-enable goals for ``chat_id`` if a disabled JSON file exists."""

    path = goals_path(chat_id)
    disabled = chat_file(chat_id, "goals_disabled.json")
    if os.path.exists(disabled):
        os.rename(disabled, path)
    memory.load_goals(chat_id)
    return {"detail": "Enabled"}


@app.get("/chat/{chat_id}/goals_enabled")
def goals_enabled_endpoint(chat_id: str):
    """Return whether goals exist for ``chat_id``."""

    memory.load_goals(chat_id)
    return {"enabled": memory.MEMORY.goals_data.enabled}


@app.get("/chat/{chat_id}/context")
def get_context_file(chat_id: str):
    """Return the character/setting context for ``chat_id``."""
    memory.load_goals(chat_id)
    goals = memory.MEMORY.goals_data
    return {"character": goals.character, "setting": goals.setting}


@app.put("/chat/{chat_id}/context")
def save_context_file(chat_id: str, data: Dict[str, str]):
    """Save character/setting context for ``chat_id`` preserving progress."""

    ensure_chat_dir(chat_id)
    obj = {
        "character": data.get("character", ""),
        "setting": data.get("setting", ""),
    }
    # Preserve goal progress if file already exists
    path = goals_path(chat_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                obj["in_progress"] = existing.get("in_progress", [])
                obj["completed"] = existing.get("completed", [])
        except Exception as e:
            print(f"Failed to load existing goals for '{chat_id}': {e}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    memory.set_goals_enabled(True)
    memory.update_goals(
        {
            "character": obj.get("character", ""),
            "setting": obj.get("setting", ""),
            "active_goals": obj.get("in_progress", []),
            "deactive_goals": obj.get("completed", []),
        }
    )
    return {"detail": "Saved"}


@app.post("/message")
def save_message(req: ChatRequest):
    """Store ``req.message`` in ``req.chat_id`` without generating a reply."""

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})
    save_item("chat_history", req.chat_id, data=history)
    return {"detail": "Message stored"}


@app.post("/chat/received")
def chat_received(req: ChatRequest):
    """Stream a model-generated reply for ``req``."""

    call = build_call(req)
    return handle_chat(call, stream=True)


@app.post("/chat")
def chat(req: ChatRequest):
    """Return a standard model-generated reply."""

    call = build_call(req)
    return handle_chat(call)


# --- Static UI Mount ------------------------------------------------------

app.mount("/", StaticFiles(directory="ui", html=True), name="static")
