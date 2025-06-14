from __future__ import annotations

import json

import os
from typing import Dict, List, Iterator

from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from . import model
from .memory import (
    MemoryManager,
    MEMORY_MANAGER,
    initialize as init_memory,
)
from .call_core import ChatRunner, build_call
from .call_templates import standard_chat
from .call_templates.standard_chat import prep_standard_chat

DEBUG_MODE = os.environ.get("DEBUG", "0") not in {"0", "false", "False"}

app = FastAPI(title="MythForgeUI", debug=DEBUG_MODE)

memory_manager: MemoryManager = MEMORY_MANAGER
chat_runner = ChatRunner(memory_manager)

init_memory(memory_manager)


@app.on_event("startup")
async def startup_event() -> None:
    """Launch the chat subprocess when the API starts."""

    prep_standard_chat()


chat_router = APIRouter()
prompt_router = APIRouter()
settings_router = APIRouter()


# --- Helpers ---------------------------------------------------------------


def get_memory_manager() -> MemoryManager:
    return memory_manager


# --- Configuration ---------------------------------------------------------


class ChatRequest(BaseModel):
    """Request model for chat-related endpoints."""

    chat_id: str
    message: str


# --- Load/Save operations --------------------------------------------------


def load_item(kind: str, name: str | None = None):
    """Load a JSON item based on ``kind`` and ``name``."""
    if kind == "chat_history" and name:
        return memory_manager.load_history(name)
    if kind == "chats":
        return memory_manager.list_chats()
    if kind == "prompts":
        if name:
            content = memory_manager.get_global_prompt(name)
            if not content:
                raise HTTPException(status_code=404, detail="Prompt not found")
            return {"name": name, "content": content}
        return memory_manager.load_global_prompts()
    if kind == "prompt_names":
        return memory_manager.list_prompt_names()
    if kind == "settings":
        return memory_manager.load_settings()
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
            if name not in memory_manager.list_chats():
                raise HTTPException(status_code=404, detail="Chat not found")
            memory_manager.delete_chat(name)
            return
        if new_name:
            if name not in memory_manager.list_chats():
                raise HTTPException(status_code=404, detail="Chat not found")
            if new_name in memory_manager.list_chats():
                raise HTTPException(
                    status_code=400, detail="Chat name already exists"
                )
            memory_manager.rename_chat(name, new_name)
            return
        memory_manager.save_history(name, data or [])
        return

    if kind == "prompts" and name:
        if delete:
            memory_manager.delete_global_prompt(name)
        elif new_name:
            if not memory_manager.get_global_prompt(name):
                raise HTTPException(status_code=404, detail="Prompt not found")
            if memory_manager.get_global_prompt(new_name):
                raise HTTPException(
                    status_code=400, detail="Prompt name already exists"
                )
            memory_manager.rename_global_prompt(name, new_name)
        else:
            if data is None:
                raise HTTPException(
                    status_code=400, detail="No prompt data provided"
                )
            memory_manager.set_global_prompt(name, str(data))
        prompts = memory_manager.load_global_prompts()
        memory_manager.update_paths(
            prompt_name=prompts[0]["name"] if prompts else ""
        )
        return

    if kind == "settings" and isinstance(data, dict):
        model.MODEL_SETTINGS.update(data)
        memory_manager.save_settings(model.MODEL_SETTINGS)
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
        memory_manager.model_settings.update(model.MODEL_SETTINGS)
        return

    raise HTTPException(status_code=400, detail="Invalid save request")


# --- Prompt Endpoints -----------------------------------------------------


@prompt_router.get("/")
def list_prompts(names_only: int = 0):
    if names_only:
        return {"prompts": load_item("prompt_names")}
    return {"prompts": load_item("prompts")}


@prompt_router.get("/{name}")
def get_prompt(name: str):
    return load_item("prompts", name)


@prompt_router.post("/")
def create_prompt(item: Dict[str, str]):
    save_item("prompts", item["name"], data=item.get("content", ""))
    return {"detail": "Created"}


@prompt_router.put("/{name}")
def update_prompt(name: str, item: Dict[str, str]):
    if item.get("name") != name:
        raise HTTPException(status_code=400, detail="Name mismatch")
    save_item("prompts", name, data=item.get("content", ""))
    return {"detail": "Updated"}


@prompt_router.put("/{name}/rename")
def rename_prompt(name: str, data: Dict[str, str]):
    new_name = data.get("new_name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name required")

    if new_name == name:
        return {"detail": f"Renamed prompt '{name}'"}

    save_item("prompts", name, new_name=new_name)
    memory_manager.update_paths(prompt_name=new_name)
    return {"detail": f"Renamed prompt '{name}'"}


@prompt_router.delete("/{name}")
def remove_prompt(name: str):
    save_item("prompts", name, delete=True)
    memory_manager.update_paths(prompt_name="")
    return {"detail": f"Deleted prompt '{name}'"}


@prompt_router.post("/select")
def select_prompt(data: Dict[str, str]):
    """Set the active global prompt to ``data['name']``."""

    name = data.get("name", "").strip()
    if not name:
        memory_manager.update_paths(prompt_name="")
        return {"detail": "Cleared"}

    content = get_global_prompt_content(name)
    if content is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    memory_manager.update_paths(prompt_name=name)
    return {"detail": f"Selected prompt '{name}'"}


# --- Settings Endpoints ---------------------------------------------------


@settings_router.get("/")
def get_settings():
    return load_item("settings")


@settings_router.put("/")
def update_settings(data: Dict[str, object]):
    save_item("settings", data=data)
    return {"detail": "Updated", "settings": load_item("settings")}


# --- Response Prompt Status -----------------------------------------------


@app.get("/response_prompt_status")
def response_prompt_status():
    return {"pending": 0}


# --- Standard Chat Endpoints ---------------------------------------------


@chat_router.get("/")
def list_chats():
    return {"chats": memory_manager.list_chats()}


@chat_router.post("/{chat_id}")
def create_chat(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Create an empty chat directory for ``chat_id``."""
    if chat_id in memory_manager.list_chats():
        raise HTTPException(status_code=400, detail="Chat already exists")
    memory_manager.save_history(chat_id, [])
    memory.toggle_goals(False)
    memory.update_paths(chat_name=chat_id)
    return {"detail": f"Created chat '{chat_id}'", "chat_id": chat_id}


@chat_router.get("/{chat_id}/history")
def get_history(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    history_data = memory.load_history(chat_id)
    memory.load_goals(chat_id)
    return {"chat_id": chat_id, "history": history_data}


@chat_router.put("/{chat_id}/history/{index}")
def edit_message(
    chat_id: str,
    index: int,
    data: Dict[str, str],
):
    """Update the content of a message at ``index`` in ``chat_id``."""
    full = memory_manager.load_history(chat_id)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full[index]["content"] = data.get("content", "")
    memory_manager.save_history(chat_id, full)
    return {"detail": "Updated"}


@chat_router.delete("/{chat_id}/history/{index}")
def delete_message(
    chat_id: str,
    index: int,
):
    """Remove the message at ``index`` from ``chat_id``."""
    full = memory_manager.load_history(chat_id)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full.pop(index)
    memory_manager.save_history(chat_id, full)
    return {"detail": "Deleted"}


@chat_router.delete("/{chat_id}")
def delete_chat(
    chat_id: str,
):
    memory_manager.delete_chat(chat_id)
    memory_manager.update_paths(chat_name="")
    return {"detail": f"Deleted chat '{chat_id}'"}


@chat_router.put("/{chat_id}")
def rename_chat(
    chat_id: str,
    data: Dict[str, str],
):
    """Rename an existing chat ``chat_id`` to ``data['new_id']``."""

    new_id = data.get("new_id", "").strip()
    if not new_id:
        raise HTTPException(status_code=400, detail="New chat id required")

    if new_id == chat_id:
        return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}

    if chat_id not in memory_manager.list_chats():
        raise HTTPException(status_code=404, detail="Chat not found")
    if new_id in memory_manager.list_chats():
        raise HTTPException(status_code=400, detail="Chat name already exists")
    memory_manager.rename_chat(chat_id, new_id)
    memory_manager.update_paths(chat_name=new_id)
    return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}


@chat_router.get("/{chat_id}/goals")
def get_goals(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Return goals data for ``chat_id`` including progress lists."""
    goals = memory.load_goals(chat_id)
    return {
        "exists": memory.goals_active,
        "character": goals.character,
        "setting": goals.setting,
        "in_progress": goals.active_goals,
        "completed": goals.deactive_goals,
    }


@chat_router.put("/{chat_id}/goals")
def save_goals(
    chat_id: str,
    data: Dict[str, object],
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Save goals for ``chat_id`` including progress tracking."""

    memory.save_goals(chat_id, data)
    return {"detail": "Saved"}


@chat_router.post("/{chat_id}/goals/disable")
def disable_goals(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Disable goals for ``chat_id`` by renaming the JSON file."""

    memory.disable_goals(chat_id)
    return {"detail": "Disabled"}


@chat_router.post("/{chat_id}/goals/enable")
def enable_goals(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Re-enable goals for ``chat_id`` if a disabled JSON file exists."""

    memory.enable_goals(chat_id)
    return {"detail": "Enabled"}


@chat_router.get("/{chat_id}/goals_enabled")
def goals_enabled_endpoint(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Return whether goals exist for ``chat_id``."""

    memory.load_goals(chat_id)
    return {"enabled": memory.goals_active}


@chat_router.get("/{chat_id}/context")
def get_context_file(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Return the character/setting context for ``chat_id``."""
    goals = memory.load_goals(chat_id)
    return {"character": goals.character, "setting": goals.setting}


@chat_router.put("/{chat_id}/context")
def save_context_file(
    chat_id: str,
    data: Dict[str, str],
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Save character/setting context for ``chat_id`` preserving progress."""

    obj = {
        "character": data.get("character", ""),
        "setting": data.get("setting", ""),
    }
    existing = memory.load_goals(chat_id)
    if existing.active_goals or existing.deactive_goals:
        obj["in_progress"] = existing.active_goals
        obj["completed"] = existing.deactive_goals
    memory.save_goals(chat_id, obj)
    return {"detail": "Saved"}


@chat_router.post("/message")
def save_message(
    req: ChatRequest,
):
    """Store ``req.message`` in ``req.chat_id`` without generating a reply."""

    memory_manager.append_message(req.chat_id, "user", req.message)
    return {"detail": "Message stored"}


@chat_router.post("/{chat_id}/message")
def send_chat_message(chat_id: str, req: ChatRequest):
    """Stream a reply for ``req.message`` using the standard chat model."""

    memory_manager.append_message(chat_id, "user", req.message)
    stream = standard_chat.send_prompt("", req.message, stream=True)

    def _generate() -> Iterator[str]:
        parts: list[str] = []
        for chunk in stream:
            text = chunk.get("text", "")
            yield text + "\n"
            parts.append(text)
        memory_manager.append_message(
            chat_id, "<|im_start|>assistant<|im_end|>", "".join(parts)
        )

    return StreamingResponse(_generate(), media_type="text/plain")


@chat_router.post("/{chat_id}/cli")
def run_cli_command(chat_id: str, req: ChatRequest):
    """Stream ``req.message`` to the running CLI process."""

    from .call_templates import standard_chat

    memory_manager.append_message(chat_id, "user", req.message)
    stream = standard_chat.send_cli_command(req.message, stream=True)

    def _generate() -> Iterator[str]:
        parts: list[str] = []
        for chunk in stream:
            text = chunk.get("text", "")
            parts.append(text)
            yield text
        memory_manager.append_message(chat_id, "assistant", "".join(parts))

    return StreamingResponse(_generate(), media_type="text/plain")


@chat_router.post("/{chat_id}/assistant")
def append_assistant_message(
    chat_id: str,
    data: Dict[str, str],
):
    """Append an assistant message to ``chat_id``."""

    memory_manager.append_message(
        chat_id, "assistant", data.get("message", "")
    )
    return {"detail": "Message stored"}


@chat_router.post("/received")
def chat_received(
    req: ChatRequest,
    runner: ChatRunner = Depends(lambda: chat_runner),
):
    """Stream a model-generated reply for ``req``."""

    memory_manager.append_message(req.chat_id, "user", req.message)
    call = build_call(req)
    return runner.process_user_message(req.chat_id, req.message, stream=True)


@chat_router.post("/")
def chat(
    req: ChatRequest,
    runner: ChatRunner = Depends(lambda: chat_runner),
):
    """Return a standard model-generated reply."""

    memory_manager.append_message(req.chat_id, "user", req.message)
    call = build_call(req)
    return runner.process_user_message(req.chat_id, req.message)


# --- Static UI Mount ------------------------------------------------------

app.include_router(chat_router, prefix="/chats")
app.include_router(prompt_router, prefix="/prompts")
app.include_router(settings_router, prefix="/settings")

app.mount("/", StaticFiles(directory="ui", html=True), name="static")
