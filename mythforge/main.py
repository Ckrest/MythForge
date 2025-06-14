from __future__ import annotations

import json

import os
import asyncio
from typing import Dict, List

from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
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
from . import model
from .memory import (
    ChatHistoryService,
    MemoryManager,
    MEMORY_MANAGER,
    initialize as init_memory,
)
from .call_core import ChatRunner, build_call, handle_chat
from .call_templates import standard_chat

app = FastAPI(title="Myth Forge Server")

history_service = ChatHistoryService()
memory_manager: MemoryManager = MEMORY_MANAGER
chat_runner = ChatRunner(history_service, memory_manager)

init_memory(memory_manager)

chat_router = APIRouter()
prompt_router = APIRouter()
settings_router = APIRouter()

# --- Server-Sent Events ----------------------------------------------------

clients: list[asyncio.Queue[str]] = []


async def _event_stream(queue: asyncio.Queue[str]):
    """Yield messages from ``queue`` to SSE clients."""

    try:
        while True:
            data = await queue.get()
            yield data
    finally:
        clients.remove(queue)


@app.get("/events")
async def sse_events():
    """Endpoint for subscribing to server-sent events."""

    queue: asyncio.Queue[str] = asyncio.Queue()
    clients.append(queue)
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(_event_stream(queue), headers=headers)


async def broadcast_reload() -> None:
    """Send a reload event to all connected clients."""

    for queue in list(clients):
        await queue.put("event: reload\ndata:\n\n")


@app.on_event("startup")
async def on_startup() -> None:
    await broadcast_reload()


def get_history_service() -> ChatHistoryService:
    return history_service


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
                raise HTTPException(status_code=400, detail="No prompt data provided")
            save_global_prompt({"name": name, "content": str(data)})
        prompts = load_global_prompts()
        memory_manager.global_prompt = prompts[0]["content"] if prompts else ""
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
    return {"detail": f"Renamed prompt '{name}'"}


@prompt_router.delete("/{name}")
def remove_prompt(name: str):
    save_item("prompts", name, delete=True)
    return {"detail": f"Deleted prompt '{name}'"}


@prompt_router.post("/select")
def select_prompt(data: Dict[str, str]):
    """Set the active global prompt to ``data['name']``."""

    name = data.get("name", "").strip()
    if not name:
        memory_manager.global_prompt = ""
        return {"detail": "Cleared"}

    content = get_global_prompt_content(name)
    if content is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    memory_manager.global_prompt = content
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
def list_chats(
    history: ChatHistoryService = Depends(get_history_service),
):
    return {"chats": history.list_chats()}


@chat_router.post("/{chat_id}")
def create_chat(
    chat_id: str,
    history: ChatHistoryService = Depends(get_history_service),
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Create an empty chat directory for ``chat_id``."""
    chat_dir = os.path.join(CHATS_DIR, chat_id)
    if os.path.exists(chat_dir):
        raise HTTPException(status_code=400, detail="Chat already exists")
    history.save_history(chat_id, [])
    memory.toggle_goals(False)
    return {"detail": f"Created chat '{chat_id}'", "chat_id": chat_id}


@chat_router.get("/{chat_id}/history")
def get_history(
    chat_id: str,
    history: ChatHistoryService = Depends(get_history_service),
    memory: MemoryManager = Depends(get_memory_manager),
):
    history_data = history.load_history(chat_id)
    memory.load_goals(chat_id)
    return {"chat_id": chat_id, "history": history_data}


@chat_router.put("/{chat_id}/history/{index}")
def edit_message(
    chat_id: str,
    index: int,
    data: Dict[str, str],
    history: ChatHistoryService = Depends(get_history_service),
):
    """Update the content of a message at ``index`` in ``chat_id``."""
    full = history.load_history(chat_id)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full[index]["content"] = data.get("content", "")
    history.save_history(chat_id, full)
    return {"detail": "Updated"}


@chat_router.delete("/{chat_id}/history/{index}")
def delete_message(
    chat_id: str,
    index: int,
    history: ChatHistoryService = Depends(get_history_service),
):
    """Remove the message at ``index`` from ``chat_id``."""
    full = history.load_history(chat_id)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full.pop(index)
    history.save_history(chat_id, full)
    return {"detail": "Deleted"}


@chat_router.delete("/{chat_id}")
def delete_chat(
    chat_id: str,
    history: ChatHistoryService = Depends(get_history_service),
):
    chat_dir = os.path.join(CHATS_DIR, chat_id)
    if os.path.isdir(chat_dir):
        for fname in os.listdir(chat_dir):
            os.remove(os.path.join(chat_dir, fname))
        os.rmdir(chat_dir)
    return {"detail": f"Deleted chat '{chat_id}'"}


@chat_router.put("/{chat_id}")
def rename_chat(
    chat_id: str,
    data: Dict[str, str],
    history: ChatHistoryService = Depends(get_history_service),
):
    """Rename an existing chat ``chat_id`` to ``data['new_id']``."""

    new_id = data.get("new_id", "").strip()
    if not new_id:
        raise HTTPException(status_code=400, detail="New chat id required")

    if new_id == chat_id:
        return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}

    old_dir = os.path.join(CHATS_DIR, chat_id)
    new_dir = os.path.join(CHATS_DIR, new_id)
    if not os.path.isdir(old_dir):
        raise HTTPException(status_code=404, detail="Chat not found")
    if os.path.exists(new_dir):
        raise HTTPException(status_code=400, detail="Chat name already exists")
    os.rename(old_dir, new_dir)
    return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}


@chat_router.get("/{chat_id}/goals")
def get_goals(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Return goals data for ``chat_id`` including progress lists."""
    memory.load_goals(chat_id)
    goals = memory.goals_data
    return {
        "exists": goals.enabled,
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

    path = goals_path(chat_id)
    disabled = chat_file(chat_id, "goals_disabled.json")
    if os.path.exists(path):
        os.rename(path, disabled)
    memory.load_goals(chat_id)
    return {"detail": "Disabled"}


@chat_router.post("/{chat_id}/goals/enable")
def enable_goals(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Re-enable goals for ``chat_id`` if a disabled JSON file exists."""

    path = goals_path(chat_id)
    disabled = chat_file(chat_id, "goals_disabled.json")
    if os.path.exists(disabled):
        os.rename(disabled, path)
    memory.load_goals(chat_id)
    return {"detail": "Enabled"}


@chat_router.get("/{chat_id}/goals_enabled")
def goals_enabled_endpoint(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Return whether goals exist for ``chat_id``."""

    memory.load_goals(chat_id)
    return {"enabled": memory.goals_data.enabled}


@chat_router.get("/{chat_id}/context")
def get_context_file(
    chat_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
):
    """Return the character/setting context for ``chat_id``."""
    memory.load_goals(chat_id)
    goals = memory.goals_data
    return {"character": goals.character, "setting": goals.setting}


@chat_router.put("/{chat_id}/context")
def save_context_file(
    chat_id: str,
    data: Dict[str, str],
    memory: MemoryManager = Depends(get_memory_manager),
):
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
    memory.toggle_goals(True)
    memory.update_goals(
        {
            "character": obj.get("character", ""),
            "setting": obj.get("setting", ""),
            "active_goals": obj.get("in_progress", []),
            "deactive_goals": obj.get("completed", []),
        }
    )
    return {"detail": "Saved"}


@chat_router.post("/message")
def save_message(
    req: ChatRequest,
    history: ChatHistoryService = Depends(get_history_service),
):
    """Store ``req.message`` in ``req.chat_id`` without generating a reply."""

    history.append_message(req.chat_id, "user", req.message)
    return {"detail": "Message stored"}


@chat_router.post("/{chat_id}/message")
def send_chat_message(chat_id: str, req: ChatRequest) -> Dict[str, str]:
    """Generate a reply for ``req.message`` in ``chat_id``."""

    if not standard_chat.chat_running():
        raise HTTPException(status_code=503, detail="Model not running")

    history_service.append_message(chat_id, "user", req.message)
    call = build_call(req)
    return handle_chat(call, history_service, memory_manager, stream=True)


@chat_router.post("/{chat_id}/cli")
def run_cli_command(chat_id: str, req: ChatRequest) -> Dict[str, str]:
    """Send ``req.message`` directly to the running CLI process."""

    from .call_templates import standard_chat

    history_service.append_message(chat_id, "user", req.message)
    result = standard_chat.send_cli_command(req.message)
    text = result.get("text", "")
    history_service.append_message(chat_id, "assistant", text)
    return {"detail": text}


@chat_router.post("/{chat_id}/assistant")
def append_assistant_message(
    chat_id: str,
    data: Dict[str, str],
    history: ChatHistoryService = Depends(get_history_service),
):
    """Append an assistant message to ``chat_id``."""

    history.append_message(chat_id, "assistant", data.get("message", ""))
    return {"detail": "Message stored"}


@chat_router.post("/received")
def chat_received(
    req: ChatRequest,
    runner: ChatRunner = Depends(lambda: chat_runner),
    history: ChatHistoryService = Depends(get_history_service),
):
    """Stream a model-generated reply for ``req``."""

    history.append_message(req.chat_id, "user", req.message)
    call = build_call(req)
    return runner.process_user_message(req.chat_id, req.message, stream=True)


@chat_router.post("/")
def chat(
    req: ChatRequest,
    runner: ChatRunner = Depends(lambda: chat_runner),
    history: ChatHistoryService = Depends(get_history_service),
):
    """Return a standard model-generated reply."""

    history.append_message(req.chat_id, "user", req.message)
    call = build_call(req)
    return runner.process_user_message(req.chat_id, req.message)


# --- Static UI Mount ------------------------------------------------------

app.include_router(chat_router, prefix="/chats")
app.include_router(prompt_router, prefix="/prompts")
app.include_router(settings_router, prefix="/settings")

app.mount("/", StaticFiles(directory="ui", html=True), name="static")
