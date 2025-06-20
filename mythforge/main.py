from __future__ import annotations

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
from .call_core import handle_chat
from .prompt_preparer import PromptPreparer
from .invoker import LLMInvoker
from .logger import LOGGER

app = FastAPI(title="MythForgeUI", debug=False)

memory_manager: MemoryManager = MEMORY_MANAGER

init_memory(memory_manager)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize directories and default prompts on startup."""
    try:
        model._get_llama(verbose=False)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.log_error(exc)


chat_router = APIRouter()
global_prompt_router = APIRouter()
settings_router = APIRouter()


# --- Helpers ---------------------------------------------------------------


def get_memory_manager() -> MemoryManager:
    """Provide the shared :class:`MemoryManager` instance."""

    return memory_manager


# --- Configuration ---------------------------------------------------------


class ChatRequest(BaseModel):
    """Request model for chat-related endpoints."""

    message: str
    chat_name: str | None = None
    global_prompt_name: str | None = None


class SendChatRequest(ChatRequest):
    """Extended request model for ``/chat/send``."""

    global_prompt: str | None = None


# --- Load/Save operations --------------------------------------------------


# --- Prompt Endpoints -----------------------------------------------------


@global_prompt_router.get("/")
def list_prompts(
    names_only: int = 0, chat_name: str = "", global_prompt_name: str = ""
):
    """Return all stored prompt entries or just their names."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    if names_only:
        return {"prompts": memory_manager.list_global_prompt_names()}
    return {"prompts": memory_manager.load_global_prompts()}


@global_prompt_router.get("/{name}")
def get_prompt(name: str, chat_name: str = "", global_prompt_name: str = ""):
    """Fetch the full content for ``name``."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    content = memory_manager.get_global_prompt(name)
    if not content:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"name": name, "content": content}


@global_prompt_router.post("/")
def create_prompt(
    item: Dict[str, str], chat_name: str = "", global_prompt_name: str = ""
):
    """Create a new prompt file from ``item``."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    memory_manager.set_global_prompt(item["name"], item.get("content", ""))
    return {"detail": "Created"}


@global_prompt_router.put("/{name}")
def update_prompt(
    name: str, item: Dict[str, str], chat_name: str = "", global_prompt_name: str = ""
):
    """Overwrite an existing prompt with new content."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    if item.get("name") != name:
        raise HTTPException(status_code=400, detail="Name mismatch")
    memory_manager.set_global_prompt(name, item.get("content", ""))
    return {"detail": "Updated"}


@global_prompt_router.put("/{name}/rename")
def rename_prompt(
    name: str, data: Dict[str, str], chat_name: str = "", global_prompt_name: str = ""
):
    """Rename a stored prompt without changing its content."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    new_name = data.get("new_name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name required")

    if new_name == name:
        return {"detail": f"Renamed prompt '{name}'"}

    if not memory_manager.get_global_prompt(name):
        raise HTTPException(status_code=404, detail="Prompt not found")
    if memory_manager.get_global_prompt(new_name):
        raise HTTPException(status_code=400, detail="Prompt name already exists")
    memory_manager.rename_global_prompt(name, new_name)
    memory_manager.update_paths(global_prompt_name=new_name)
    return {"detail": f"Renamed prompt '{name}'"}


@global_prompt_router.delete("/{name}")
def remove_prompt(name: str, chat_name: str = "", global_prompt_name: str = ""):
    """Delete the prompt identified by ``name``."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    memory_manager.delete_global_prompt(name)
    memory_manager.update_paths(global_prompt_name="")
    return {"detail": f"Deleted prompt '{name}'"}


@global_prompt_router.post("/select")
def select_prompt(
    data: Dict[str, str], chat_name: str = "", global_prompt_name: str = ""
):
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    """Set ``data['name']`` as the active system prompt."""

    name = data.get("name", "").strip()
    if not name:
        memory_manager.update_paths(global_prompt_name="")
        return {"detail": "Cleared"}

    content = memory_manager.get_global_prompt(name)
    if not content:
        raise HTTPException(status_code=404, detail="Prompt not found")
    memory_manager.update_paths(global_prompt_name=name)
    return {"detail": f"Selected prompt '{name}'"}


# --- Settings Endpoints ---------------------------------------------------


@settings_router.get("/")
def load_settings_endpoint(chat_name: str = "", global_prompt_name: str = ""):
    """Return the current server settings."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    return memory_manager.load_settings()


@settings_router.put("/")
def update_settings_endpoint(
    data: Dict[str, object], chat_name: str = "", global_prompt_name: str = ""
):
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    updated = memory_manager.update_settings(data)
    return {"detail": "Updated", "settings": updated}


# --- Response Prompt Status -----------------------------------------------


@app.get("/response_prompt_status")
def response_prompt_status(chat_name: str = "", global_prompt_name: str = ""):
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    from .background import has_pending_tasks

    debug = memory_manager.pop_debug_messages()
    return {"pending": int(has_pending_tasks()), "debug": debug}


# --- Standard Chat Endpoints ---------------------------------------------


@chat_router.get("/")
def list_chats(chat_name: str = "", global_prompt_name: str = ""):
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    return {"chats": memory_manager.list_chats()}


@chat_router.post("/{chat_name}")
def create_chat(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Create an empty chat directory for ``chat_name``."""
    if chat_name in memory_manager.list_chats():
        raise HTTPException(status_code=400, detail="Chat already exists")
    memory_manager.save_chat_history(chat_name, [])
    memory.toggle_goals(False)
    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    return {"detail": f"Created chat '{chat_name}'", "chat_name": chat_name}


@chat_router.get("/{chat_name}/chat_history")
def load_chat_history_endpoint(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    chat_history_data = memory.load_chat_history(chat_name)
    memory.load_goals(chat_name)
    return {"chat_name": chat_name, "chat_history": chat_history_data}


@chat_router.put("/{chat_name}/chat_history/{index}")
def save_message_endpoint(
    chat_name: str,
    index: int,
    data: Dict[str, str],
    global_prompt_name: str = "",
):
    """Persist edits to a specific message in chat_history."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    full = memory_manager.load_chat_history(chat_name)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full[index]["content"] = data.get("content", "")
    memory_manager.save_chat_history(chat_name, full)
    return {"detail": "Updated"}


@chat_router.delete("/{chat_name}/chat_history/{index}")
def delete_message_endpoint(
    chat_name: str,
    index: int,
    global_prompt_name: str = "",
):
    """Remove the message at ``index`` from stored chat_history."""
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=global_prompt_name
    )
    full = memory_manager.load_chat_history(chat_name)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full.pop(index)
    memory_manager.save_chat_history(chat_name, full)
    return {"detail": "Deleted"}


@chat_router.delete("/{chat_name}")
def delete_chat(
    chat_name: str,
    global_prompt_name: str = "",
):
    """Erase all chat_history for ``chat_name``."""
    memory_manager.delete_chat(chat_name)
    memory_manager.update_paths(chat_name="", global_prompt_name=global_prompt_name)
    return {"detail": f"Deleted chat '{chat_name}'"}


@chat_router.put("/{chat_name}")
def rename_chat(
    chat_name: str,
    data: Dict[str, str],
    global_prompt_name: str = "",
):
    """Update the folder name for the chat session."""

    new_id = data.get("new_id", "").strip()
    if not new_id:
        raise HTTPException(status_code=400, detail="New chat id required")

    if new_id == chat_name:
        return {"detail": f"Renamed chat '{chat_name}' to '{new_id}'"}

    if chat_name not in memory_manager.list_chats():
        raise HTTPException(status_code=404, detail="Chat not found")
    if new_id in memory_manager.list_chats():
        raise HTTPException(status_code=400, detail="Chat name already exists")
    memory_manager.rename_chat(chat_name, new_id)
    memory_manager.update_paths(chat_name=new_id, global_prompt_name=global_prompt_name)
    return {"detail": f"Renamed chat '{chat_name}' to '{new_id}'"}


@chat_router.get("/{chat_name}/goals")
def get_goals(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Load current goals associated with ``chat_name``."""
    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    goals = memory.load_goals(chat_name)
    state = memory.load_goal_state(chat_name)
    return {
        "exists": memory.goals_active,
        "character": goals.character,
        "setting": goals.setting,
        "in_progress": state.get("goals", []),
        "completed": state.get("completed_goals", []),
    }


@chat_router.put("/{chat_name}/goals")
def save_goals(
    chat_name: str,
    data: Dict[str, object],
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Persist modified goals to disk."""

    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    memory.save_goals(chat_name, data)
    state = memory.load_goal_state(chat_name)
    if "in_progress" in data:
        state["goals"] = data.get("in_progress", [])
    if "completed" in data:
        state["completed_goals"] = data.get("completed", [])
    memory.save_goal_state(chat_name, state)
    return {"detail": "Saved"}


@chat_router.post("/{chat_name}/goals/disable")
def disable_goals(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Stop automatic goal insertion for this chat."""

    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    memory.disable_goals(chat_name)
    return {"detail": "Disabled"}


@chat_router.post("/{chat_name}/goals/enable")
def enable_goals(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Resume auto-appending goals for this chat."""

    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    memory.enable_goals(chat_name)
    return {"detail": "Enabled"}


@chat_router.get("/{chat_name}/goals_enabled")
def goals_enabled_endpoint(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Return ``True`` if goal integration is active."""

    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    memory.load_goals(chat_name)
    return {"enabled": memory.goals_active}


@chat_router.get("/{chat_name}/context")
def get_context_file(
    chat_name: str,
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Return the stored context file contents."""
    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    goals = memory.load_goals(chat_name)
    return {"character": goals.character, "setting": goals.setting}


@chat_router.put("/{chat_name}/context")
def save_context_file(
    chat_name: str,
    data: Dict[str, str],
    memory: MemoryManager = Depends(get_memory_manager),
    global_prompt_name: str = "",
):
    """Store an uploaded context file for this chat."""

    obj = {
        "character": data.get("character", ""),
        "setting": data.get("setting", ""),
    }
    memory.update_paths(chat_name=chat_name, global_prompt_name=global_prompt_name)
    memory.save_goals(chat_name, obj)
    return {"detail": "Saved"}


@chat_router.post("/{chat_name}/cli")
def run_cli_command(chat_name: str, req: ChatRequest):
    """Execute a shell command and stream its output."""

    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=req.global_prompt_name
    )
    chat_history = memory_manager.load_chat_history(chat_name)
    chat_history.append({"role": "user", "content": req.message})
    memory_manager.save_chat_history(chat_name, chat_history)
    prepared = PromptPreparer().prepare("", req.message)
    stream = LLMInvoker().invoke(prepared, {"stream": True})

    def _generate() -> Iterator[str]:
        parts: list[str] = []
        for chunk in stream:
            text = chunk.get("text", "")
            parts.append(text)
            yield text
        chat_history = memory_manager.load_chat_history(chat_name)
        chat_history.append({"role": "assistant", "content": "".join(parts)})
        memory_manager.save_chat_history(chat_name, chat_history)

    return StreamingResponse(_generate(), media_type="text/plain")


@chat_router.post("/{chat_name}/message")
def chat_message(chat_name: str, req: ChatRequest):
    """Process a user message and stream the assistant reply."""

    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=req.global_prompt_name or ""
    )
    chat_history = memory_manager.load_chat_history(chat_name)
    chat_history.append({"role": "user", "content": req.message})
    memory_manager.save_chat_history(chat_name, chat_history)
    global_prompt = memory_manager.get_global_prompt(req.global_prompt_name or "")
    return handle_chat(
        chat_name,
        req.message,
        global_prompt,
        memory=memory_manager,
        stream=True,
        current_chat_name=chat_name,
        current_global_prompt=req.global_prompt_name,
    )


@app.post("/chat/send")
def send_chat(req: SendChatRequest):
    """Handle a chat message using JSON body for identifiers."""

    memory_manager.update_paths(
        chat_name=req.chat_name,
        global_prompt_name=req.global_prompt_name,
    )

    chat_name = memory_manager.chat_name
    global_prompt_name = memory_manager.global_prompt_name

    chat_history = memory_manager.load_chat_history(chat_name)
    chat_history.append({"role": "user", "content": req.message})
    memory_manager.save_chat_history(chat_name, chat_history)

    memory_manager.load_goals(chat_name)

    global_prompt = memory_manager.get_global_prompt(global_prompt_name)

    return handle_chat(
        chat_name,
        req.message,
        global_prompt,
        memory=memory_manager,
        stream=True,
        current_chat_name=chat_name,
        current_global_prompt=global_prompt_name,
    )


@chat_router.post("/{chat_name}/assistant")
def append_assistant_message(
    chat_name: str,
    req: ChatRequest,
):
    """Add an assistant response without calling the model."""

    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=req.global_prompt_name or ""
    )
    chat_history = memory_manager.load_chat_history(chat_name)
    chat_history.append({"role": "assistant", "content": req.message})
    memory_manager.save_chat_history(chat_name, chat_history)
    return {"detail": "Appended"}


@chat_router.post("/message")
def append_user_message(req: ChatRequest):
    """Store a user message without generating a reply."""

    chat_name = req.chat_name or memory_manager.chat_name
    if not chat_name:
        raise HTTPException(status_code=400, detail="No active chat")
    memory_manager.chat_name = chat_name
    memory_manager.global_prompt_name = (
        req.global_prompt_name or memory_manager.global_prompt_name
    )
    memory_manager.update_paths(
        chat_name=chat_name, global_prompt_name=memory_manager.global_prompt_name
    )
    chat_history = memory_manager.load_chat_history(chat_name)
    chat_history.append({"role": "user", "content": req.message})
    memory_manager.save_chat_history(chat_name, chat_history)
    return {"detail": "Message stored", "chat_name": chat_name}


# --- Static UI Mount ------------------------------------------------------

app.include_router(chat_router, prefix="/chats")
app.include_router(global_prompt_router, prefix="/prompts")
app.include_router(settings_router, prefix="/settings")

app.mount("/", StaticFiles(directory="ui", html=True), name="static")
