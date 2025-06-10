import json
import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import model_launch, model_call

app = FastAPI(title="Myth Forge Server")

# --- Configuration ---------------------------------------------------------

CHATS_DIR = "chats"
GLOBAL_PROMPTS_DIR = "global_prompts"


class ChatRequest(BaseModel):
    """Request model for chat-related endpoints."""

    chat_id: str
    message: str
    global_prompt: str | None = None


# --- Helper utilities ------------------------------------------------------


def load_json(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from '{path}': {e}")
        except Exception as e:
            print(f"Failed to load JSON from '{path}': {e}")
    return []


def save_json(path: str, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def chat_file(chat_id: str, filename: str) -> str:
    """Return the path for ``filename`` within ``chat_id``'s directory."""
    return os.path.join(CHATS_DIR, chat_id, filename)


def ensure_chat_dir(chat_id: str) -> str:
    """Create and return the directory path for ``chat_id``."""
    path = os.path.join(CHATS_DIR, chat_id)
    os.makedirs(path, exist_ok=True)
    return path


def goals_path(chat_id: str) -> str:
    """Return the path to ``chat_id``'s goals JSON file."""

    return chat_file(chat_id, "goals.json")


def goals_exists(chat_id: str) -> bool:
    """Return ``True`` if goals are enabled for ``chat_id``."""

    return os.path.exists(goals_path(chat_id))


def read_text_file(path: str) -> str:
    """Return the contents of ``path`` as a string."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read text from '{path}': {e}")
    return ""


def write_text_file(path: str, text: str) -> None:
    """Write ``text`` to ``path`` creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_model_call(system_text: str, user_text: str, call_type: str):
    """Return a model call based on ``call_type``."""

    prompt = model_call.model_call(system_text, user_text, call_type)
    kwargs = model_launch.GENERATION_CONFIG.copy()
    kwargs["stream"] = call_type == "standard_chat"
    return model_launch.call_llm(prompt, **kwargs)


# --- Global prompt utilities ----------------------------------------------


def _prompt_path(name: str) -> str:
    """Return the filesystem path for a prompt ``name``."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return os.path.join(GLOBAL_PROMPTS_DIR, f"{safe}.json")


def load_global_prompts() -> List[Dict[str, str]]:
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    prompts: List[Dict[str, str]] = []
    for fname in sorted(os.listdir(GLOBAL_PROMPTS_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(GLOBAL_PROMPTS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load prompt '{fname}': {e}")
            continue
        if isinstance(data, dict) and "name" in data and "content" in data:
            prompts.append({"name": data["name"], "content": data["content"]})
        else:
            print(f"Ignoring invalid global prompt file: {fname}")
    return prompts


def list_prompt_names() -> List[str]:
    """Return only the names of available global prompts."""
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    names: List[str] = []
    for fname in sorted(os.listdir(GLOBAL_PROMPTS_DIR)):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(GLOBAL_PROMPTS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load prompt '{fname}': {e}")
            continue
        if isinstance(data, dict) and "name" in data:
            names.append(data["name"])
    return names


def get_global_prompt_content(name: str) -> str | None:
    """Return the content string for a prompt ``name`` if it exists."""
    path = _prompt_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("content") if isinstance(data, dict) else None
    except Exception as e:
        print(f"Failed to load prompt '{name}': {e}")
        return None


def save_global_prompt(prompt: Dict[str, str]) -> None:
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    path = _prompt_path(prompt["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"name": prompt["name"], "content": prompt["content"]},
            f,
            indent=2,
            ensure_ascii=False,
        )


def delete_global_prompt(name: str) -> None:
    path = _prompt_path(name)
    if os.path.exists(path):
        os.remove(path)


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
        return model_launch.MODEL_SETTINGS
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
                raise HTTPException(
                    status_code=400, detail="Chat name already exists"
                )
            os.rename(old_dir, new_dir)
            return
        ensure_chat_dir(name)
        save_json(chat_file(name, "full.json"), data or [])
        return

    if kind == "prompts" and name:
        if delete:
            delete_global_prompt(name)
            return
        if new_name:
            old_path = _prompt_path(name)
            new_path = _prompt_path(new_name)
            if not os.path.exists(old_path):
                raise HTTPException(status_code=404, detail="Prompt not found")
            if os.path.exists(new_path):
                raise HTTPException(
                    status_code=400, detail="Prompt name already exists"
                )
            os.rename(old_path, new_path)
            return
        if data is None:
            raise HTTPException(
                status_code=400, detail="No prompt data provided"
            )
        save_global_prompt({"name": name, "content": str(data)})
        return

    if kind == "settings" and isinstance(data, dict):
        model_launch.MODEL_SETTINGS.update(data)
        save_json(
            model_launch.MODEL_SETTINGS_PATH, model_launch.MODEL_SETTINGS
        )
        for key in (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repeat_penalty",
            "stop",
        ):
            if key in model_launch.MODEL_SETTINGS:
                model_launch.GENERATION_CONFIG[key] = (
                    model_launch.MODEL_SETTINGS[key]
                )
        model_launch.DEFAULT_MAX_TOKENS = model_launch.MODEL_SETTINGS.get(
            "max_tokens", model_launch.DEFAULT_MAX_TOKENS
        )
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
    return {"detail": f"Created chat '{chat_id}'"}


@app.get("/history/{chat_id}")
def get_history(chat_id: str):
    try:
        return load_item("chat_history", chat_id)
    except HTTPException as exc:
        raise exc


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
    """Return character and setting goals for ``chat_id``."""

    path = goals_path(chat_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {
                    "exists": True,
                    "character": data.get("character", ""),
                    "setting": data.get("setting", ""),
                }
        except Exception as e:
            print(f"Failed to load goals for '{chat_id}': {e}")
    return {"exists": False, "character": "", "setting": ""}


@app.put("/chat/{chat_id}/goals")
def save_goals(chat_id: str, data: Dict[str, str]):
    """Save character and setting goals for ``chat_id``."""

    ensure_chat_dir(chat_id)
    obj = {
        "character": data.get("character", ""),
        "setting": data.get("setting", ""),
    }
    with open(goals_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return {"detail": "Saved"}


@app.post("/chat/{chat_id}/goals/disable")
def disable_goals(chat_id: str):
    """Disable goals for ``chat_id`` by renaming the file."""

    path = goals_path(chat_id)
    disabled = chat_file(chat_id, "goals_disabled")
    if os.path.exists(path):
        os.rename(path, disabled)
    return {"detail": "Disabled"}


@app.post("/chat/{chat_id}/goals/enable")
def enable_goals(chat_id: str):
    """Re-enable goals for ``chat_id`` if a disabled file exists."""

    path = goals_path(chat_id)
    disabled = chat_file(chat_id, "goals_disabled")
    if os.path.exists(disabled):
        os.rename(disabled, path)
    return {"detail": "Enabled"}


@app.get("/chat/{chat_id}/goals_enabled")
def goals_enabled_endpoint(chat_id: str):
    """Return whether goals exist for ``chat_id``."""

    return {"enabled": goals_exists(chat_id)}


@app.get("/chat/{chat_id}/context")
def get_context_file(chat_id: str):
    """Return the character/setting context for ``chat_id``."""

    path = goals_path(chat_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {
                    "character": data.get("character", ""),
                    "setting": data.get("setting", ""),
                }
        except Exception as e:
            print(f"Failed to load context for '{chat_id}': {e}")
    return {"character": "", "setting": ""}


@app.put("/chat/{chat_id}/context")
def save_context_file(chat_id: str, data: Dict[str, str]):
    """Save character/setting context for ``chat_id``."""

    ensure_chat_dir(chat_id)
    obj = {
        "character": data.get("character", ""),
        "setting": data.get("setting", ""),
    }
    with open(goals_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return {"detail": "Saved"}


@app.post("/message")
def save_message(req: ChatRequest):
    """Store ``req.message`` in ``req.chat_id`` without generating a reply."""

    ensure_chat_dir(req.chat_id)
    history = load_item("chat_history", req.chat_id)
    history.append({"role": "user", "content": req.message})
    save_item("chat_history", req.chat_id, data=history)
    return {"detail": "Message stored"}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream a model-generated reply and store it in chat history."""

    return model_call.chat_stream(req)


@app.post("/chat")
def chat(req: ChatRequest):
    """Return a standard model-generated reply."""

    return model_call.chat(req)


# --- Static UI Mount ------------------------------------------------------

app.mount("/", StaticFiles(directory="ui", html=True), name="static")
