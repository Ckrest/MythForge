import os
import json
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import model_launch
from pydantic import BaseModel

app = FastAPI(title="Myth Forge Server")

# ========== Configuration ==========
CHATS_DIR = "chats"
GLOBAL_PROMPTS_DIR = "global_prompts"


class ChatRequest(BaseModel):
    """Request model for chat-related endpoints."""

    chat_id: str
    message: str
    global_prompt: str | None = None


# ========== Helpers ==========
def load_json(path):
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


def chat_file(chat_id: str, filename: str) -> str:
    """Return the path for ``filename`` within ``chat_id``'s directory."""
    return os.path.join(CHATS_DIR, chat_id, filename)


def ensure_chat_dir(chat_id: str) -> str:
    """Create and return the directory path for ``chat_id``."""
    path = os.path.join(CHATS_DIR, chat_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Global Prompts CRUD ─────────────────────────────────────────────────
def _prompt_path(name: str) -> str:
    """Return the filesystem path for a prompt ``name``."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return os.path.join(GLOBAL_PROMPTS_DIR, f"{safe}.json")


def load_global_prompts():
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    prompts = []
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


def save_global_prompt(prompt: Dict[str, str]):
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    path = _prompt_path(prompt["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"name": prompt["name"], "content": prompt["content"]},
            f,
            indent=2,
            ensure_ascii=False,
        )


def delete_global_prompt(name: str):
    path = _prompt_path(name)
    if os.path.exists(path):
        os.remove(path)


# ========== Standard Chat Endpoints ==========
@app.get("/chats")
def list_chats():
    os.makedirs(CHATS_DIR, exist_ok=True)
    chat_ids = [
        d for d in os.listdir(CHATS_DIR) if os.path.isdir(os.path.join(CHATS_DIR, d))
    ]
    return {"chats": chat_ids}


@app.post("/chats/{chat_id}")
def create_chat(chat_id: str):
    """Create an empty chat directory for ``chat_id``."""
    chat_dir = os.path.join(CHATS_DIR, chat_id)
    if os.path.exists(chat_dir):
        raise HTTPException(status_code=400, detail="Chat already exists")
    os.makedirs(chat_dir, exist_ok=True)
    save_json(chat_file(chat_id, "full.json"), [])
    return {"detail": f"Created chat '{chat_id}'"}


@app.get("/history/{chat_id}")
def get_history(chat_id: str):
    path = chat_file(chat_id, "full.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chat not found")
    return load_json(path)


@app.put("/history/{chat_id}/{index}")
def edit_message(chat_id: str, index: int, data: Dict[str, str]):
    """Update the content of a message at ``index`` in ``chat_id``."""
    full_path = chat_file(chat_id, "full.json")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Chat not found")
    full = load_json(full_path)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full[index]["content"] = data.get("content", "")
    save_json(full_path, full)

    return {"detail": "Updated"}


@app.delete("/history/{chat_id}/{index}")
def delete_message(chat_id: str, index: int):
    """Remove the message at ``index`` from ``chat_id``."""
    full_path = chat_file(chat_id, "full.json")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Chat not found")
    full = load_json(full_path)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full.pop(index)
    save_json(full_path, full)

    return {"detail": "Deleted"}


@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str):
    chat_dir = os.path.join(CHATS_DIR, chat_id)
    if not os.path.isdir(chat_dir):
        raise HTTPException(status_code=404, detail="Chat not found")
    for fname in os.listdir(chat_dir):
        os.remove(os.path.join(chat_dir, fname))
    os.rmdir(chat_dir)
    return {"detail": f"Deleted chat '{chat_id}'"}


@app.put("/chat/{chat_id}")
def rename_chat(chat_id: str, data: Dict[str, str]):
    """Rename an existing chat ``chat_id`` to ``data['new_id']``.

    Renames the associated history files on disk.  If no history files exist
    yet for ``chat_id``, a ``404`` is returned.  If files for the new name
    already exist, a ``400`` is returned.
    """

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


@app.post("/message")
def save_message(req: ChatRequest):
    """Store ``req.message`` in ``req.chat_id`` without generating a reply."""

    ensure_chat_dir(req.chat_id)
    path = chat_file(req.chat_id, "full.json")
    history = load_json(path)
    history.append({"role": "user", "content": req.message})
    save_json(path, history)

    return {"detail": "Message stored"}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Return a streamed placeholder assistant response."""

    ensure_chat_dir(req.chat_id)
    path = chat_file(req.chat_id, "full.json")
    history = load_json(path)
    history.append({"role": "user", "content": req.message})
    assistant_reply = "[Model call disabled]"
    history.append({"role": "assistant", "content": assistant_reply})
    save_json(path, history)

    def generate():
        meta = {"prompt": req.global_prompt or ""}
        yield json.dumps(meta) + "\n"
        for line in assistant_reply.splitlines():
            yield line + "\n"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/chat")
def chat(req: ChatRequest):
    """Compatibility endpoint that returns a simple message."""

    return {"detail": "Model call disabled"}


# ========== Static UI Mount ==========
app.mount("/", StaticFiles(directory="ui", html=True), name="static")
