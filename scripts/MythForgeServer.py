import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from queue import Queue
from threading import Thread, Lock
from .disable import log_function
from pydantic import BaseModel
from typing import Dict, List

from . import goals as goal_tracker
from .goals import (
    load_state,
    ensure_initial_state,
    init_state_from_prompt,
    check_and_generate_goals,
    state_as_prompt_fragment,
    record_user_message,
    record_assistant_message,
    evaluate_and_update_goals,
)

# Prompt formatting utilities
from .model_call import format_prompt
from . import model_launch

app = FastAPI(title="Myth Forge Server")

# ========== CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Data Models ==========
class ChatRequest(BaseModel):
    chat_id: str
    message: str
    global_prompt: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None


class ChatResponse(BaseModel):
    response: str
    prompt_preview: str


class PromptItem(BaseModel):
    name: str
    content: str


# ========== Configuration ==========
CHATS_DIR = "chats"
GLOBAL_PROMPTS_DIR = "global_prompts"

# Import model configuration and helpers
MODEL_SETTINGS_PATH = model_launch.MODEL_SETTINGS_PATH
MODEL_SETTINGS = model_launch.MODEL_SETTINGS
GENERATION_CONFIG = model_launch.GENERATION_CONFIG
DEFAULT_MAX_TOKENS = model_launch.DEFAULT_MAX_TOKENS
call_llm = model_launch.call_llm


# Helper utilities for per-chat directories
def chat_file(chat_id: str, filename: str) -> str:
    """Return the path for ``filename`` within ``chat_id``'s directory."""
    return os.path.join(CHATS_DIR, chat_id, filename)


def ensure_chat_dir(chat_id: str) -> str:
    """Create and return the directory path for ``chat_id``."""
    path = os.path.join(CHATS_DIR, chat_id)
    os.makedirs(path, exist_ok=True)
    return path


# ========== Response Prompt Queue ==========
# Some operations (e.g. summarization, goal evaluation) trigger their own
# prompts after a response is generated. Running these concurrently can
# conflict with the single shared LLM instance. A simple queue ensures these
# follow-up prompts execute sequentially.

RESPONSE_PROMPT_QUEUE: Queue = Queue()
_RESP_PENDING = 0
_RESP_LOCK = Lock()


def _response_prompt_worker() -> None:
    while True:
        fn = RESPONSE_PROMPT_QUEUE.get()
        if fn is None:
            break
        try:
            fn()
        except Exception as e:
            print(f"[response_prompt_queue] task failed: {e}")
        finally:
            with _RESP_LOCK:
                globals()["_RESP_PENDING"] = max(0, _RESP_PENDING - 1)
            RESPONSE_PROMPT_QUEUE.task_done()


_RESPONSE_PROMPT_THREAD = Thread(target=_response_prompt_worker, daemon=True)
_RESPONSE_PROMPT_THREAD.start()


def enqueue_response_prompt(fn) -> None:
    """Add ``fn`` to the post-response processing queue."""
    with _RESP_LOCK:
        globals()["_RESP_PENDING"] += 1
    RESPONSE_PROMPT_QUEUE.put(fn)


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


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def strip_leading_tag(text: str, tag: str) -> str:
    """Remove a leading ``tag:`` from ``text`` if present."""
    text = text.lstrip()
    prefix = f"{tag}:"
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix) :].lstrip()
    return text


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


@app.get("/prompts")
def list_prompts(names_only: bool = False):
    if names_only:
        return {"prompts": list_prompt_names()}
    return {"prompts": load_global_prompts()}


@app.get("/prompts/{name}")
def fetch_prompt(name: str):
    content = get_global_prompt_content(name)
    if content is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"name": name, "content": content}


@app.post("/prompts")
def create_prompt(item: PromptItem):
    if os.path.exists(_prompt_path(item.name)):
        raise HTTPException(status_code=400, detail="Prompt name already exists")
    save_global_prompt({"name": item.name, "content": item.content})
    return {
        "detail": "Created",
        "prompt": {"name": item.name, "content": item.content},
    }


@app.put("/prompts/{name}")
def update_prompt(name: str, item: PromptItem):
    if item.name != name:
        raise HTTPException(
            status_code=400,
            detail="Cannot rename prompt; delete & recreate instead",
        )
    if not os.path.exists(_prompt_path(name)):
        raise HTTPException(status_code=404, detail="Prompt not found")
    save_global_prompt({"name": name, "content": item.content})
    return {
        "detail": "Updated",
        "prompt": {"name": name, "content": item.content},
    }


@app.put("/prompts/{name}/rename")
def rename_prompt(name: str, data: Dict[str, str]):
    """Rename a prompt ``name`` to ``data['new_name']``."""
    new_name = data.get("new_name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name required")
    if not os.path.exists(_prompt_path(name)):
        raise HTTPException(status_code=404, detail="Prompt not found")
    if os.path.exists(_prompt_path(new_name)) and new_name != name:
        raise HTTPException(status_code=400, detail="Prompt name already exists")

    same_name = new_name == name

    old_path = _prompt_path(name)
    new_path = _prompt_path(new_name)
    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="Prompt not found")
    with open(old_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["name"] = new_name
    if not same_name:
        os.rename(old_path, new_path)
    with open(new_path if not same_name else old_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"detail": f"Renamed prompt '{name}' to '{new_name}'"}


@app.delete("/prompts/{name}")
def delete_prompt(name: str):
    if not os.path.exists(_prompt_path(name)):
        raise HTTPException(status_code=404, detail="Prompt not found")
    delete_global_prompt(name)
    return {"detail": f"Deleted prompt '{name}'"}


@log_function("state_writer_caller")
def build_prompt(chat_id, user_message, global_prompt_name):
    """Construct the next prompt using the LLaMA3 header-token format."""

    ensure_chat_dir(chat_id)
    full_path = chat_file(chat_id, "full.json")

    # Append user message to history if it is not empty.  When generating with
    # an empty prompt, we do not want to record a blank user message.
    full_log = load_json(full_path)
    if user_message.strip():
        full_log.append({"role": "user", "content": user_message})
        record_user_message(chat_id)
        chosen_content = get_global_prompt_content(global_prompt_name)
        if chosen_content and "**goals**" in chosen_content and len(full_log) == 1:
            init_state_from_prompt(chat_id, chosen_content, user_message)
        state = load_state(chat_id)
        if goal_tracker.DEBUG_MODE or state.get("messages_since_goal_eval", 0) >= 2:
            check_and_generate_goals(call_llm, chat_id)
    # Ensure history files exist even if no message was appended
    save_json(full_path, full_log)

    # Determine system prompt
    chosen_content = get_global_prompt_content(global_prompt_name)
    system_prompt = chosen_content if chosen_content else "You are a helpful assistant."
    assistant_name = "assistant"

    # Incorporate character state and goals if available
    state = load_state(chat_id)
    fragment = state_as_prompt_fragment(state)
    if fragment:
        system_prompt = system_prompt + "\n" + fragment

    # Convert full history into role/content pairs
    history: List[Dict[str, str]] = []
    for m in full_log:
        role = assistant_name if m.get("role") == "bot" else m.get("role")
        history.append({"role": role, "content": m.get("content", "")})

    prompt_str = format_prompt(system_prompt, history, assistant_name)
    return prompt_str, assistant_name


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


@app.post("/chat")
@log_function("state_writer_caller")
def chat(req: ChatRequest):
    chat_id = req.chat_id
    user_message = req.message
    global_prompt = req.global_prompt or ""

    ensure_chat_dir(chat_id)
    full_path = chat_file(chat_id, "full.json")
    if not os.path.exists(full_path):
        save_json(full_path, [])

    prompt, assistant_name = build_prompt(chat_id, user_message, global_prompt)

    config = GENERATION_CONFIG.copy()
    if req.temperature is not None:
        config["temperature"] = req.temperature
    if req.top_k is not None:
        config["top_k"] = req.top_k
    if req.top_p is not None:
        config["top_p"] = req.top_p
    if req.min_p is not None:
        config["min_p"] = req.min_p
    if req.repeat_penalty is not None:
        config["repeat_penalty"] = req.repeat_penalty
    max_tokens = req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS

    print("DEBUG raw_prompt:", prompt)
    output = call_llm(prompt, max_tokens=max_tokens, **config)
    response_text = output["choices"][0]["text"].strip()
    response_text = strip_leading_tag(response_text, assistant_name)

    # Append bot to full history
    full_log = load_json(full_path)
    full_log.append({"role": "bot", "content": response_text})
    save_json(full_path, full_log)
    if record_assistant_message(chat_id):
        enqueue_response_prompt(lambda: evaluate_and_update_goals(call_llm, chat_id))
    if len(full_log) == 2:
        first_user = full_log[0].get("content", "") if full_log else ""
        gprompt_content = get_global_prompt_content(global_prompt) or ""
        enqueue_response_prompt(
            lambda fu=first_user, gp=gprompt_content, rt=response_text: ensure_initial_state(
                call_llm, chat_id, gp, fu, rt
            )
        )
        if "**goals**" in gprompt_content:
            check_and_generate_goals(call_llm, chat_id)

    return ChatResponse(response=response_text, prompt_preview=prompt)


# ─── Streaming Chat Endpoint with Prompt JSON Prefix ────────────────────────
@app.post("/chat/stream")
@log_function("state_writer_caller")
def chat_stream(req: ChatRequest):
    """
    Streams tokens as the model generates them.
    ``build_prompt()`` records the user's message to both history files,
    so this function only streams the assistant's response and then saves it.
    The first line sent to the client is a JSON object with the prompt text.
    All subsequent data are token chunks from the model.
    """
    chat_id = req.chat_id
    user_message = req.message
    global_prompt = req.global_prompt or ""

    ensure_chat_dir(chat_id)
    full_path = chat_file(chat_id, "full.json")
    if not os.path.exists(full_path):
        save_json(full_path, [])

    # 1) Build prompt
    prompt, assistant_name = build_prompt(chat_id, user_message, global_prompt)

    # build_prompt() already saved the user's message to both history files,
    # so we can immediately begin streaming the model's response.

    @log_function("state_writer_caller")
    def generate_and_stream():
        # First, send a single line of JSON with the prompt:
        meta = json.dumps({"prompt": prompt}, ensure_ascii=False)
        # Send one line of JSON metadata followed by raw token text
        yield meta + "\n"

        text_accumulator = ""
        pending = ""
        prefix_trimmed = False
        prefix = f"{assistant_name}:"

        print("DEBUG raw_prompt:", prompt)
        config = GENERATION_CONFIG.copy()
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.top_k is not None:
            config["top_k"] = req.top_k
        if req.top_p is not None:
            config["top_p"] = req.top_p
        if req.min_p is not None:
            config["min_p"] = req.min_p
        if req.repeat_penalty is not None:
            config["repeat_penalty"] = req.repeat_penalty
        max_tokens = (
            req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS
        )

        for output in call_llm(
            prompt,
            max_tokens=max_tokens,
            stream=True,
            **config,
        ):
            chunk = output["choices"][0]["text"]
            if prefix_trimmed:
                text_accumulator += chunk
                yield chunk
                continue

            pending += chunk
            check = pending.lstrip()
            if check.lower().startswith(prefix.lower()):
                if len(check) > len(prefix):
                    trimmed = strip_leading_tag(check, assistant_name)
                    text_accumulator += trimmed
                    yield trimmed
                    prefix_trimmed = True
                    pending = ""
            else:
                text_accumulator += pending
                yield pending
                prefix_trimmed = True
                pending = ""

        # Once done, append full bot response to both histories:
        text_accumulator = strip_leading_tag(text_accumulator, assistant_name)

        # (1) Full history
        full_history = load_json(full_path)
        full_history.append({"role": "bot", "content": text_accumulator})
        save_json(full_path, full_history)
        if record_assistant_message(chat_id):
            enqueue_response_prompt(
                lambda: evaluate_and_update_goals(call_llm, chat_id)
            )

        # Goal initialization if this was the first exchange
        if len(full_history) == 2:
            first_user = full_history[0].get("content", "") if full_history else ""
            gprompt_content = get_global_prompt_content(global_prompt) or ""
            enqueue_response_prompt(
                lambda fu=first_user, gp=gprompt_content, rt=text_accumulator: ensure_initial_state(
                    call_llm, chat_id, gp, fu, rt
                )
            )
            if "**goals**" in gprompt_content:
                check_and_generate_goals(call_llm, chat_id)

        return StreamingResponse(generate_and_stream(), media_type="text/plain")


# ─── Endpoint: Log Message Without Generating ------------------------------
@app.post("/message")
@log_function("state_writer_caller")
def log_message(req: ChatRequest):
    """Record a user message without generating a response."""

    chat_id = req.chat_id
    user_message = req.message
    global_prompt = req.global_prompt or ""

    ensure_chat_dir(chat_id)
    full_path = chat_file(chat_id, "full.json")
    if not os.path.exists(full_path):
        save_json(full_path, [])

    # build_prompt appends the user message to history files
    build_prompt(chat_id, user_message, global_prompt)

    return {"detail": "Message recorded"}


# ========== Settings Endpoints ==========
@app.get("/settings")
def get_settings():
    return MODEL_SETTINGS


@app.put("/settings")
def update_settings(data: Dict[str, object]):
    """Update settings and persist them to ``model_settings.json``."""
    MODEL_SETTINGS.update(data)
    save_json(MODEL_SETTINGS_PATH, MODEL_SETTINGS)
    for key in (
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repeat_penalty",
        "stop",
    ):
        if key in MODEL_SETTINGS:
            GENERATION_CONFIG[key] = MODEL_SETTINGS[key]
    global DEFAULT_MAX_TOKENS
    DEFAULT_MAX_TOKENS = MODEL_SETTINGS.get("max_tokens", DEFAULT_MAX_TOKENS)
    return {"detail": "Updated", "settings": MODEL_SETTINGS}


# ========== Response Prompt Status ==========
@app.get("/response_prompt_status")
def response_prompt_status():
    """Return the number of queued or running response prompts."""
    with _RESP_LOCK:
        pending = _RESP_PENDING
    return {"pending": pending}


# ========== Static UI Mount ==========
app.mount("/", StaticFiles(directory="ui", html=True), name="static")

# Apply automatic logging to all functions in this module
import sys
from .disable import patch_module_functions

patch_module_functions(sys.modules[__name__], "server")
