import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List
from llama_cpp import Llama

# Prompt formatting utilities
from airoboros_prompter import format_airoboros

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

class ChatResponse(BaseModel):
    response: str
    prompt_preview: str

class PromptItem(BaseModel):
    name: str
    content: str

# ========== Configuration ==========
MODELS_DIR            = "models"
CHATS_DIR             = "chats"
GLOBAL_PROMPTS_DIR    = "Global_prompts"
# Match LM Studio defaults for a 4 GB VRAM setup
DEFAULT_CTX_SIZE      = 4096
DEFAULT_N_BATCH       = 512
DEFAULT_N_THREADS     = os.cpu_count() or 1
SUMMARIZE_THRESHOLD   = 20
SUMMARIZE_BATCH       = 12

# Generation settings matching LM Studio's defaults
GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    # ``n_batch`` is set when ``Llama`` is instantiated. Passing it to
    # ``Llama.__call__`` can break older ``llama_cpp`` versions, so it is
    # intentionally omitted here.
    "stop": ["<|start_header_id|>", "<|eot_id|>"],
}

# ========== Model Loading ==========
def discover_model_path():
    """Return the path to the first ``.gguf`` model file under ``MODELS_DIR``.

    Previously the code assumed models were stored in subdirectories and that
    each subdirectory contained a file named ``model.gguf``.  Users often place
    the downloaded model file directly under ``models`` or use a different
    filename.  This function now walks the directory tree and returns the first
    file with a ``.gguf`` extension regardless of its directory structure.
    """

    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory '{MODELS_DIR}' not found")

    for root, _dirs, files in os.walk(MODELS_DIR):
        for fname in files:
            if fname.lower().endswith(".gguf"):
                return os.path.join(root, fname)

    raise FileNotFoundError(f"No .gguf model files found under '{MODELS_DIR}'")


llm = Llama(
    model_path=discover_model_path(),
    n_ctx=DEFAULT_CTX_SIZE,
    n_batch=DEFAULT_N_BATCH,
    n_threads=DEFAULT_N_THREADS,
    prompt_template="",
)

# Determine which keyword arguments ``llm.__call__`` accepts.  Older versions
# of ``llama_cpp`` do not support certain generation parameters (e.g.
# ``n_batch``).  We inspect the call signature once so we can filter unsupported
# keys when generating text.
try:
    import inspect

    _CALL_KWARGS = set(inspect.signature(llm.__call__).parameters)
except Exception:  # pragma: no cover - signature introspection may fail
    _CALL_KWARGS = set()


def call_llm(prompt: str, **kwargs):
    """Call the ``llm`` object with only the supported keyword arguments."""

    if _CALL_KWARGS:
        filtered = {k: v for k, v in kwargs.items() if k in _CALL_KWARGS}
    else:  # Fallback if inspection failed
        filtered = kwargs
    return llm(prompt, **filtered)

# ========== Helpers ==========
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def strip_leading_tag(text: str, tag: str) -> str:
    """Remove a leading ``tag:`` from ``text`` if present."""
    text = text.lstrip()
    prefix = f"{tag}:"
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):].lstrip()
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


def save_global_prompt(prompt: Dict[str, str]):
    os.makedirs(GLOBAL_PROMPTS_DIR, exist_ok=True)
    path = _prompt_path(prompt["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": prompt["name"], "content": prompt["content"]}, f, indent=2, ensure_ascii=False)


def delete_global_prompt(name: str):
    path = _prompt_path(name)
    if os.path.exists(path):
        os.remove(path)

@app.get("/prompts")
def list_prompts():
    return {"prompts": load_global_prompts()}

@app.post("/prompts")
def create_prompt(item: PromptItem):
    prompts = load_global_prompts()
    if any(p["name"] == item.name for p in prompts):
        raise HTTPException(status_code=400, detail="Prompt name already exists")
    save_global_prompt({"name": item.name, "content": item.content})
    return {"detail": "Created", "prompt": {"name": item.name, "content": item.content}}

@app.put("/prompts/{name}")
def update_prompt(name: str, item: PromptItem):
    if item.name != name:
        raise HTTPException(status_code=400, detail="Cannot rename prompt; delete & recreate instead")
    prompts = load_global_prompts()
    if not any(p["name"] == name for p in prompts):
        raise HTTPException(status_code=404, detail="Prompt not found")
    save_global_prompt({"name": name, "content": item.content})
    return {"detail": "Updated", "prompt": {"name": name, "content": item.content}}

@app.delete("/prompts/{name}")
def delete_prompt(name: str):
    prompts = load_global_prompts()
    if not any(p["name"] == name for p in prompts):
        raise HTTPException(status_code=404, detail="Prompt not found")
    delete_global_prompt(name)
    return {"detail": f"Deleted prompt '{name}'"}

# ========== Summarization & Trimming ==========
def summarize_chunk(chunk):
    text = "\n".join([f"{m['role']}: {m['content']}" for m in chunk])
    prompt = (
        f"BEGININPUT\n{text}\nENDINPUT\n"
        f"BEGININSTRUCTION\nSummarize the above conversation in 2-3 sentences.\nENDINSTRUCTION"
    )
    # Use ``call_llm`` to ensure only supported kwargs are passed to the model
    output = call_llm(prompt, max_tokens=150)
    return {"type": "summary", "content": output["choices"][0]["text"].strip()}

def trim_context(chat_id):
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    history = load_json(trimmed_path)

    # Collect indices of raw messages so we know where to insert summaries
    raw_indices = [i for i, m in enumerate(history) if m.get("type") == "raw"]

    if len(raw_indices) >= SUMMARIZE_THRESHOLD:
        # Grab the earliest batch of raw messages to summarize
        batch_indices = raw_indices[:SUMMARIZE_BATCH]
        oldest_batch = [history[i] for i in batch_indices]

        summary = summarize_chunk(oldest_batch)

        # Remove the raw messages we just summarized
        for i in reversed(batch_indices):
            history.pop(i)

        # Insert the summary at the position of the first removed message
        history.insert(batch_indices[0], summary)

        save_json(trimmed_path, history)

    return history

def build_prompt(chat_id, user_message, message_index, global_prompt_name):
    """Construct the next prompt using the Airoboros format."""

    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    full_path = f"{CHATS_DIR}/{chat_id}_full.json"

    # Append user message to full history
    full_log = load_json(full_path)
    full_log.append({"role": "user", "content": user_message})
    save_json(full_path, full_log)

    # Load existing trimmed context and append the new user message.
    # Summarization is handled separately after the assistant responds so
    # the full user+assistant exchange can be summarized together.
    context = load_json(trimmed_path)
    context.append({"type": "raw", "role": "user", "content": user_message})
    save_json(trimmed_path, context)

    # Determine system prompt
    all_prompts = load_global_prompts()
    chosen_content = next((p["content"] for p in all_prompts if p["name"] == global_prompt_name), None)
    system_prompt = chosen_content if chosen_content else "You are a helpful assistant."
    assistant_name = global_prompt_name if chosen_content else "assistant"

    # Gather summaries and raw history for Airoboros formatting
    summaries: List[str] = []
    history: List[Dict[str, str]] = []
    for m in context:
        if m.get("type") == "summary":
            summaries.append(f"SUMMARY: {m['content']}")
        else:
            role = assistant_name if m.get("role") == "bot" else m.get("role")
            history.append({"role": role, "content": m.get("content", "")})

    prompt_str = format_airoboros(
        system_prompt,
        summaries,
        history,
        user_message,
        assistant_name,
    )
    return prompt_str, assistant_name

# ========== Standard Chat Endpoints ==========
@app.get("/chats")
def list_chats():
    os.makedirs(CHATS_DIR, exist_ok=True)
    files = os.listdir(CHATS_DIR)
    chat_ids = [fname[:-len("_full.json")] for fname in files if fname.endswith("_full.json")]
    return {"chats": chat_ids}

@app.get("/history/{chat_id}")
def get_history(chat_id: str):
    path = f"{CHATS_DIR}/{chat_id}_full.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chat not found")
    return load_json(path)

@app.put("/history/{chat_id}/{index}")
def edit_message(chat_id: str, index: int, data: Dict[str, str]):
    """Update the content of a message at ``index`` in ``chat_id``."""
    full_path = f"{CHATS_DIR}/{chat_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Chat not found")
    full = load_json(full_path)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full[index]["content"] = data.get("content", "")
    save_json(full_path, full)

    trimmed = load_json(trimmed_path)
    raw_idx = 0
    for i, m in enumerate(trimmed):
        if m.get("type") == "raw":
            if raw_idx == index:
                trimmed[i]["content"] = data.get("content", "")
                break
            raw_idx += 1
    save_json(trimmed_path, trimmed)
    return {"detail": "Updated"}

@app.delete("/history/{chat_id}/{index}")
def delete_message(chat_id: str, index: int):
    """Remove the message at ``index`` from ``chat_id``."""
    full_path = f"{CHATS_DIR}/{chat_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Chat not found")
    full = load_json(full_path)
    if index < 0 or index >= len(full):
        raise HTTPException(status_code=400, detail="Invalid index")
    full.pop(index)
    save_json(full_path, full)

    trimmed = load_json(trimmed_path)
    raw_idx = 0
    for i, m in enumerate(trimmed):
        if m.get("type") == "raw":
            if raw_idx == index:
                trimmed.pop(i)
                break
            raw_idx += 1
    save_json(trimmed_path, trimmed)
    return {"detail": "Deleted"}

@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str):
    full_path    = f"{CHATS_DIR}/{chat_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    existed = False
    if os.path.exists(full_path):
        os.remove(full_path)
        existed = True
    if os.path.exists(trimmed_path):
        os.remove(trimmed_path)
        existed = True
    if not existed:
        raise HTTPException(status_code=404, detail="Chat not found")
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

    old_full = f"{CHATS_DIR}/{chat_id}_full.json"
    old_trim = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    if not (os.path.exists(old_full) or os.path.exists(old_trim)):
        raise HTTPException(status_code=404, detail="Chat not found")

    new_full = f"{CHATS_DIR}/{new_id}_full.json"
    new_trim = f"{CHATS_DIR}/{new_id}_trimmed.json"
    if os.path.exists(new_full) or os.path.exists(new_trim):
        raise HTTPException(status_code=400, detail="Chat name already exists")

    if os.path.exists(old_full):
        os.rename(old_full, new_full)
    if os.path.exists(old_trim):
        os.rename(old_trim, new_trim)

    return {"detail": f"Renamed chat '{chat_id}' to '{new_id}'"}

@app.post("/chat")
def chat(req: ChatRequest):
    chat_id       = req.chat_id
    user_message  = req.message
    global_prompt = req.global_prompt or ""

    os.makedirs(CHATS_DIR, exist_ok=True)
    full_path    = f"{CHATS_DIR}/{chat_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    if not os.path.exists(full_path):
        save_json(full_path, [])
        save_json(trimmed_path, [])

    message_index = len(load_json(full_path))
    prompt, assistant_name = build_prompt(
        chat_id, user_message, message_index, global_prompt
    )

    # Generate the assistant response using LM Studio generation settings
    output = call_llm(prompt, max_tokens=250, **GENERATION_CONFIG)
    response_text = output["choices"][0]["text"].strip()
    response_text = strip_leading_tag(response_text, assistant_name)

    # Append bot to full history
    full_log = load_json(full_path)
    full_log.append({"role": "bot", "content": response_text})
    save_json(full_path, full_log)

    # Append bot to trimmed context
    trimmed = load_json(trimmed_path)
    trimmed.append({"type": "raw", "role": "bot", "content": response_text})
    save_json(trimmed_path, trimmed)
    # Now that both sides of the exchange are recorded, attempt summarization
    trim_context(chat_id)

    return ChatResponse(response=response_text, prompt_preview=prompt)

# ─── Streaming Chat Endpoint with Prompt JSON Prefix ────────────────────────
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streams tokens as the model generates them.
    ``build_prompt()`` records the user's message to both history files,
    so this function only streams the assistant's response and then saves it.
    The first line sent to the client is a JSON object with the prompt text.
    All subsequent data are token chunks from the model.
    """
    chat_id       = req.chat_id
    user_message  = req.message
    global_prompt = req.global_prompt or ""

    os.makedirs(CHATS_DIR, exist_ok=True)
    full_path    = f"{CHATS_DIR}/{chat_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    if not os.path.exists(full_path):
        save_json(full_path, [])
        save_json(trimmed_path, [])

    # 1) Build prompt
    message_index = len(load_json(full_path))
    prompt, assistant_name = build_prompt(
        chat_id, user_message, message_index, global_prompt
    )


    # build_prompt() already saved the user's message to both history files,
    # so we can immediately begin streaming the model's response.

    def generate_and_stream():
        # First, send a single line of JSON with the prompt:
        meta = json.dumps({"prompt": prompt}, ensure_ascii=False)
        yield meta + "\n"

        text_accumulator = ""
        pending = ""
        prefix_trimmed = False
        prefix = f"{assistant_name}:"

        for output in call_llm(
            prompt,
            max_tokens=250,
            stream=True,
            **GENERATION_CONFIG,
        ):
            chunk = output["choices"][0]["text"]
            if prefix_trimmed:
                text_accumulator += chunk
                yield chunk + "\n"
                continue

            pending += chunk
            check = pending.lstrip()
            if check.lower().startswith(prefix.lower()):
                if len(check) > len(prefix):
                    trimmed = strip_leading_tag(check, assistant_name)
                    text_accumulator += trimmed
                    yield trimmed + "\n"
                    prefix_trimmed = True
                    pending = ""
            else:
                text_accumulator += pending
                yield pending + "\n"
                prefix_trimmed = True
                pending = ""

        # Once done, append full bot response to both histories:
        text_accumulator = strip_leading_tag(text_accumulator, assistant_name)

        # (1) Full history
        full_history = load_json(full_path)
        full_history.append({"role": "bot", "content": text_accumulator})
        save_json(full_path, full_history)

        # (2) Trimmed context
        trimmed = load_json(trimmed_path)
        trimmed.append({"type": "raw", "role": "bot", "content": text_accumulator})
        save_json(trimmed_path, trimmed)
        # Attempt summarization now that the exchange is complete
        trim_context(chat_id)

    return StreamingResponse(
        generate_and_stream(),
        media_type="text/plain"
    )

# ─── Endpoint: Log Message Without Generating ------------------------------
@app.post("/message")
def log_message(req: ChatRequest):
    """Record a user message without generating a response."""

    chat_id       = req.chat_id
    user_message  = req.message
    global_prompt = req.global_prompt or ""

    os.makedirs(CHATS_DIR, exist_ok=True)
    full_path    = f"{CHATS_DIR}/{chat_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    if not os.path.exists(full_path):
        save_json(full_path, [])
        save_json(trimmed_path, [])

    message_index = len(load_json(full_path))
    # build_prompt appends the user message to history files
    build_prompt(chat_id, user_message, message_index, global_prompt)

    return {"detail": "Message recorded"}

# ========== Static UI Mount ==========
app.mount("/", StaticFiles(directory=".", html=True), name="static")
