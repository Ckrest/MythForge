import os
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
from lmstudio_prompter import format_prompt, GENERATION_CONFIG

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
    chat_id: str
    sanitized_id: str

class PromptItem(BaseModel):
    name: str
    content: str

# ========== Configuration ==========
MODELS_DIR            = "models"
CHATS_DIR             = "chats"
CHAT_MAP_FILE         = os.path.join(CHATS_DIR, "chat_map.json")
INJECTION_FILE        = "random_injections.txt"
GLOBAL_PROMPTS_FILE   = "global_prompts.json"
DEFAULT_CTX_SIZE      = 2048
SUMMARIZE_THRESHOLD   = 12
SUMMARIZE_BATCH       = 6

INVALID_PATH_CHARS = r'[<>:"/\\|?*]'

def sanitize_chat_id(chat_id: str) -> str:
    """Return a filesystem-safe version of ``chat_id``."""
    import re
    return re.sub(INVALID_PATH_CHARS, "_", chat_id)

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
)

# ========== Helpers ==========
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_chat_map():
    if os.path.exists(CHAT_MAP_FILE):
        with open(CHAT_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_chat_map(mapping):
    os.makedirs(CHATS_DIR, exist_ok=True)
    with open(CHAT_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

def ensure_chat_mapping(sanitized: str, original: str):
    mapping = load_chat_map()
    if mapping.get(sanitized) != original:
        mapping[sanitized] = original
        save_chat_map(mapping)

def get_injection():
    if not os.path.exists(INJECTION_FILE):
        return ""
    with open(INJECTION_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return random.choice(lines) if lines else ""

# ─── Global Prompts CRUD ─────────────────────────────────────────────────
def load_global_prompts():
    if not os.path.exists(GLOBAL_PROMPTS_FILE):
        with open(GLOBAL_PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []

    with open(GLOBAL_PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Ignoring malformed global prompts: expected list, got {type(data)}")
        return []

    sanitized = []
    for entry in data:
        if isinstance(entry, dict) and "name" in entry and "content" in entry:
            sanitized.append({"name": entry["name"], "content": entry["content"]})
        else:
            # Ignore invalid entries but log for visibility
            print(f"Ignoring invalid global prompt entry: {entry}")

    return sanitized

def save_global_prompts(prompts):
    with open(GLOBAL_PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

@app.get("/prompts")
def list_prompts():
    return {"prompts": load_global_prompts()}

@app.post("/prompts")
def create_prompt(item: PromptItem):
    prompts = load_global_prompts()
    if any(p["name"] == item.name for p in prompts):
        raise HTTPException(status_code=400, detail="Prompt name already exists")
    prompts.append({"name": item.name, "content": item.content})
    save_global_prompts(prompts)
    return {"detail": "Created", "prompt": {"name": item.name, "content": item.content}}

@app.put("/prompts/{name}")
def update_prompt(name: str, item: PromptItem):
    prompts = load_global_prompts()
    for p in prompts:
        if p["name"] == name:
            if item.name != name:
                raise HTTPException(status_code=400, detail="Cannot rename prompt; delete & recreate instead")
            p["content"] = item.content
            save_global_prompts(prompts)
            return {"detail": "Updated", "prompt": p}
    raise HTTPException(status_code=404, detail="Prompt not found")

@app.delete("/prompts/{name}")
def delete_prompt(name: str):
    prompts = load_global_prompts()
    filtered = [p for p in prompts if p["name"] != name]
    if len(filtered) == len(prompts):
        raise HTTPException(status_code=404, detail="Prompt not found")
    save_global_prompts(filtered)
    return {"detail": f"Deleted prompt '{name}'"}

# ========== Summarization & Trimming ==========
def summarize_chunk(chunk):
    text = "\n".join([f"{m['role']}: {m['content']}" for m in chunk])
    prompt = (
        f"BEGININPUT\n{text}\nENDINPUT\n"
        f"BEGININSTRUCTION\nSummarize the above conversation in 2-3 sentences.\nENDINSTRUCTION"
    )
    output = llm(prompt, max_tokens=150)
    return {"type": "summary", "content": output["choices"][0]["text"].strip()}

def trim_context(chat_id):
    chat_id = sanitize_chat_id(chat_id)
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    history = load_json(trimmed_path)
    raw_msgs = [m for m in history if m.get("type") == "raw"]
    summaries = [m for m in history if m.get("type") == "summary"]

    if len(raw_msgs) >= SUMMARIZE_THRESHOLD:
        oldest_batch = raw_msgs[:SUMMARIZE_BATCH]
        summary = summarize_chunk(oldest_batch)
        raw_msgs = raw_msgs[SUMMARIZE_BATCH:]
        new_history = summaries + [summary] + raw_msgs
        save_json(trimmed_path, new_history)
        return new_history

    return history

def build_prompt(chat_id, user_message, message_index, global_prompt_name):
    chat_id = sanitize_chat_id(chat_id)
    trimmed_path = f"{CHATS_DIR}/{chat_id}_trimmed.json"
    full_path    = f"{CHATS_DIR}/{chat_id}_full.json"

    # Append user to full history
    full_log = load_json(full_path)
    full_log.append({"role": "user", "content": user_message})
    save_json(full_path, full_log)

    # Possibly summarize oldest raws
    context = trim_context(chat_id)
    context.append({"type": "raw", "role": "user", "content": user_message})
    save_json(trimmed_path, context)

    # Choose system prompt
    all_prompts    = load_global_prompts()
    chosen_content = next((p["content"] for p in all_prompts if p["name"] == global_prompt_name), None)
    system_prompt  = chosen_content if chosen_content else "You are a helpful assistant."

    # Random injection every other message
    injection = get_injection() if message_index % 2 == 0 else ""

    # Build the message list for the LM Studio template
    system_content = system_prompt + ("\n" + injection if injection else "")
    messages = [{"role": "system", "content": system_content}]

    for m in context:
        if m.get("type") == "summary":
            messages.append({"role": "system", "content": f"SUMMARY: {m['content']}"})
        else:
            role = "assistant" if m["role"] == "bot" else m["role"]
            messages.append({"role": role, "content": m["content"]})

    prompt_str = format_prompt({"messages": messages}, bos_token="<s>")
    return prompt_str

# ========== Standard Chat Endpoints ==========
@app.get("/chats")
def list_chats():
    os.makedirs(CHATS_DIR, exist_ok=True)
    files = os.listdir(CHATS_DIR)
    sanitized_ids = [f[:-len("_full.json")] for f in files if f.endswith("_full.json")]
    mapping = load_chat_map()
    changed = False
    chats = []
    for sid in sanitized_ids:
        original = mapping.get(sid)
        if original is None:
            original = sid
            mapping[sid] = original
            changed = True
        chats.append({"id": original, "sanitized_id": sid})
    if changed:
        save_chat_map(mapping)
    return {"chats": chats}

@app.get("/history/{chat_id}")
def get_history(chat_id: str):
    chat_id = sanitize_chat_id(chat_id)
    path = f"{CHATS_DIR}/{chat_id}_full.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chat not found")
    return load_json(path)

@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str):
    sanitized = sanitize_chat_id(chat_id)
    full_path    = f"{CHATS_DIR}/{sanitized}_full.json"
    trimmed_path = f"{CHATS_DIR}/{sanitized}_trimmed.json"
    mapping = load_chat_map()
    original = mapping.pop(sanitized, sanitized)
    existed = False
    if os.path.exists(full_path):
        os.remove(full_path)
        existed = True
    if os.path.exists(trimmed_path):
        os.remove(trimmed_path)
        existed = True
    if existed:
        save_chat_map(mapping)
    if not existed:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"detail": "Deleted chat", "chat": {"id": original, "sanitized_id": sanitized}}

@app.post("/chat")
def chat(req: ChatRequest):
    sanitized_id  = sanitize_chat_id(req.chat_id)
    user_message  = req.message
    global_prompt = req.global_prompt or ""
    ensure_chat_mapping(sanitized_id, req.chat_id)

    os.makedirs(CHATS_DIR, exist_ok=True)
    full_path    = f"{CHATS_DIR}/{sanitized_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{sanitized_id}_trimmed.json"
    if not os.path.exists(full_path):
        save_json(full_path, [])
        save_json(trimmed_path, [])

    message_index = len(load_json(full_path))
    prompt = build_prompt(sanitized_id, user_message, message_index, global_prompt)

    output = llm(
        prompt,
        max_tokens=250,
        temperature=GENERATION_CONFIG["temperature"],
        top_k=GENERATION_CONFIG["top_k"],
        top_p=GENERATION_CONFIG["top_p"],
        min_p=GENERATION_CONFIG["min_p"],
        repeat_penalty=GENERATION_CONFIG["repeat_penalty"],
        stop=GENERATION_CONFIG["stop"],
    )
    response_text = output["choices"][0]["text"].strip()

    # Append bot to full history
    full_log = load_json(full_path)
    full_log.append({"role": "bot", "content": response_text})
    save_json(full_path, full_log)

    # Append bot to trimmed context
    trimmed = load_json(trimmed_path)
    trimmed.append({"type": "raw", "role": "bot", "content": response_text})
    save_json(trimmed_path, trimmed)

    return ChatResponse(
        response=response_text,
        prompt_preview=prompt,
        chat_id=req.chat_id,
        sanitized_id=sanitized_id,
    )

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
    sanitized_id  = sanitize_chat_id(req.chat_id)
    user_message  = req.message
    global_prompt = req.global_prompt or ""
    ensure_chat_mapping(sanitized_id, req.chat_id)

    os.makedirs(CHATS_DIR, exist_ok=True)
    full_path    = f"{CHATS_DIR}/{sanitized_id}_full.json"
    trimmed_path = f"{CHATS_DIR}/{sanitized_id}_trimmed.json"
    if not os.path.exists(full_path):
        save_json(full_path, [])
        save_json(trimmed_path, [])

    # 1) Build prompt
    message_index = len(load_json(full_path))
    prompt = build_prompt(sanitized_id, user_message, message_index, global_prompt)


    # build_prompt() already saved the user's message to both history files,
    # so we can immediately begin streaming the model's response.

    def generate_and_stream():
        # First, send a single line of JSON with the prompt:
        meta = json.dumps({
            "prompt": prompt,
            "chat_id": req.chat_id,
            "sanitized_id": sanitized_id,
        }, ensure_ascii=False)
        yield meta + "\n"

        # Prepare to accumulate the bot’s full response
        text_accumulator = ""

        # Next, stream the model’s token chunks
        for output in llm(
            prompt,
            max_tokens=250,
            stream=True,
            temperature=GENERATION_CONFIG["temperature"],
            top_k=GENERATION_CONFIG["top_k"],
            top_p=GENERATION_CONFIG["top_p"],
            min_p=GENERATION_CONFIG["min_p"],
            repeat_penalty=GENERATION_CONFIG["repeat_penalty"],
            stop=GENERATION_CONFIG["stop"],
        ):
            chunk = output["choices"][0]["text"]
            text_accumulator += chunk
            yield chunk

        # Once done, append full bot response to both histories:

        # (1) Full history
        full_history = load_json(full_path)
        full_history.append({"role": "bot", "content": text_accumulator})
        save_json(full_path, full_history)

        # (2) Trimmed context
        trimmed = load_json(trimmed_path)
        trimmed.append({"type": "raw", "role": "bot", "content": text_accumulator})
        save_json(trimmed_path, trimmed)

    return StreamingResponse(
        generate_and_stream(),
        media_type="text/plain"
    )

# ========== Static UI Mount ==========
app.mount("/", StaticFiles(directory="static", html=True), name="static")
