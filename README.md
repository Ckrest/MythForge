# MythForge

MythForge is a local chat server built on top of [FastAPI](https://fastapi.tiangolo.com/) and `llama_cpp`. It exposes a simple web UI and a JSON API for interactive story generation and experimentation with llama models.

## Features

- Web interface served from the `ui/` directory
- JSON API for programmatic chat access
- Pluggable system prompts and goal oriented prompts
- Persistence of chats, prompts and logs on disk
- Configurable model launch parameters via `model_settings.json`

## Directory overview

- `mythforge/` – Python backend implementation
- `ui/` – Static files for the browser UI
- `models/` – place your `.gguf` model files here
- `chats/` – per-chat history storage
- `global_prompts/` – system prompt files
- `server_logs/` – JSON event logs

## Requirements

MythForge requires Python 3.10 or later. Install dependencies using pip:

```bash
pip install fastapi uvicorn llama-cpp-python pydantic
```

Additional packages may be needed depending on your configuration.

## Running

1. Copy or symlink your llama model into the `models/` directory.
2. Launch the server:

```bash
python -m uvicorn mythforge.main:app --host 0.0.0.0 --port 8000
```

3. Open `http://localhost:8000/` in your browser to access the UI.

## Customisation

Model parameters can be changed by editing `model_settings.json`. Global prompts can be added under `global_prompts/` and will be available in the UI. Logs are stored under `server_logs/`.

## License

This project is distributed under the terms of the MIT license.
