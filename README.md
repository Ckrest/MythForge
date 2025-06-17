# MythForge

MythForge is a local chat server built on top of [FastAPI](https://fastapi.tiangolo.com/) and `llama_cpp`. It exposes a simple web UI and a JSON API for interactive story generation and experimentation built with FASTAPI with llama models.

## Features

- Web interface served from the `ui/` directory
- Character-based conversations with persistent memory
- Automatic goals created and updated throughout chat sessions
- System prompts and goal oriented prompts
- Clean text-based UI (HTML/JS/CSS) designed for PC and IPad usage
- Configurable model launch parameters via `model_settings.json`

## Screenshots

<img src="https://i.imgur.com/Ia8TUkX.png">

<img src="https://i.imgur.com/znjL00G.png">

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
2. Launch the server with "RunMythForge.bat"
python -m uvicorn mythforge.main:app --host 0.0.0.0 --port 8000


3. Connect with `http:YourLocalIP:8000/MythForgeUI.html` in your browser to access the UI.

## API Endpoints

MythForge exposes a JSON API for programmatic access. Examples:

- `POST /chat/send` – Send a message to the assistant
- `GET /settings/` – Retrieve current model settings
- `PUT /settings/` – Update generation parameters
- `GET /prompts/` – List all global prompts

## Customization

Model parameters can be changed by editing `model_settings.json`. Global prompts can be added under `global_prompts/` and will be available in the UI. Logs are stored under `server_logs/`.
