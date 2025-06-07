# MythForge

MythForge is a FastAPI based chat server that uses `llama-cpp-python` to run a
local language model.  A minimal web UI is served from the `static/` directory.

## Setup

1. Install the Python dependencies:

   ```bash
   pip install -r requirements/requirements.txt
   ```

2. Provide a `.gguf` model.  Place the file anywhere under the `models/`
   directory.  The server automatically discovers the first `.gguf` file found in
   that folder tree.

## Running

Start the API using `uvicorn` directly or one of the provided helper scripts:

```bash
# Linux / macOS
./RunMythForge.sh

# Windows
RunMythForge.bat
```

Both scripts launch the server on `http://0.0.0.0:8000` with live reload
enabled.  You can also run the command manually:

```bash
uvicorn MythForgeServer:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/MythForgeUI.html` in your browser to use the web
UI.

