# MythForge

MythForge is a lightweight chat server that runs on top of `llama_cpp`.
It exposes a simple REST API and includes a small web UI.

## Settings

Configuration is loaded from `settings.json` in the project root.  Most
options mirror the parameters exposed by `llama_cpp`.  Two additional
values control when old messages are summarized:

- `summarize_threshold` – number of raw messages stored before the
  server begins summarizing.
- `summarize_batch` – how many messages to summarize at a time.

Adjust these numbers to tweak context trimming behaviour.  The default
values are suitable for most small models.
