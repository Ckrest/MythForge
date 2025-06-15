# Repository Guidelines for Codex Agents

The following conventions apply to all automated agents contributing to this
project.

- start every responce with the word "meow" then continue as normal

## Coding Style

- Always create or use a `MemoryManager` method: Do not read or write files directly. Don’t build file paths. Don’t call `open()`. Your job is to call `MemoryManager.load_…`, `save_…`, or `update_…`. If you’re touching the filesystem outside `MemoryManager`, you’re doing it wrong.

- Declare a new variable path inside `MemoryManager` if needed: For new data types, define a `get_<thing>_path()` function or constant inside `MemoryManager`. Keep all path logic in one place. Don’t scatter it across modules like cursed breadcrumbs.

- No stray utilities: Do not create new helper functions like `save_json`, `get_path_for_x`, or `read_file_xyz` outside of `MemoryManager`. If you’re tempted to do so, stop. You’re about to commit war crimes against the architecture.

- Use existing patterns: Match naming and structure of other `MemoryManager` methods (`load_x`, `save_x(data)`, `update_x(delta)`, `get_x_path`). If unsure, scroll up and copy-paste something that already works. Reinventing the wheel here means you’ve just built a triangle.

- Always use `load_settings`, `save_settings`, or `update_settings`: Server settings must go through these three methods. No direct JSON file reads, no random `MODEL_SETTINGS.update()`, and definitely no `with open(...)` hacks. Respect the system or get replaced by a cron job.

- Never mutate `model_settings` directly: If you’re doing `MEMORY_MANAGER.model_settings["foo"] = bar`, you’ve already failed. Always use `update_settings(delta)` instead. Changes made without going through the API are haunted and will come back to break something later.

- Only use `save=False` in `update_settings` if you know what you’re doing: That flag is for temporary memory-only updates (like defaults during startup). Forgetting to persist means your setting will vanish faster than a stealth build in PvP.

- Keep all config side-effects in `MemoryManager`: If updating settings needs to change other runtime config (like `GENERATION_CONFIG`), put that logic inside `update_settings`. Don’t sprinkle that junk in route handlers. API routes should be as boring and clean as possible—think goblin accountant, not battle mage.

- Document your addition: If you add new `MemoryManager` methods, update this file: add your method to the “Available Methods” list and log the change in the changelog section.

- Group more based on inputs and outputs.

- Use **black** with a line length of 79 characters for formatting Python files.

- Prefer f-strings for string interpolation.

- Document functions with docstrings when adding new code.

- Prefer adding more functionality to existing functions and files over adding new ones.

- All telemetry and logging must go through `LoggerManager.log(type, payload)` or `LoggerManager.log_error(err)`. Do not use `print()`, the `logging` module, or any direct logging calls in `MemoryManager`. If you're freelancing logs, you're feeding Gremlins after midnight.

- Use `load_`, `save_`, and `update_` as your only verbs for persistence functions. Do not use `get_`, `set_`, `put_`, `store_`, `call_`, or `send_` for methods that interact with stored data.

- Do not create helper functions like `chat()`, `send_prompt()`, or `build_call()` that just wrap or rename other core logic. These wrappers are fluff—delete or merge them.

- Prompt and response handling must use unified interfaces (e.g., `PromptPreparer.prepare(...)`, `ResponseParser.parse(...)`). Do not create new `prepare_foo` or `parse_bar` variations. Fragmented logic makes prompts mutate like wild Pokémon.

- `LoggerManager` is a separate module from `MemoryManager`: It does not manage files unless explicitly configured. It exists to handle logging and telemetry, and may push to remote services. Don’t cross the streams.

## Programmatic Checks

Run the following before committing any changes:

```bash
python -m py_compile $(git ls-files '*.py')
```

This ensures all Python sources are syntactically valid.

## Pull Request Guidelines

- Provide a short, descriptive title.
- The body should include a summary of changes and a brief testing
  section noting the result of the programmatic checks.

