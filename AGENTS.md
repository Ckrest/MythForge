# Repository Guidelines for Codex Agents

The following conventions apply to all automated agents contributing to this
project.

- Prefix Responses: Start every generated response with the word meow then continue as normal.

- Make chat functions and global_prompt functions mirror each other in thier methods and functions, and enforce that naming sceam with every function of variable that interacts with them

## Coding Style

- MemoryManager Enforcement: Always use MemoryManager.load_*, save_*, or update_* methods for any persistence. Do not read/write files, build paths, or call open() outside of MemoryManager.

- Path Declarations Centralized: For new data types, declare a get_<thing>_path() method or constant inside MemoryManager. All filesystem logic must live there, not in route handlers or utilities.

- Zero Stray Utilities: Do not create free‑standing helper functions like save_json, get_path_for_x, or read_file_xyz outside MemoryManager. If your change needs new persistence logic, extend MemoryManager instead.

- Pattern Consistency: Match existing MemoryManager naming: use load_x, save_x(data), update_x(delta), and get_x_path(). Copy‑paste from a working example if uncertain.

- Settings Workflow: Always use load_settings, save_settings, or update_settings for server settings. Never mutate model_settings directly or use with open(...) hacks.

- Temporary Flags: Use save=False in update_settings only for ephemeral defaults at startup. Permanent changes must omit that flag.

- Side‑Effect Isolation: Any runtime configuration side‑effects (e.g., updating GENERATION_CONFIG) must be inside update_settings, not in route handlers.

- Documentation & Changelog: When adding new MemoryManager methods, update the “Available Methods” list in the class file and record the change in the changelog section.

- Formatting & Style: Use black (line length ≤79). Prefer f‑strings. Document new functions with docstrings.

- Telemetry & Logging: Route all logs and errors through LoggerManager.log(type, payload) or LoggerManager.log_error(err). Do not use print() or the logging module directly in MemoryManager.

- Unified Prompt/Response Interfaces: Use PromptPreparer.prepare(...) and ResponseParser.parse(...). No new prepare_foo or parse_bar methods.

- Verb Discipline: For persistence, only use verbs load_, save_, and update_. Do not use get_, set_, put_, store_, call_, or send_ for data operations.

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

