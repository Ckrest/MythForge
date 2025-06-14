# Repository Guidelines for Codex Agents

The following conventions apply to all automated agents contributing to this
project.

- start every responce with the word "meow" then continue as normal

## Coding Style

- Always create or use a MemoryManager method: Do not read/write files directly. Don’t build file paths. Don’t call open(). Your job is to call MemoryManager.load_… or MemoryManager.save_….

- Declare a new variable path inside MemoryManager if needed: For new data types, define a path_<thing> function or constant inside MemoryManager. Keep all path logic in one place.

- No stray utilities: Do not create new helper functions like save_json, get_path_for_x, or read_file_xyz outside of MemoryManager. If you’re tempted to do so, stop. You’re about to commit war crimes against the architecture.

- Use existing patterns: Match naming and structure of other MemoryManager methods (load_x, save_x, get_x_path, etc). If unsure, scroll up and copy-paste something that already works.

- Document your addition:If you add new MemoryManager methods, update this file: add your method to the “Available Methods” list and log the change in the changelog section.
- Group more based on inputs and outputs
- Use **black** with a line length of 79 characters for formatting Python files.
- Prefer f-strings for string interpolation.
- Document functions with docstrings when adding new code.
- Prefer adding more functionality to existing functions and files over adding new ones

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

