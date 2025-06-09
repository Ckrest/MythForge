# Repository Guidelines for Codex Agents

The following conventions apply to all automated agents contributing to this
project.

## Coding Style

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

