import os
import json
import datetime
import uuid
import atexit
import functools
import inspect
from typing import Any, Dict

LOG_DIR = "server_logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_data = []
_session_id = os.environ.get("MF_SESSION_ID") or uuid.uuid4().hex
_log_file = os.path.join(LOG_DIR, f"{_session_id}.json")


def _flush():
    """Append buffered log entries to the log file in JSONL format."""
    if not _log_data:
        return
    with open(_log_file, 'a', encoding='utf-8') as f:
        for entry in _log_data:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            f.write("\n\n")
    _log_data.clear()

atexit.register(_flush)


def _get_function_details(fn):
    """Return a detailed description of the function for logging."""
    try:
        file = inspect.getsourcefile(fn) or ""
        line = inspect.getsourcelines(fn)[1]
    except (OSError, TypeError):
        file = ""
        line = None
    return {
        "name": fn.__name__,
        "qualified_name": fn.__qualname__,
        "module": fn.__module__,
        "file": file,
        "line": line,
        "signature": str(inspect.signature(fn)),
        "doc": inspect.getdoc(fn),
    }


def log_entry(tag: str, func, args, kwargs, result) -> None:
    entry = {
        "function_name": func.__name__,
        "time": datetime.datetime.now().isoformat(),
        "tag": tag,
        "function": _get_function_details(func),
        "args": repr(args),
        "kwargs": repr(kwargs),
        "result": repr(result),
    }
    _log_data.append(entry)
    # Persist immediately so the file exists even if the process runs long.
    _flush()


def log_event(tag: str, data: Dict[str, Any]) -> None:
    caller = inspect.currentframe().f_back
    func_name = caller.f_code.co_name if caller else "<unknown>"
    entry = {
        "function_name": func_name,
        "time": datetime.datetime.now().isoformat(),
        "tag": tag,
        **data,
    }
    _log_data.append(entry)
    _flush()


def log_function(tag: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                res = fn(*args, **kwargs)
                log_entry(tag, fn, args, kwargs, res)
                return res
            except Exception as e:
                log_entry(tag, fn, args, kwargs, f"ERROR: {e}")
                raise
        wrapper._patched = True
        return wrapper
    return decorator


def patch_module_functions(module, tag: str) -> None:
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if func.__module__ != module.__name__:
            continue
        if getattr(func, "_patched", False):
            continue
        setattr(module, name, log_function(tag)(func))


def get_log_path() -> str:
    return _log_file
