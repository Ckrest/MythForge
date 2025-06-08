import os
import json
import datetime
import atexit
import functools
import inspect

LOG_DIR = "server_logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_data = []
_log_file = os.path.join(LOG_DIR, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def _flush():
    with open(_log_file, 'w', encoding='utf-8') as f:
        json.dump(_log_data, f, indent=2, ensure_ascii=False)

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
        "time": datetime.datetime.now().isoformat(),
        "tag": tag,
        "function": _get_function_details(func),
        "args": repr(args),
        "kwargs": repr(kwargs),
        "result": repr(result),
    }
    _log_data.append(entry)
    # Immediately persist logs so the file exists even if the process
    # is long-running.  This ensures that a log file is created as soon
    # as the first entry is recorded instead of only on shutdown.
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
