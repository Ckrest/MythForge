import json
import os
import server_log


def test_log_event_includes_function_name(tmp_path, monkeypatch):
    log_path = tmp_path / "log.json"
    monkeypatch.setattr(server_log, "_log_file", str(log_path))
    server_log._log_data.clear()

    def sample():
        server_log.log_event("test_tag", {"a": 1})

    sample()

    data = [json.loads(e) for e in log_path.read_text().split("\n\n") if e.strip()]
    entry = data[0]
    assert entry["function_name"] == "sample"
    assert entry["tag"] == "test_tag"


def test_log_function_includes_function_name(tmp_path, monkeypatch):
    log_path = tmp_path / "log_func.json"
    monkeypatch.setattr(server_log, "_log_file", str(log_path))
    server_log._log_data.clear()

    @server_log.log_function("decorated")
    def foo(x):
        return x * 2

    foo(3)

    data = [json.loads(e) for e in log_path.read_text().split("\n\n") if e.strip()]
    entry = data[0]
    assert entry["function_name"] == "foo"
    assert entry["tag"] == "decorated"
