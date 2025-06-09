import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a minimal stub for llama_cpp to avoid import errors when importing
# MythForgeServer in the test environment.
import types
sys.modules.setdefault("llama_cpp", types.SimpleNamespace(Llama=lambda **k: None))

# Ensure a dummy model file exists so MythForgeServer's model discovery succeeds
os.makedirs("models", exist_ok=True)
with open("models/dummy.gguf", "w", encoding="utf-8") as f:
    f.write("dummy")

from scripts.MythForgeServer import load_json
from scripts.goals import _load_json


def test_load_json_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("")
    assert load_json(str(path)) == []
    assert _load_json(str(path)) == []


def test_load_json_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert load_json(str(path)) == []
    assert _load_json(str(path)) == []

