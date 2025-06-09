import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a minimal stub for llama_cpp to avoid loading a real model
sys.modules.setdefault("llama_cpp", types.SimpleNamespace(Llama=lambda **k: None))

# Ensure a dummy model file exists so MythForgeServer's model discovery succeeds
os.makedirs("models", exist_ok=True)
with open("models/dummy.gguf", "w", encoding="utf-8") as f:
    f.write("dummy")

from scripts import MythForgeServer

class Dummy:
    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, prompt, **kwargs):
        if kwargs.get("stream"):
            return ({"choices": [{"text": o}]} for o in self.outputs)
        return {"choices": [{"text": self.outputs[0]}]}

def test_call_llm_non_stream(monkeypatch):
    monkeypatch.setattr(MythForgeServer, "llm", Dummy(["hello"]))
    res = MythForgeServer.call_llm("hi")
    assert isinstance(res, dict)
    assert res["choices"][0]["text"] == "hello"

def test_call_llm_stream(monkeypatch):
    monkeypatch.setattr(MythForgeServer, "llm", Dummy(["a", "b"]))
    gen = MythForgeServer.call_llm("hi", stream=True)
    assert list(gen) == [
        {"choices": [{"text": "a"}]},
        {"choices": [{"text": "b"}]},
    ]

