import json
import sys
import types

from mythforge.utils import load_json
from mythforge.main import ChatRequest
from mythforge.call_core import build_call, parse_response, stream_parsed
from mythforge.call_templates import (
    standard_chat,
    goal_generation,
    logic_check,
)
from mythforge import memory


def test_load_json_basic(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("[1, 2, 3]")
    assert load_json(str(path)) == [1, 2, 3]


def test_load_json_missing(tmp_path):
    path = tmp_path / "missing.json"
    assert load_json(str(path)) == []


def test_build_call(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr("mythforge.utils.GLOBAL_PROMPTS_DIR", str(prompt_dir))
    data = {"name": "Example", "content": "Hello"}
    (prompt_dir / "Example.json").write_text(json.dumps(data))
    req = ChatRequest(chat_id="1", message="hi")
    call = build_call(req)
    assert call.call_type == "standard_chat"
    assert call.global_prompt == ""

    system_text, user_text = standard_chat.prepare(call)
    assert call.global_prompt == "Hello"


def test_memory_global_prompt(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr("mythforge.utils.GLOBAL_PROMPTS_DIR", str(prompt_dir))
    data = {"name": "Example", "content": "Hello"}
    (prompt_dir / "Example.json").write_text(json.dumps(data))
    memory.initialize()
    memory.set_global_prompt("Custom")
    req = ChatRequest(chat_id="1", message="hi")
    call = build_call(req)
    system_text, _ = standard_chat.prepare(call)
    assert call.global_prompt == "Custom"


def test_goal_generation_uses_memory_global_prompt(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr("mythforge.utils.GLOBAL_PROMPTS_DIR", str(prompt_dir))
    memory.initialize()
    memory.set_global_prompt("Stored")

    req = ChatRequest(chat_id="1", message="hi")
    call = build_call(req)
    system_text, _ = goal_generation.prepare(call)
    assert call.global_prompt == "Stored"


def test_parse_response_extract_text():
    assert parse_response({"text": "hello"}) == "hello"


def test_stream_parsed_extract_text():
    chunks = [{"text": "a"}, {"text": "b"}]
    assert list(stream_parsed(chunks)) == ["a", "b"]


def test_format_for_model_single_line():
    from mythforge.call_core import format_for_model

    result = format_for_model("sys", "user")
    assert "\n" not in result
    assert result == (
        '--prompt "<|im_start|>sys<|im_end|><|im_start|>user user<|im_end|>'
        '<|im_start|>assistant"'
    )


def test_append_message_skips_blank(tmp_path, monkeypatch):
    monkeypatch.setattr("mythforge.utils.CHATS_DIR", str(tmp_path))
    svc = memory.ChatHistoryService()
    svc.append_message("c1", "user", "   ")
    assert svc.load_history("c1") == []
    svc.append_message("c1", "user", "hello")
    assert svc.load_history("c1") == [{"role": "user", "content": "hello"}]


def test_logic_check_invocation(monkeypatch):
    calls = {}

    def fake_select(background: bool = False):
        calls["background"] = background
        return "model.gguf"

    class DummyLLM:
        def __init__(self, model_path: str, n_ctx: int) -> None:
            calls["model_path"] = model_path
            calls["n_ctx"] = n_ctx

        def __call__(self, prompt: str, max_tokens: int):
            calls["prompt"] = prompt
            calls["max_tokens"] = max_tokens
            return {"choices": [{"text": "ok"}]}

    monkeypatch.setattr(
        "mythforge.call_templates.logic_check._select_model_path", fake_select
    )
    module = types.SimpleNamespace(Llama=DummyLLM)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setattr(
        "mythforge.call_templates.logic_check.format_for_model",
        lambda s, u: f"{s}|{u}",
    )

    result = logic_check.send_prompt("sys", "user")
    assert result == {"text": "ok"}
    assert calls == {
        "background": True,
        "model_path": "model.gguf",
        "n_ctx": 4096,
        "prompt": "sys|user",
        "max_tokens": 256,
    }
