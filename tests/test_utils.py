import json

from mythforge.utils import load_json
from mythforge.main import ChatRequest
from mythforge.call_core import build_call
from mythforge.call_templates import (
    standard_chat,
    helper,
    default as default_template,
    goal_generation,
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

    system_text, user_text = standard_chat.prepare(call, [])
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
    system_text, _ = standard_chat.prepare(call, [])
    assert call.global_prompt == "Custom"


def test_other_templates_use_memory_global_prompt(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr("mythforge.utils.GLOBAL_PROMPTS_DIR", str(prompt_dir))
    memory.initialize()
    memory.set_global_prompt("Stored")

    req = ChatRequest(chat_id="1", message="hi")
    call = build_call(req)
    system_text, _ = helper.prepare(call, [])
    assert call.global_prompt == "Stored"

    call = build_call(req)
    system_text, _ = default_template.prepare(call, [])
    assert call.global_prompt == "Stored"

    call = build_call(req)
    system_text, _ = goal_generation.prepare(call, [])
    assert call.global_prompt == "Stored"
