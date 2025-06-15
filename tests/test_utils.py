import json
import types
import sys

from mythforge.main import ChatRequest
from mythforge.call_core import CallData
from mythforge.call_templates import standard_chat, goal_generation
from mythforge.prompt_preparer import PromptPreparer
from mythforge.response_parser import ResponseParser
from mythforge import memory


def test_memory_history_basic(tmp_path):
    mgr = memory.MemoryManager(root_dir=str(tmp_path))
    mgr.save_history("c1", [])
    hist = mgr.load_history("c1")
    hist.append({"role": "user", "content": "hello"})
    mgr.save_history("c1", hist)
    assert mgr.load_history("c1") == [{"role": "user", "content": "hello"}]


def test_prepare_call(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr(memory.MEMORY_MANAGER, "prompts_dir", str(prompt_dir))
    data = {"name": "Example", "content": "Hello"}
    (prompt_dir / "Example.json").write_text(json.dumps(data))
    req = ChatRequest(chat_id="1", message="hi")
    call = CallData(chat_id=req.chat_id, message=req.message)
    system_text, _ = standard_chat.prepare(call)
    assert call.global_prompt == "Hello"


def test_global_prompt_usage(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr(memory.MEMORY_MANAGER, "prompts_dir", str(prompt_dir))
    data = {"name": "Example", "content": "Hello"}
    (prompt_dir / "Example.json").write_text(json.dumps(data))
    memory.initialize()
    memory.set_global_prompt("Custom")
    req = ChatRequest(chat_id="1", message="hi")
    call = CallData(chat_id=req.chat_id, message=req.message)
    system_text, _ = goal_generation.prepare(call)
    assert call.global_prompt == "Custom"


def test_response_parser_extract_text():
    parser = ResponseParser().load({"text": "hello"})
    assert parser.parse() == "hello"


def test_response_parser_stream():
    chunks = [{"text": "a"}, {"text": "b"}]
    parser = ResponseParser().load(chunks)
    assert list(parser.parse()) == ["a", "b"]


def test_prompt_preparer_format():
    prep = PromptPreparer().prepare("sys", "user")
    assert "\n" not in prep
    assert prep == (
        '--prompt "<|im_start|>sys<|im_end|><|im_start|>user user<|im_end|>'
        '<|im_start|>assistant"'
    )
