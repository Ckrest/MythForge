import json
import types
import sys
from mythforge.call_core import CallData
from mythforge.call_templates import standard_chat, goal_generation
from mythforge.prompt_preparer import PromptPreparer
from mythforge.response_parser import ResponseParser
from mythforge.invoker import LLMInvoker
from mythforge import memory


def test_memory_history_basic(tmp_path):
    mgr = memory.MemoryManager(root_dir=str(tmp_path))
    mgr.save_history("c1", [])
    hist = mgr.load_history("c1")
    hist.append({"role": "user", "content": "hello"})
    mgr.save_history("c1", hist)
    assert mgr.load_history("c1") == [{"role": "user", "content": "hello"}]


def test_prepare_call(monkeypatch):
    call = CallData(
        chat_name="1", message="hi", global_prompt="Hello", options={}
    )

    invoked = {}

    def fake_invoke(self, prompt, opts=None):
        invoked["prompt"] = prompt
        return {"text": "ok"}

    monkeypatch.setattr(LLMInvoker, "invoke", fake_invoke)

    result = standard_chat.prepare_and_chat(call)
    assert result == "ok"
    assert invoked["prompt"].startswith("--prompt")


def test_global_prompt_usage(monkeypatch):
    call = CallData(
        chat_name="1", message="hi", global_prompt="Custom", options={}
    )

    monkeypatch.setattr(
        LLMInvoker, "invoke", lambda self, prompt, opts=None: {"text": "ok"}
    )

    result = goal_generation.generate_goals(
        call.global_prompt, call.message, {}
    )
    assert result == "ok"


def test_response_parser_extract_text():
    parser = ResponseParser().load({"text": "hello"})
    assert parser.parse() == "hello"


def test_response_parser_stream():
    chunks = [{"text": "a"}, {"text": "b"}]
    parser = ResponseParser().load(chunks)
    assert list(parser.parse()) == ["a", "b"]


def test_response_parser_stream_dicts():
    chunks = [{"text": "hello"}, {"text": "world"}]
    parsed = ResponseParser().load(chunks).parse()
    assert list(parsed) == ["hello", "world"]


def test_prompt_preparer_format():
    prep = PromptPreparer().prepare("sys", "user")
    assert "\n" not in prep
    assert prep == (
        '--prompt "<|im_start|>sys<|im_end|><|im_start|>user user<|im_end|>'
        '<|im_start|>assistant"'
    )


def test_prompt_preparer_escaping():
    prep = PromptPreparer().prepare('s "s"', 'u "u" <x>')
    assert '\\"' in prep
    assert '<|im_start|>user u \\"u\\" <x><|im_end|>' in prep
