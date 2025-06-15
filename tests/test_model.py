import types

from mythforge.call_core import CallData, handle_chat
from mythforge.memory import MEMORY_MANAGER
from mythforge.invoker import LLMInvoker
from mythforge.logger import LoggerManager
from mythforge import model


def test_handle_chat_invokes_llm(monkeypatch):
    call = CallData(chat_id="c1", message="hi", options={"stream": False})
    count = {"n": 0}

    def fake_invoke(self, prompt, opts=None):
        count["n"] += 1
        return {"text": "ok"}

    monkeypatch.setattr(LLMInvoker, "invoke", fake_invoke)
    monkeypatch.setattr(LoggerManager, "log", lambda self, t, p: None)

    resp = handle_chat(call)
    assert resp["detail"] == "ok"
    assert count["n"] == 1


def test_llm_invoker_invoke(monkeypatch):
    def fake_call(system, prompt, **opts):
        return {"text": "done"}

    monkeypatch.setattr(model, "call_llm", fake_call)
    inv = LLMInvoker()
    assert inv.invoke("prompt") == {"text": "done"}


def test_logger_log_error(monkeypatch, tmp_path):
    mgr = LoggerManager(memory_manager=MEMORY_MANAGER)
    records = []
    monkeypatch.setattr(mgr, "log", lambda et, pl: records.append((et, pl)))
    mgr.log_error(Exception("boom"))
    assert records == [("errors", {"error": "boom"})]
