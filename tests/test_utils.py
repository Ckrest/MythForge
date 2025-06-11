import json

from mythforge.utils import load_json
from mythforge.main import ChatRequest, import_message_data


def test_load_json_basic(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("[1, 2, 3]")
    assert load_json(str(path)) == [1, 2, 3]


def test_load_json_missing(tmp_path):
    path = tmp_path / "missing.json"
    assert load_json(str(path)) == []


def test_import_message_data(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    monkeypatch.setattr("mythforge.main.GLOBAL_PROMPTS_DIR", str(prompt_dir))
    data = {"name": "Example", "content": "Hello"}
    (prompt_dir / "Example.json").write_text(json.dumps(data))
    req = ChatRequest(chat_id="1", message="hi", global_prompt="Example", call_type="")
    updated = import_message_data(req)
    assert updated.call_type == "user_message"
    assert updated.global_prompt == "Hello"
