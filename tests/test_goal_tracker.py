import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from goal_tracker import (
    parse_and_merge_goals,
    GoalsListModel,
    format_goal_eval_response,
    CHATS_DIR,
)

def test_duplicate_ids():
    state = {
        "completed_goals": [{"id": "1", "description": "done", "method": "", "status": "completed"}],
        "goals": [{"id": "2", "description": "a", "method": ""}],
    }
    model = GoalsListModel.parse_obj({
        "goals": [
            {"id": "1", "description": "dup", "method": "", "status": "completed"},
            {"id": "2", "description": "dup active", "method": ""},
            {"id": "3", "description": "new", "method": ""},
        ]
    })
    parse_and_merge_goals(model, state, 3, lambda *a, **k: [])
    ids = {g["id"] for g in state["goals"]}
    assert "1" not in ids
    assert "2" in ids
    assert "3" in ids


def test_format_goal_eval_response_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "abc"
    result = format_goal_eval_response("not json", chat_id)
    assert result is None
    error_path = tmp_path / chat_id / "goal_eval_error.txt"
    assert error_path.exists()
    assert error_path.read_text() == "not json"


def test_format_goal_eval_response_valid(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "good"
    text = '{"goals": [{"id": "1", "description": "d", "method": "", "status": "completed"}]}'
    result = format_goal_eval_response(text, chat_id)
    assert result is not None
    assert not (tmp_path / chat_id / "goal_eval_error.txt").exists()

