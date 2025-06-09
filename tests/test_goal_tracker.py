import sys
import os
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import goal_tracker
from goal_tracker import (
    GoalsListModel,
    format_goal_eval_response,
    CHATS_DIR,
    init_state_from_prompt,
    check_and_generate_goals,
    evaluate_and_update_goals,
)

def test_duplicate_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "dup"
    state = {
        "completed_goals": [{"id": "1", "description": "done", "method": "", "status": "completed"}],
        "goals": [{"id": "2", "description": "a", "method": ""}],
        "messages_since_goal_eval": 5,
        "character_profile": "c",
        "scene_context": "s",
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(tmp_path / chat_id / "trimmed.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    def fake_call(_prompt, max_tokens=300):
        return {
            "choices": [
                {
                    "text": json.dumps(
                        {
                            "goals": [
                                {"id": "1", "description": "dup", "method": "", "status": "completed"},
                                {"id": "2", "description": "dup active", "method": "", "status": "in_progress"},
                                {"id": "3", "description": "new", "method": "", "status": "in_progress"},
                            ]
                        }
                    )
                }
            ]
        }

    monkeypatch.setattr("goal_tracker.generate_goals", lambda *a, **k: [])
    evaluate_and_update_goals(fake_call, chat_id, min_active=2)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    ids = {g["id"] for g in new_state["goals"]}
    assert "1" not in ids
    assert "2" in ids
    assert "3" in ids
    assert new_state["messages_since_goal_eval"] == 0


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


def test_init_state_from_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "init"
    goal_tracker.init_state_from_prompt(chat_id, "profile", "scene")
    path = tmp_path / chat_id / "state.json"
    data = json.loads(path.read_text())
    assert data["character_profile"] == "profile"
    assert data["scene_context"] == "scene"


def test_check_and_generate_goals_resets_counter(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "gen"
    state = {
        "character_profile": "p",
        "scene_context": "s",
        "goals": [],
        "messages_since_goal_eval": 5,
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)

    def fake_call(_prompt, max_tokens=200):
        return {"choices": [{"text": '[{"id":"1","description":"d","method":""}]'}]}

    goal_tracker.check_and_generate_goals(fake_call, chat_id)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    assert new_state["goals"][0]["id"] == "1"
    assert new_state["messages_since_goal_eval"] == 0


def test_check_and_generate_goals_retry_success(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "retry_success"
    state = {
        "character_profile": "p",
        "scene_context": "s",
        "goals": [],
        "messages_since_goal_eval": 5,
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)

    calls = {"n": 0}

    def fake_call(_prompt, max_tokens=200):
        calls["n"] += 1
        if calls["n"] < 2:
            return {"choices": [{"text": "[]"}]}
        return {"choices": [{"text": '[{"id":"1","description":"d","method":""}]'}]}

    goal_tracker.check_and_generate_goals(fake_call, chat_id)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    assert new_state["goals"][0]["id"] == "1"
    assert new_state["messages_since_goal_eval"] == 0
    assert calls["n"] == 2


def test_check_and_generate_goals_retry_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "retry_fail"
    state = {
        "character_profile": "p",
        "scene_context": "s",
        "goals": [],
        "messages_since_goal_eval": 5,
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)

    calls = {"n": 0}

    def fake_call(_prompt, max_tokens=200):
        calls["n"] += 1
        return {"choices": [{"text": "[]"}]}

    goal_tracker.check_and_generate_goals(fake_call, chat_id)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    assert new_state["goals"] == []
    assert new_state["messages_since_goal_eval"] == 5
    assert calls["n"] == 2


def test_evaluate_and_update_goals_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "backoff"
    state = {
        "completed_goals": [],
        "goals": [{"id": "1", "description": "g", "method": "", "status": "in_progress"}],
        "messages_since_goal_eval": 4,
        "character_profile": "c",
        "scene_context": "s",
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(tmp_path / chat_id / "trimmed.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    calls = {"n": 0}
    def fake_call(_prompt, max_tokens=300):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"choices": [{"text": "not json"}]}
        return {"choices": [{"text": json.dumps({"goals": [state["goals"][0]]})}]}

    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda t: sleeps.append(t))
    monkeypatch.setattr("goal_tracker.generate_goals", lambda *a, **k: [])

    evaluate_and_update_goals(fake_call, chat_id, max_retries=3)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    assert sleeps == [1.0, 2.0]
    assert new_state["messages_since_goal_eval"] == 0
    assert calls["n"] == 3


def test_evaluate_and_update_goals_regenerates(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "regen"
    state = {
        "completed_goals": [],
        "goals": [
            {"id": "1", "description": "a", "method": "", "status": "in_progress"},
            {"id": "2", "description": "b", "method": "", "status": "in_progress"},
        ],
        "messages_since_goal_eval": 4,
        "character_profile": "c",
        "scene_context": "s",
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(tmp_path / chat_id / "trimmed.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    def fake_call(_prompt, max_tokens=300):
        return {
            "choices": [{
                "text": json.dumps({"goals": [
                    {"id": "1", "description": "a", "method": "", "status": "completed"},
                    {"id": "2", "description": "b", "method": "", "status": "abandoned"}
                ]})
            }]
        }

    monkeypatch.setattr(
        "goal_tracker.generate_goals",
        lambda *a, **k: [
            {"id": "3", "description": "c", "method": ""},
            {"id": "4", "description": "d", "method": ""},
        ],
    )

    evaluate_and_update_goals(fake_call, chat_id, min_active=2)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    ids = {g["id"] for g in new_state["goals"]}
    assert ids == {"3", "4"}
    assert len(new_state["completed_goals"]) == 2
    assert new_state["messages_since_goal_eval"] == 0


def test_case_insensitive_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "dedup"
    state = {
        "completed_goals": [],
        "goals": [{"id": "1", "description": "Goal", "method": "", "status": "in_progress"}],
        "messages_since_goal_eval": 4,
        "character_profile": "c",
        "scene_context": "s",
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(tmp_path / chat_id / "trimmed.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    def fake_call(_prompt, max_tokens=300):
        return {
            "choices": [{
                "text": json.dumps({"goals": [
                    {"id": "1", "description": "goal", "method": "", "status": "in_progress"}
                ]})
            }]
        }

    monkeypatch.setattr(
        "goal_tracker.generate_goals",
        lambda *a, **k: [
            {"id": "2", "description": "GOAL", "method": ""},
            {"id": "3", "description": "unique", "method": ""},
        ],
    )

    evaluate_and_update_goals(fake_call, chat_id, min_active=2)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    descs = {g["description"].lower() for g in new_state["goals"]}
    assert descs == {"goal", "unique"}


def test_preserve_goal_details_on_completion(tmp_path, monkeypatch):
    monkeypatch.setattr("goal_tracker.CHATS_DIR", tmp_path)
    chat_id = "preserve"
    state = {
        "completed_goals": [],
        "goals": [
            {"id": "1", "description": "orig", "method": "m", "status": "in_progress"}
        ],
        "messages_since_goal_eval": 4,
        "character_profile": "c",
        "scene_context": "s",
    }
    os.makedirs(tmp_path / chat_id, exist_ok=True)
    with open(tmp_path / chat_id / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(tmp_path / chat_id / "trimmed.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    def fake_call(_prompt, max_tokens=300):
        return {
            "choices": [{
                "text": json.dumps({"goals": [
                    {"id": "1", "description": "changed", "method": "x", "status": "completed"}
                ]})
            }]
        }

    monkeypatch.setattr("goal_tracker.generate_goals", lambda *a, **k: [])

    evaluate_and_update_goals(fake_call, chat_id, min_active=1)
    new_state = json.loads((tmp_path / chat_id / "state.json").read_text())
    assert new_state["goals"] == []
    assert new_state["completed_goals"] == [{"id": "1", "description": "orig", "method": "m", "status": "completed"}]

