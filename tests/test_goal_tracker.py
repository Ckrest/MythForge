import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from goal_tracker import parse_and_merge_goals, GoalsListModel

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

