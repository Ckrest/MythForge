from mythforge.memory import MemoryManager


def test_load_save_history_called(tmp_path, monkeypatch):
    mgr = MemoryManager(root_dir=str(tmp_path))
    calls = {"load": 0, "save": 0}

    def fake_load(cid):
        calls["load"] += 1
        return []

    def fake_save(cid, hist):
        calls["save"] += 1

    monkeypatch.setattr(mgr, "load_history", fake_load)
    monkeypatch.setattr(mgr, "save_history", fake_save)

    mgr.save_history("c1", [])
    mgr.load_history("c1")

    assert calls == {"load": 1, "save": 1}
