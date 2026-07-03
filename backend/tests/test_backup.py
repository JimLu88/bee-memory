"""v5 复审 R3 回归 — 备份幂等 + restore 只取最新一代 (防二次备份混代致重建乱码)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, memory, ppr, semantic  # noqa: E402
from app.backup import coordinator  # noqa: E402


class _FakePool:
    """内存池: put 存 bytes, get 取回. 模拟远端分片存储."""
    def __init__(self):
        self.store = {}

    def put(self, shard_id, data):
        self.store[shard_id] = bytes(data)
        return {"remote_ref": shard_id, "account_id": "", "pending_upload": False}

    def get(self, ref):
        return self.store.get(ref)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "m.sqlite"
    for mod in (memory, associative, semantic, ppr, coordinator):
        monkeypatch.setattr(mod, "DB_PATH", p)
    monkeypatch.setattr(semantic, "embed_text", lambda t, timeout=None: None)  # 不连 Ollama
    fake = _FakePool()
    monkeypatch.setattr(coordinator.pl, "by_name", lambda name: fake)
    with memory._conn():
        pass
    return p


def _store(content, importance=5):
    from app.memory import StoreRequest, store
    return store(StoreRequest(kind="semantic", content=content, importance=importance))["memory_id"]


def test_backup_is_idempotent(db):
    mid = _store("备份幂等测试内容独有甲")
    coordinator.backup_memory(mid)
    coordinator.backup_memory(mid)   # 二次备份
    with coordinator._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM backup_shards WHERE memory_id=?", (mid,)).fetchone()[0]
    assert n == 5, "二次备份应替换旧代而非累积成 10 片"


def test_restore_roundtrip_after_rebackup(db):
    mid = _store("重建往返测试内容独有乙")
    coordinator.backup_memory(mid)
    coordinator.backup_memory(mid)   # 二次后仍应能正确恢复
    r = coordinator.restore_memory(mid)
    assert r and r["status"] == "ok"
    assert r["content"] == "重建往返测试内容独有乙"


def test_restore_none_when_absent(db):
    assert coordinator.restore_memory("m-nonexistent") is None
