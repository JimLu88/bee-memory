"""v4 P3 测试 — 睡眠循环 (stability/MOC/vault渲染/遗忘报告). 嵌入 mock, vault 用 tmp."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, memory, ppr, semantic, sleep_cycle  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "memories.sqlite"
    for mod in (memory, associative, semantic, ppr):
        monkeypatch.setattr(mod, "DB_PATH", p)
    monkeypatch.setattr(semantic, "embed_text", lambda t, timeout=None: None)   # 不连 Ollama
    monkeypatch.setattr(semantic, "embed_batch", lambda ts: [None for _ in ts])
    semantic._QCACHE.clear()
    monkeypatch.setattr(sleep_cycle, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr(sleep_cycle, "MOC_MIN_DEGREE", 1)  # 小数据集也能出 MOC
    semantic.invalidate_cache(); ppr.invalidate_cache()
    with memory._conn():
        pass
    return p


def _store(kind, content, importance=2):
    from app.memory import StoreRequest, store
    return store(StoreRequest(kind=kind, content=content, importance=importance))["memory_id"]


def test_sleep_cycle_full_run(db, tmp_path):
    _store("episodic", "利润口径独有甲 与 备货策略独有乙 的关系讨论")
    _store("episodic", "利润口径独有甲 深入分析 与 备货策略独有乙")
    _store("procedural", "关键流程绝不能忘", importance=5)
    res = sleep_cycle.run_sleep_cycle(do_forget=False, render_vault=True)
    assert res["status"] == "ok"
    assert res["consolidate"]["status"] == "ok"
    assert res["stability_updated"] >= 3
    assert res["mocs"] >= 1, "应生成至少一个概念地图 MOC"
    assert res["vault"]["concept_notes"] >= 1
    assert (tmp_path / "vault" / "README.md").exists()
    assert list((tmp_path / "vault" / "concepts").glob("*.md")), "应有概念笔记"
    assert res["forget"]["dry_run"] is True


def test_stability_filled(db):
    mid = _store("procedural", "重要SOP", importance=5)
    with memory._conn() as c:
        sleep_cycle._update_stability(c)
        s, d = c.execute("SELECT stability, difficulty FROM memories WHERE id=?", (mid,)).fetchone()
    assert s and s > 30, "高重要度 → 高存储强度"
    assert d == 0.0, "importance=5 → difficulty=0"


def test_moc_is_recallable(db):
    _store("episodic", "海马体索引 和 扩散激活 甲")
    _store("episodic", "海马体索引 和 扩散激活 乙")
    with memory._conn() as c:
        associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as c:
        n = sleep_cycle._generate_mocs(c)
        mocs = c.execute("SELECT COUNT(*) FROM memories WHERE kind='moc'").fetchone()[0]
    assert n >= 1 and mocs >= 1


def test_moc_idempotent(db):
    _store("episodic", "利润口径独有甲 与 备货策略独有乙 A")
    _store("episodic", "利润口径独有甲 与 备货策略独有乙 B")
    with memory._conn() as c:
        associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as c:
        sleep_cycle._generate_mocs(c)
        n1 = c.execute("SELECT COUNT(*) FROM memories WHERE kind='moc'").fetchone()[0]
    with memory._conn() as c:
        sleep_cycle._generate_mocs(c)
        n2 = c.execute("SELECT COUNT(*) FROM memories WHERE kind='moc'").fetchone()[0]
    assert n1 == n2, "重复跑不应重复建 MOC (幂等 upsert)"
