"""v4 P1 测试 — 语义向量 / PPR / 混合检索. 嵌入用确定性假向量 (不依赖 Ollama)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, memory, ppr, semantic  # noqa: E402


def _fake_vec(text: str, dim: int = 1024) -> list[float]:
    """确定性假嵌入: 文本 hash 播种归一化. 含相同关键词的文本更接近 (叠加关键词方向)."""
    h = int(hashlib.md5((text or "").encode("utf-8")).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    for kw in ("利润", "备货", "海马", "风险"):
        if kw in (text or ""):
            kwrng = np.random.default_rng(int(hashlib.md5(kw.encode()).hexdigest(), 16) % (2**32))
            v += 3.0 * kwrng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) or 1.0)
    return v.tolist()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "memories.sqlite"
    for mod in (memory, associative, semantic, ppr):
        monkeypatch.setattr(mod, "DB_PATH", p)
    monkeypatch.setattr(semantic, "embed_text", lambda t, timeout=None: _fake_vec(t))
    monkeypatch.setattr(semantic, "embed_batch", lambda ts: [_fake_vec(t) for t in ts])
    semantic.invalidate_cache(); ppr.invalidate_cache(); semantic._QCACHE.clear()
    with memory._conn():
        pass
    return p


def _store(kind, content, importance=2):
    from app.memory import StoreRequest, store
    return store(StoreRequest(kind=kind, content=content, importance=importance))["memory_id"]


def test_token_count_column_added(db):
    with memory._conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(memories)")}
    assert {"token_count", "stability", "difficulty"} <= cols


def test_store_sets_token_count_and_vector(db):
    mid = _store("episodic", "畔色木作利润口径核算的详细流程说明文档内容")
    with memory._conn() as c:
        tc = c.execute("SELECT token_count FROM memories WHERE id=?", (mid,)).fetchone()[0]
        vec = c.execute("SELECT COUNT(*) FROM memories_vec WHERE memory_id=?", (mid,)).fetchone()[0]
    assert tc and tc > 0
    assert vec == 1, "写入应即嵌入"


def test_vector_search_ranks_semantically(db):
    a = _store("episodic", "利润口径与成本核算")
    b = _store("episodic", "利润分配和口径调整")
    _store("episodic", "天气晴朗适合散步")
    with memory._conn() as conn:
        hits = semantic.vector_search(conn, "利润口径", k=3)
    ids = [h[0] for h in hits]
    assert a in ids and b in ids
    assert set(ids[:2]) == {a, b}


def test_ppr_spreads_over_edges(db):
    a = _store("episodic", "海马体索引 与 扩散激活 记忆核心")
    b = _store("episodic", "再谈 海马体索引 和 扩散激活 实现")
    associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as conn:
        scores = ppr.personalized_pagerank(conn, [a])
    assert b in scores and scores[b] > 0, "PPR 应从 A 扩散到共享概念的 B"


def test_connect_ppr_finds_bridge(db):
    _store("episodic", "利润口径独有甲 的核算原则")
    _store("episodic", "利润口径独有甲 和 备货策略独有乙 二者关联")
    _store("episodic", "备货策略独有乙 的分层方法")
    associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as conn:
        a_ids = associative.entry_search(conn, "利润口径独有甲", k=5)
        b_ids = associative.entry_search(conn, "备货策略独有乙", k=5)
        res = ppr.connect_ppr(conn, a_ids, b_ids)
    assert res["connected"] is True and res["connectors"]


def test_hybrid_recall_compact(db):
    _store("episodic", "利润口径 与 备货策略 的关系甲")
    _store("episodic", "利润口径 深入讨论 乙")
    associative.reindex_concepts(rebuild_edges=True)
    res = associative.hybrid_recall("利润口径", k=5, compact=True)
    assert res["items"], "混合检索应有结果"
    it = res["items"][0]
    assert {"id", "title", "snippet", "score", "via"} <= set(it.keys())
    assert "content" not in it, "紧凑模式不返回全文 (省 token)"


def test_hybrid_recall_full_mode(db):
    _store("episodic", "风险管理 与 尾部风险 的关系")
    associative.reindex_concepts(rebuild_edges=True)
    res = associative.hybrid_recall("风险", k=3, compact=False)
    assert res["items"] and "content" in res["items"][0]


def test_hybrid_recall_persona_filter(db):
    from app.memory import StoreRequest, store
    m1 = store(StoreRequest(kind="knowledge_book", content="利润口径 A", meta={"persona_id": "p1"}))["memory_id"]
    store(StoreRequest(kind="knowledge_book", content="利润口径 B", meta={"persona_id": "p2"}))
    associative.reindex_concepts(rebuild_edges=True)
    res = associative.hybrid_recall("利润口径", k=5, persona_id="p1")
    ids = [it["id"] for it in res["items"]]
    assert m1 in ids


def test_get_by_id_exact(db):
    mid = _store("episodic", "精确取全文的测试内容独有词XYZ")
    from app.memory import get_by_id
    row = get_by_id(mid)
    assert row["id"] == mid and "独有词XYZ" in row["content"]


def test_auto_enroll_high_importance(db):
    mid = _store("procedural", "重要SOP必须复习", importance=5)
    with memory._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM review_state WHERE memory_id=?", (mid,)).fetchone()[0]
    assert n == 1, "importance>=4 应自动入复习闸"
    low = _store("episodic", "普通碎片", importance=2)
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM review_state WHERE memory_id=?", (low,)).fetchone()[0] == 0


def test_provenance_source_in_recall(db):
    from app.memory import StoreRequest, store
    store(StoreRequest(kind="knowledge_book", content="复式记账原理独有甲", meta={"title": "会计学原理", "author": "张三"}))
    associative.reindex_concepts(rebuild_edges=True)
    res = associative.hybrid_recall("复式记账原理独有甲", k=3)
    assert res["items"] and res["items"][0].get("source") == "会计学原理"


def test_reindex_content_dedup(db):
    for _ in range(3):
        _store("knowledge_book", "完全相同的书本内容独有乙")  # 3 份同内容
    _store("knowledge_book", "另一本不同的书独有丙")
    res = associative.reindex_concepts(rebuild_edges=True)
    assert res["scanned"] == 4
    assert res["unique_contents"] == 2, "同内容应折叠为一个代表 (概念图去重)"


def test_hard_delete_cleans_vector(db):
    mid = _store("episodic", "会被删的带向量记忆")
    from app.memory import ForgetIn, forget
    forget(ForgetIn(memory_id=mid, force=True))
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM memories_vec WHERE memory_id=?", (mid,)).fetchone()[0] == 0


def test_backfill_embeds_missing(db):
    with memory._conn() as c:
        import time as _t
        c.execute("INSERT INTO memories(id,kind,content,created_ts,last_recall_ts) VALUES ('m-x','episodic','补嵌入测试',?,?)",
                  (int(_t.time()), int(_t.time())))
    res = semantic.backfill()
    assert res["embedded"] >= 1
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM memories_vec WHERE memory_id='m-x'").fetchone()[0] == 1
