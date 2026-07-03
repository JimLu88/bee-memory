"""v4 关联层测试 — 概念图/FTS/共现边/forget/consolidate/connect + 默认 recall 不回退.

隔离: 把 memory/associative/spaced_repetition 三模块的 DB_PATH 指到临时库, 各测试独立.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 `import app.xxx` 可用
BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, memory, spaced_repetition  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """每个测试一个干净临时 sqlite; 三模块 DB_PATH 同步指过去."""
    p = tmp_path / "memories.sqlite"
    monkeypatch.setattr(memory, "DB_PATH", p)
    monkeypatch.setattr(associative, "DB_PATH", p)
    monkeypatch.setattr(spaced_repetition, "DB_PATH", p)
    with memory._conn():  # 触发建表
        pass
    return p


def _store(kind, content, importance=2, meta=None):
    from app.memory import StoreRequest, store
    return store(StoreRequest(kind=kind, content=content, importance=importance, meta=meta or {}))["memory_id"]


# ---------- 抽取 ----------
def test_extract_entities_zh_en_and_wikilink():
    ents = associative.extract_entities("畔色木作的利润口径核算 involves ProfitEngine [[定价倒推]]")
    assert "定价倒推" in ents          # wikilink
    assert "ProfitEngine" in ents       # 英文标识符
    assert any("利润" in e for e in ents)
    assert "可以" not in associative.extract_entities("可以做这个")  # 停用词过滤


def test_extract_wikilinks():
    assert associative.extract_wikilinks("见 [[利润口径]] 与 [[备货策略|备货]]") == ["利润口径", "备货策略"]


# ---------- schema ----------
def test_schema_tables_created(db):
    with memory._conn() as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("memories", "edges", "review_state", "concepts", "mem_concepts",
              "concept_edges", "dangling_refs", "memories_fts"):
        assert t in names, f"缺表 {t}"


# ---------- store 增量索引 ----------
def test_store_indexes_concepts_and_fts(db):
    mid = _store("episodic", "畔色木作定制家具的利润口径核算流程")
    with memory._conn() as c:
        n_mc = c.execute("SELECT COUNT(*) FROM mem_concepts WHERE memory_id=?", (mid,)).fetchone()[0]
        n_fts = c.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id=?", (mid,)).fetchone()[0]
    assert n_mc > 0, "store 应增量落概念"
    assert n_fts == 1, "store 应写 FTS"


def test_store_records_dangling(db):
    _store("episodic", "参考 [[尚未存在的概念]] 做决策")
    with memory._conn() as c:
        row = c.execute("SELECT mention_count FROM dangling_refs WHERE name=?", ("尚未存在的概念",)).fetchone()
    assert row is not None and row[0] == 1


# ---------- reindex 建记忆↔记忆共现边 (核心断路修复) ----------
def test_reindex_builds_memory_edges(db):
    a = _store("episodic", "利润口径 和 备货策略 是畔色的两个核心")
    b = _store("episodic", "重新讨论 利润口径 与 备货策略 的关系")
    c_ = _store("episodic", "今天天气不错适合散步锻炼身体")
    res = associative.reindex_concepts(rebuild_edges=True)
    assert res["status"] == "ok"
    assert res["cooccur_edges"] > 0, "共享概念的记忆之间必须建边 (否则扩散激活无油)"
    with memory._conn() as conn:
        ab = conn.execute("SELECT COUNT(*) FROM edges WHERE src=? AND dst=?", (a, b)).fetchone()[0]
        ac = conn.execute("SELECT COUNT(*) FROM edges WHERE src=? AND dst=?", (a, c_)).fetchone()[0]
    assert ab >= 1, "A-B 共享概念应连边"
    assert ac == 0, "A-C 无共享判别概念不应连边"


def test_reindex_edges_feed_spread_activation(db):
    a = _store("episodic", "海马体索引 与 扩散激活 是记忆大脑核心")
    b = _store("episodic", "再谈 海马体索引 和 扩散激活 的工程实现")
    associative.reindex_concepts(rebuild_edges=True)
    bonus = memory._spread_activation([a])
    assert bonus.get(b, 0) > 0, "共现边应让扩散激活从 A 波及 B"


# ---------- FTS 检索 ----------
def test_fts_search_finds_by_substring(db):
    mid = _store("knowledge_book", "复式记账法与借贷平衡原理在财务核算中的应用")
    with memory._conn() as c:
        hits = associative.fts_search(c, "借贷平衡", k=10)
    assert mid in hits


def test_recall_default_unchanged_uses_like(db):
    """fts=0 (默认) 必须仍是 content LIKE 行为, 不回退."""
    mid = _store("episodic", "定制家具的独特关键词ABC123")
    r = memory.recall(query="独特关键词ABC123", k=5)  # 默认 fts=0
    assert mid in [it["id"] for it in r["items"]]


def test_recall_fts_mode(db):
    mid = _store("episodic", "跨项目关联的知识图谱扩散召回机制")
    r = memory.recall(query="知识图谱扩散", k=5, fts=1)
    assert mid in [it["id"] for it in r["items"]]


# ---------- forget 实装 + 护栏 ----------
def test_forget_by_id(db):
    mid = _store("episodic", "一条可以被遗忘的低价值碎片", importance=1)
    from app.memory import ForgetIn, forget
    res = forget(ForgetIn(memory_id=mid))
    assert res["deleted"] == 1
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM memories WHERE id=?", (mid,)).fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM mem_concepts WHERE memory_id=?", (mid,)).fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id=?", (mid,)).fetchone()[0] == 0


def test_forget_protects_high_importance(db):
    mid = _store("procedural", "关键 SOP 绝不能忘", importance=5)
    from app.memory import ForgetIn, forget
    assert forget(ForgetIn(memory_id=mid))["status"] == "protected"
    assert forget(ForgetIn(memory_id=mid, force=True))["deleted"] == 1  # force 覆盖


def test_forget_dry_run(db):
    mid = _store("episodic", "碎片", importance=0)
    from app.memory import ForgetIn, forget
    assert forget(ForgetIn(memory_id=mid, dry_run=True))["deleted"] == 0
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM memories WHERE id=?", (mid,)).fetchone()[0] == 1


# ---------- consolidate 实装 ----------
def test_consolidate_returns_real_stats(db):
    _store("episodic", "利润口径 与 备货策略 A")
    _store("episodic", "利润口径 与 备货策略 B")
    res = memory.consolidate()
    assert res["status"] == "ok"
    assert res["reindex"]["status"] == "ok"
    assert res["reindex"]["scanned"] == 2
    assert "dangling_promoted" in res


def test_consolidate_promotes_dangling(db):
    _store("episodic", "关于 [[利润口径]] 的第一次讨论 利润口径")
    _store("episodic", "关于 [[利润口径]] 的第二次讨论 利润口径")
    _store("episodic", "关于 [[利润口径]] 的第三次讨论 利润口径")
    res = memory.consolidate(promote_threshold=3)
    assert res["dangling_promoted"] >= 1


# ---------- connect (A-B 关联) ----------
def test_connect_finds_path(db):
    _store("episodic", "利润口径独有词甲 的核算原则说明文档")
    _store("episodic", "利润口径独有词甲 和 备货策略独有词乙 都是核心，二者相关联")
    _store("episodic", "备货策略独有词乙 的 ABC 分层方法说明")
    associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as c:
        a_ids = associative.fts_search(c, "利润口径独有词甲", k=8)
        b_ids = associative.fts_search(c, "备货策略独有词乙", k=8)
        res = associative.connect_path(c, a_ids, b_ids, max_hops=3)
    assert res["connected"] is True, f"应能连通 A-B: {res}"


# ---------- SM-2 未被破坏 ----------
def test_spaced_repetition_still_works(db):
    mid = _store("procedural", "需要复习的重要流程")
    from app.spaced_repetition import EnrollIn, enroll, due
    enroll(mid, EnrollIn(initial_interval_days=0))
    assert any(it["id"] == mid for it in due(limit=10)["items"])
