"""v5 Phase 2 测试 — 双时序失效 / episodic→semantic 蒸馏 / 类型化边 / 语境 / FSRS.
LLM 与嵌入全 mock, 不依赖 Ollama."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, llm, memory, ppr, semantic, sleep_cycle  # noqa: E402
from app import spaced_repetition  # noqa: E402


def _fake_vec(text, timeout=None, dim=1024):
    h = int(hashlib.md5((text or "").encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    for kw in ("定价", "部署", "风险"):
        if kw in (text or ""):
            k = np.random.default_rng(int(hashlib.md5(kw.encode()).hexdigest(), 16) % (2**32))
            v += 3.0 * k.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) or 1.0)
    return v.tolist()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "m.sqlite"
    for mod in (memory, associative, semantic, ppr, spaced_repetition):
        monkeypatch.setattr(mod, "DB_PATH", p)
    monkeypatch.setattr(semantic, "embed_text", _fake_vec)
    monkeypatch.setattr(semantic, "embed_batch", lambda ts: [_fake_vec(t) for t in ts])
    monkeypatch.setattr(sleep_cycle, "VAULT_DIR", tmp_path / "vault")
    semantic.invalidate_cache(); ppr.invalidate_cache(); semantic._QCACHE.clear()
    with memory._conn():
        pass
    return p


def _store(kind, content, importance=3, mode_id=""):
    from app.memory import StoreRequest, store
    return store(StoreRequest(kind=kind, content=content, importance=importance, mode_id=mode_id))["memory_id"]


def test_invalidate_hides_from_recall(db):
    a = _store("semantic", "定价口径旧版甲")
    _store("semantic", "定价口径旧版乙")
    associative.reindex_concepts(rebuild_edges=True)
    from app.associative import InvalidateIn, invalidate_endpoint
    invalidate_endpoint(InvalidateIn(memory_id=a))
    res = associative.hybrid_recall("定价口径", k=8)
    assert a not in [it["id"] for it in res["items"]], "失效记忆不该出现在常规召回"
    with memory._conn() as c:
        assert c.execute("SELECT invalid_at FROM memories WHERE id=?", (a,)).fetchone()[0] is not None


def test_direct_supersede_is_disabled_by_governance(db):
    old = _store("semantic", "定价旧结论")
    new = _store("semantic", "定价新结论")
    from app.associative import SupersedeIn, supersede_endpoint
    with pytest.raises(Exception, match="governance/supersede"):
        supersede_endpoint(SupersedeIn(old_id=old, new_id=new))
    with memory._conn() as c:
        row = c.execute("SELECT invalid_at, superseded_by FROM memories WHERE id=?", (old,)).fetchone()
        assert row[0] is None and not row[1]


def test_exact_get_reactivates_deep_memory(db):
    mid = _store("semantic", "一个很久没有调用、但可被精确找回的决定", importance=5)
    old_ts = int(time.time()) - 365 * 86400
    with memory._conn() as c:
        c.execute(
            "UPDATE memories SET created_ts=?, last_recall_ts=?, recall_count=0, stability=14 WHERE id=?",
            (old_ts, old_ts, mid),
        )

    result = memory.get_by_id(mid)
    assert result["memory_state_before"]["tier"] == "deep"
    assert result["memory_state_after"]["tier"] == "foreground"
    assert result["memory_state_after"]["recall_count"] == 1


def test_bundle_reconstructs_long_memory_and_reactivates_all_chunks(db):
    bundle_id = "mb-1234567890abcdef"
    parts = ["第一段长期记忆。", "第二段长期记忆。", "第三段长期记忆。"]
    ids = []
    for index, part in enumerate(parts):
        stored = memory.store(memory.StoreRequest(
            kind="semantic",
            content=json.dumps({"content": part}, ensure_ascii=False),
            importance=5,
            mode_id="tachikoma:test",
            meta={
                "bundle_id": bundle_id,
                "chunk_index": index,
                "chunk_count": len(parts),
            },
        ))
        ids.append(stored["memory_id"])
    old_ts = int(time.time()) - 365 * 86400
    with memory._conn() as c:
        c.execute(
            f"UPDATE memories SET created_ts=?, last_recall_ts=?, recall_count=0, stability=14 "
            f"WHERE id IN ({','.join('?' * len(ids))})",
            [old_ts, old_ts, *ids],
        )

    result = memory.get_bundle(bundle_id)
    assert result["found"] is True
    assert result["content"] == "".join(parts)
    assert [item["id"] for item in result["items"]] == ids
    assert all(item["memory_state_before"]["tier"] == "deep" for item in result["items"])
    with memory._conn() as c:
        counts = [
            c.execute("SELECT recall_count FROM memories WHERE id=?", (mid,)).fetchone()[0]
            for mid in ids
        ]
    assert counts == [1, 1, 1]


def test_four_memory_tiers_follow_retrievability():
    now = 2_000_000_000
    row = {
        "kind": "semantic",
        "importance": 3,
        "created_ts": now - 100 * 86400,
        "recall_count": 0,
        "stability": 14.0,
    }
    expected = {
        1: "foreground",
        5: "active",
        20: "latent",
        60: "deep",
    }
    for days, tier in expected.items():
        state = memory._memory_state({**row, "last_recall_ts": now - days * 86400}, now)
        assert state["tier"] == tier


def test_distill_creates_semantic_with_provenance(db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda p, system="": {
        "title": "定价与部署口径", "insight": "促销价按日常倒推系数; 系数改动须与引擎同步部署否则失效。", "worth": True})
    _store("episodic", "定价改倒推系数", importance=4)
    _store("episodic", "定价系数须与部署同步", importance=4)
    r = sleep_cycle._distill_episodics()
    assert r["distilled"] >= 1
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM memories WHERE kind='semantic'").fetchone()[0] >= 1
        cons = c.execute("SELECT COUNT(*) FROM memories WHERE meta LIKE '%\"consolidated\": true%'").fetchone()[0]
        assert cons >= 1
        assert c.execute("SELECT COUNT(*) FROM edges WHERE kind='provenance'").fetchone()[0] >= 2


def test_distill_skips_when_llm_down(db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    _store("episodic", "某经历", importance=5)
    r = sleep_cycle._distill_episodics()
    assert r.get("distilled", 0) == 0 and "skipped" in r


def test_distill_never_touches_books(db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda p, system="": {"title": "x", "insight": "y", "worth": True})
    _store("knowledge_book", "某本书的内容", importance=5)
    r = sleep_cycle._distill_episodics()
    assert r["distilled"] == 0, "书本不参与经验固化"


def test_typed_edges_labels_relation(db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda p, system="": {
        "rel": "causes", "direction": "a_to_b", "because": "改动系数导致弹窗失效"})
    _store("semantic", "定价系数改动 甲")
    _store("semantic", "定价弹窗失效 乙")
    associative.reindex_concepts(rebuild_edges=True)
    r = sleep_cycle._typed_edges()
    assert r["typed"] >= 1
    with memory._conn() as c:
        te = c.execute("SELECT rel_type, because FROM typed_edges").fetchone()
        assert te[0] == "causes" and "弹窗" in te[1]


def test_typed_edges_none_skipped(db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda p, system="": {"rel": "none"})
    _store("semantic", "毫不相关的甲")
    _store("semantic", "毫不相关的乙")
    associative.reindex_concepts(rebuild_edges=True)
    sleep_cycle._typed_edges()
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM typed_edges").fetchone()[0] == 0


def test_encoding_context_boost(db):
    m1 = _store("semantic", "定价口径 项目A版", mode_id="projA")
    _store("semantic", "定价口径 项目B版", mode_id="projB")
    associative.reindex_concepts(rebuild_edges=True)
    no_boost = associative.hybrid_recall("定价口径", k=2)
    boosted = associative.hybrid_recall("定价口径", k=2, boost_mode="projA")
    a_boosted = next((it["score"] for it in boosted["items"] if it["id"] == m1), None)
    a_plain = next((it["score"] for it in no_boost["items"] if it["id"] == m1), None)
    assert a_boosted is not None and a_plain is not None
    assert a_boosted > a_plain


def test_memory_recall_filters_invalid(db):
    """回归 bug#1/#4: memory.recall (persona/默认路径) 也要过滤失效记忆."""
    a = _store("knowledge_book", "定价口径旧版甲")
    _store("knowledge_book", "定价口径新版乙")
    from app.associative import InvalidateIn, invalidate_endpoint
    invalidate_endpoint(InvalidateIn(memory_id=a))
    from app.memory import recall
    r = recall(query="定价口径", k=8, fts=0)
    assert a not in [it["id"] for it in r["items"]], "memory.recall 默认路径不该回失效记忆"
    r2 = recall(query="定价口径", k=8, fts=1)
    assert a not in [it["id"] for it in r2["items"]], "memory.recall FTS 路径也不该回失效记忆"


def test_provenance_edge_survives_reindex(db, monkeypatch):
    """回归 bug#5/#7: 蒸馏溯源边 (kind=provenance) 不被 reindex 清掉."""
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda p, system="": {"title": "T", "insight": "知识精华内容", "worth": True})
    _store("episodic", "定价改倒推系数甲", importance=4)
    _store("episodic", "定价系数须与部署同步乙", importance=4)
    sleep_cycle._distill_episodics()
    with memory._conn() as c:
        before = c.execute("SELECT COUNT(*) FROM edges WHERE kind='provenance'").fetchone()[0]
    associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as c:
        after = c.execute("SELECT COUNT(*) FROM edges WHERE kind='provenance'").fetchone()[0]
    assert before >= 2 and after == before, "reindex 不能清掉 provenance 边"


def test_forget_cleans_typed_edges(db, monkeypatch):
    """回归 bug#13: forget 要清 typed_edges."""
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda p, system="": {
        "rel": "causes", "direction": "a_to_b", "because": "甲导致乙"})
    a = _store("semantic", "定价系数改动 甲")
    _store("semantic", "定价弹窗失效 乙")
    associative.reindex_concepts(rebuild_edges=True)
    sleep_cycle._typed_edges()
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM typed_edges").fetchone()[0] >= 1
    from app.memory import ForgetIn, forget
    forget(ForgetIn(memory_id=a, force=True))
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM typed_edges WHERE src=? OR dst=?", (a, a)).fetchone()[0] == 0


def test_load_matrix_dim_guard(db):
    """回归 R2#1: memories_vec 混维时 vector_search 不该崩溃 (跳过异维行)."""
    _store("semantic", "定价正常向量记忆")
    import struct
    with memory._conn() as c:  # 塞一条错维向量 (512)
        bad = struct.pack("512f", *([0.1] * 512))
        c.execute("INSERT OR REPLACE INTO memories_vec(memory_id,dim,vec,embedded_ts) VALUES ('m-bad',512,?,0)", (bad,))
    semantic.invalidate_cache()
    with memory._conn() as c:
        hits = semantic.vector_search(c, "定价", k=3)  # 不抛异常
    assert isinstance(hits, list)


def test_relation_search_reads_embedding(db):
    """回归 R2#5: typed_edges.because 向量能被语义检索 (关系可检索)."""
    import struct
    emb = _fake_vec("定价系数导致弹窗失效")
    eb = struct.pack(f"{len(emb)}f", *emb)
    with memory._conn() as c:
        c.execute("INSERT INTO typed_edges(id,src,dst,rel_type,because,weight,embedding,created_ts) "
                  "VALUES ('te-1','a','b','causes','定价系数导致弹窗失效',2.0,?,0)", (eb,))
    with memory._conn() as c:
        hits = associative.relation_search(c, "定价系数导致弹窗失效", k=3, min_sim=0.5)
    assert hits and hits[0]["rel"] == "causes"


def test_invalidate_cleans_review_and_cooccur(db):
    """回归 R2#2/#8: invalidate 立即退复习闸 + 摘共现边."""
    mid = _store("semantic", "会失效的重要知识 关联甲", importance=5)  # 自动入复习闸
    _store("semantic", "关联甲 的另一条")
    associative.reindex_concepts(rebuild_edges=True)
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM review_state WHERE memory_id=?", (mid,)).fetchone()[0] == 1
    from app.associative import InvalidateIn, invalidate_endpoint
    invalidate_endpoint(InvalidateIn(memory_id=mid))
    with memory._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM review_state WHERE memory_id=?", (mid,)).fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM edges WHERE (src=? OR dst=?) AND kind='cooccur'",
                         (mid, mid)).fetchone()[0] == 0


def test_due_excludes_invalid(db):
    """回归 R2#2: 失效记忆不再出现在复习队列."""
    import time as _t
    mid = _store("semantic", "重要待复习", importance=5)
    with memory._conn() as c:
        c.execute("UPDATE review_state SET next_review_ts=? WHERE memory_id=?", (int(_t.time()) - 10, mid))
    from app.spaced_repetition import due
    assert mid in [it["id"] for it in due(limit=20)["items"]]
    from app.associative import InvalidateIn, invalidate_endpoint
    invalidate_endpoint(InvalidateIn(memory_id=mid))
    assert mid not in [it["id"] for it in due(limit=20)["items"]]


def test_sleep_cycle_lock_skips_second(db, monkeypatch, tmp_path):
    """回归 R2#7: 已持锁时第二次 run_sleep_cycle 直接跳过."""
    monkeypatch.setattr(llm, "available", lambda: False)
    acq = sleep_cycle._acquire_lock()
    assert acq is not None
    lock, token = acq
    r2 = sleep_cycle.run_sleep_cycle()  # 锁被占
    assert r2["status"] == "skipped_already_running"
    # 别人的锁不被误删: 用错 token 释放不删
    sleep_cycle._release_lock(lock, "wrong-token")
    assert lock.exists()
    sleep_cycle._release_lock(lock, token)
    assert not lock.exists()


def test_stability_uses_review_state(db):
    mid = _store("semantic", "会被复习的知识", importance=3)
    with memory._conn() as c:
        c.execute("INSERT OR REPLACE INTO review_state(memory_id,ef,interval_days,repetitions,next_review_ts,last_grade) "
                  "VALUES (?,2.6,30,5,?,5)", (mid, 9999999999))
        sleep_cycle._update_stability(c)
        s_reviewed = c.execute("SELECT stability FROM memories WHERE id=?", (mid,)).fetchone()[0]
    mid2 = _store("semantic", "没复习的知识", importance=3)
    with memory._conn() as c:
        sleep_cycle._update_stability(c)
        s_plain = c.execute("SELECT stability FROM memories WHERE id=?", (mid2,)).fetchone()[0]
    assert s_reviewed > s_plain, "复习过的记忆存储强度更高 (更久不忘)"
