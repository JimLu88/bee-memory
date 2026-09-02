from __future__ import annotations

import sys
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, cognitive, memory  # noqa: E402


def _use_temp_db(monkeypatch, tmp_path):
    db = tmp_path / "cognitive.sqlite"
    monkeypatch.setattr(memory, "DB_PATH", db)
    return db


def test_store_registers_typed_multilabel_memory(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(associative, "index_one_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(memory.semantic, "embed_and_store", lambda *args, **kwargs: None)
    result = memory.store(memory.StoreRequest(
        kind="procedural",
        content="先核对产品清单，再安排样品验收。",
        importance=4,
        meta={
            "source_id": "test:procedure:1",
            "domains": ["工作", "供应链"],
            "entities": ["样品"],
            "confidence": 0.95,
            "evidence": [{"type": "note", "locator": "obsidian://work/sample"}],
        },
    ))
    with memory._conn() as conn:
        row = conn.execute(
            "SELECT memory_form,lifecycle_stage,confidence,source_id FROM memory_cognitive WHERE memory_id=?",
            (result["memory_id"],),
        ).fetchone()
        facets = set(conn.execute(
            "SELECT facet_type,facet_value FROM memory_facets WHERE memory_id=?",
            (result["memory_id"],),
        ).fetchall())
    assert row == ("procedural", "consolidated", 0.95, "test:procedure:1")
    with memory._conn() as conn:
        assert conn.execute(
            "SELECT memory_tier,verification_status,review_status,owner_id FROM memory_cognitive WHERE memory_id=?",
            (result["memory_id"],),
        ).fetchone() == ("M0", "unreviewed", "quarantine", "jim")
    assert ("domain", "工作") in facets
    assert ("domain", "供应链") in facets
    assert ("entity", "样品") in facets


def test_hippocampal_stage_is_idempotent(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    req = cognitive.HippocampalStageRequest(
        summary="用户确认以后优先检查任务依赖。",
        idempotency_key="turn:test:1",
        domains=["工作"],
        entities=["任务依赖"],
    )
    first = cognitive.stage_hippocampal(req)
    second = cognitive.stage_hippocampal(req)
    assert first["deduplicated"] is False
    assert second["deduplicated"] is True
    assert second["episode_id"] == first["episode_id"]


def test_hippocampal_stage_rejects_secrets(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as error:
        cognitive.stage_hippocampal(cognitive.HippocampalStageRequest(
            summary="API key: sk-abcdefghijklmnopqrstuvwxyz123456",
            idempotency_key="turn:secret:1",
        ))
    assert error.value.status_code == 400


def test_planned_recall_filters_books_and_respects_budget(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    with memory._conn() as conn:
        now = cognitive._now()
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) VALUES (?,?,?,?,?,?,?)",
            [
                ("m-work", "procedural", "样品先核验，再发货。", now, now, "{}", 40),
                ("m-book", "knowledge_book", "书中关于质量管理的章节。", now, now, "{}", 40),
            ],
        )
        cognitive.register_memory(conn, "m-work", "procedural", {"domains": ["工作"]})
        cognitive.register_memory(conn, "m-book", "knowledge_book", {"source": "books", "domains": ["工作"]})
    monkeypatch.setattr(cognitive, "_rrf_candidates", lambda *args, **kwargs: (
        [
            {"id": "m-work", "kind": "procedural", "score": 0.8, "snippet": "样品先核验，再发货。", "token_count": 40, "via": "字面"},
            {"id": "m-book", "kind": "knowledge_book", "score": 0.9, "snippet": "书中章节", "token_count": 40, "via": "语义"},
        ],
        {"channels": {"test": 2}, "partial": False, "stop_reason": "complete", "candidate_audit": []},
    ))
    result = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="样品发货", depth="L2", domains=["工作"], token_budget=100,
    ))
    assert [item["id"] for item in result["items"]] == ["m-work"]
    assert result["token_count"] <= 100
    deep = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="质量管理", depth="L3", domains=["工作"], include_books=True,
    ))
    assert {item["id"] for item in deep["items"]} == {"m-work", "m-book"}


def test_memory_form_is_a_boost_not_a_hard_filter(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    with memory._conn() as conn:
        now = cognitive._now()
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) VALUES (?,?,?,?,?,?,?)",
            [
                ("m-sem", "semantic", "A useful project fact", now, now, "{}", 20),
                ("m-proc", "procedural", "A useful project procedure", now, now, "{}", 20),
            ],
        )
        cognitive.register_memory(conn, "m-sem", "semantic", {"domains": ["work"]})
        cognitive.register_memory(conn, "m-proc", "procedural", {"domains": ["work"]})
    monkeypatch.setattr(cognitive, "_rrf_candidates", lambda *args, **kwargs: ([
                {"id": "m-sem", "kind": "semantic", "score": 0.9, "snippet": "fact", "token_count": 20},
                {"id": "m-proc", "kind": "procedural", "score": 0.8, "snippet": "procedure", "token_count": 20},
            ], {"channels": {"test": 2}, "partial": False,
                "stop_reason": "complete", "elapsed_ms": 0.1}))
    result = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="project", depth="L2", domains=["work"], memory_forms=["procedural"],
    ))
    assert {item["id"] for item in result["items"]} == {"m-sem", "m-proc"}
    assert result["items"][0]["id"] == "m-sem"
    assert "记忆类型:procedural" in result["items"][1]["why_recalled"]


def test_legacy_memories_are_backfilled_in_bounded_batches(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    with memory._conn() as conn:
        now = cognitive._now()
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) VALUES (?,?,?,?,?,?,?)",
            [
                (f"legacy-{index}", "knowledge_book" if index == 2 else "episodic",
                 f"legacy memory {index}", now, now, "{}", 10)
                for index in range(3)
            ],
        )
    # Merely reconnecting must not migrate the entire legacy table.
    with memory._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_cognitive").fetchone()[0] == 0
    first = cognitive.cognitive_backfill(cognitive.CognitiveBackfillRequest(limit=2))
    assert first["processed"] == 2
    assert first["complete"] is False
    second = cognitive.cognitive_backfill(cognitive.CognitiveBackfillRequest(limit=2))
    assert second["processed"] == 1
    assert second["complete"] is True
    with memory._conn() as conn:
        rows = conn.execute(
            "SELECT memory_form,lifecycle_stage FROM memory_cognitive ORDER BY memory_id"
        ).fetchall()
    assert rows == [("episodic", "consolidated"), ("episodic", "consolidated"),
                    ("semantic", "deep")]


def test_hard_delete_cleans_cognitive_metadata(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.execute(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) "
            "VALUES (?,?,?,?,?,?,?)",
            ("live", "episodic", "live memory", now, now, "{}", 10),
        )
        cognitive.register_memory(conn, "live", "episodic", {
            "domains": ["work"],
            "evidence": [{"type": "note", "locator": "test://live"}],
        })
    with memory._conn() as conn:
        memory._hard_delete(conn, "live")
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_cognitive WHERE memory_id='live'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_facets WHERE memory_id='live'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id='live'"
        ).fetchone()[0] == 0
    stats = cognitive.cognitive_stats()
    assert stats["total_memories"] == 0
    assert stats["classified_memories"] == 0


def test_acl_is_resolved_before_retrieval(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            [
                ("mine", "semantic", "my private memory", now, now, "{}"),
                ("other", "semantic", "another private memory", now, now, "{}"),
                ("shared", "semantic", "explicitly shared memory", now, now, "{}"),
            ],
        )
        cognitive.register_memory(conn, "mine", "semantic", {"owner_id": "jim"}, trusted_governance=True)
        cognitive.register_memory(conn, "other", "semantic", {"owner_id": "alice"}, trusted_governance=True)
        cognitive.register_memory(conn, "shared", "semantic", {
            "owner_id": "alice", "visibility": "restricted",
            "acl": [{"type": "user", "id": "jim", "permission": "read"}],
        }, trusted_governance=True)
        allowed = cognitive._allowed_memory_ids(
            conn, principal_type="user", principal_id="jim", roles=[]
        )
    assert allowed == {"mine", "shared"}


def test_rrf_prefers_items_supported_by_multiple_channels(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            [
                ("both", "semantic", "supported by lexical and vector", now, now, "{}"),
                ("lex", "semantic", "lexical only", now, now, "{}"),
                ("vec", "semantic", "vector only", now, now, "{}"),
            ],
        )
        monkeypatch.setattr(associative, "fts_search", lambda *args, **kwargs: ["both", "lex"])
        monkeypatch.setattr(memory.semantic, "vector_search", lambda *args, **kwargs: [
            ("both", 0.8), ("vec", 0.75),
        ])
        items, trace = cognitive._rrf_candidates(
            conn, "query", allowed={"both", "lex", "vec"},
            budget={"lexical": 4, "vector": 4, "graph": 0}, timeout_ms=500,
        )
    assert items[0]["id"] == "both"
    assert items[0]["channel_ranks"] == {"lexical": 1, "vector": 1, "graph": None}
    assert trace["channels"] == {"lexical": 2, "vector": 2, "graph": 0}


def test_rrf_filters_ranked_ids_by_acl_without_enumerating_whole_library(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            [
                ("mine", "semantic", "authorized result", now, now, "{}"),
                ("secret", "semantic", "unauthorized result", now, now, "{}"),
            ],
        )
        cognitive.register_memory(conn, "mine", "semantic", {"owner_id": "jim"}, trusted_governance=True)
        cognitive.register_memory(conn, "secret", "semantic", {"owner_id": "alice"}, trusted_governance=True)
        monkeypatch.setattr(associative, "fts_search", lambda *args, **kwargs: ["secret", "mine"])
        monkeypatch.setattr(memory.semantic, "vector_search", lambda *args, **kwargs: [
            ("secret", 0.9), ("mine", 0.8),
        ])
        items, trace = cognitive._rrf_candidates(
            conn, "query", allowed=None,
            budget={"lexical": 4, "vector": 4, "graph": 0}, timeout_ms=500,
            principal_type="user", principal_id="jim", roles=[], enforce_acl=True,
        )
    assert [item["id"] for item in items] == ["mine"]
    assert trace["channels"] == {"lexical": 1, "vector": 1, "graph": 0}


def test_rrf_reports_partial_when_lexical_deadline_interrupts(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    with memory._conn() as conn:
        monkeypatch.setattr(
            associative, "fts_search",
            lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("interrupted")),
        )
        monkeypatch.setattr(memory.semantic, "vector_search", lambda *args, **kwargs: [])
        items, trace = cognitive._rrf_candidates(
            conn, "query", allowed=set(),
            budget={"lexical": 4, "vector": 0, "graph": 0}, timeout_ms=50,
        )
    assert items == []
    assert trace["partial"] is True
    assert trace["stop_reason"] == "timeout"


def test_rrf_hard_deadline_interrupts_slow_virtual_table_work(monkeypatch, tmp_path):
    """A virtual-table-like rank cannot escape the recall wall-clock budget."""
    _use_temp_db(monkeypatch, tmp_path)

    def deliberately_slow_fts(conn, query, k=20, **kwargs):
        conn.execute(
            """WITH RECURSIVE count_up(x) AS (
                 VALUES(0) UNION ALL SELECT x + 1 FROM count_up WHERE x < 1000000000
               ) SELECT sum(x) FROM count_up"""
        ).fetchone()
        return []

    monkeypatch.setattr(associative, "fts_search", deliberately_slow_fts)
    monkeypatch.setattr(memory.semantic, "vector_search", lambda *args, **kwargs: [])
    with memory._conn() as conn:
        started = time.perf_counter()
        items, trace = cognitive._rrf_candidates(
            conn, "broad query", allowed=set(),
            budget={"lexical": 4, "vector": 0, "graph": 0}, timeout_ms=50,
        )
        elapsed = time.perf_counter() - started

    assert items == []
    assert trace["partial"] is True
    assert trace["stop_reason"] == "timeout"
    assert elapsed < 0.5


def test_bounded_fts_keeps_bm25_and_cross_fragment_coverage(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            [
                ("strong-alpha", "semantic", "alpha project alpha project alpha", now, now, "{}"),
                ("beta-only", "semantic", "beta planning details", now, now, "{}"),
                ("noise", "semantic", "unrelated material", now, now, "{}"),
            ],
        )
        for memory_id, content in conn.execute("SELECT id,content FROM memories"):
            associative.index_one_memory(conn, memory_id, content)
        ids = associative.fts_search(
            conn, "alpha beta", k=3, candidate_pool=12,
        )

    assert ids[0] == "strong-alpha"
    assert "beta-only" in ids
    assert "noise" not in ids


def test_hot_fts_excludes_deep_books_and_rebuilds_from_lifecycle(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.executemany(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            [
                ("hot", "semantic", "quality sample current project", now, now, "{}"),
                ("deep", "knowledge_book", "quality sample archived book", now, now, "{}"),
            ],
        )
        for memory_id, kind in (("hot", "semantic"), ("deep", "knowledge_book")):
            content = conn.execute("SELECT content FROM memories WHERE id=?", (memory_id,)).fetchone()[0]
            associative.index_one_memory(conn, memory_id, content)
            cognitive.register_memory(conn, memory_id, kind, {"owner_id": "jim"})
        hot_ids = associative.fts_search(
            conn, "quality sample", k=10, candidate_pool=20, scope="hot",
        )
        all_ids = associative.fts_search(
            conn, "quality sample", k=10, candidate_pool=20, scope="all",
        )
        rebuilt = cognitive.rebuild_hot_fts(conn)

    assert hot_ids == ["hot"]
    assert set(all_ids) == {"hot", "deep"}
    assert rebuilt["rows"] == 1


def test_hot_fts_warmup_sets_readiness_after_bounded_queries(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.execute(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            ("warm", "semantic", "塔奇克马 记忆 项目 任务", now, now, "{}"),
        )
        associative.index_one_memory(conn, "warm", "塔奇克马 记忆 项目 任务")
        cognitive.register_memory(conn, "warm", "semantic", {"owner_id": "jim"})
    monkeypatch.setattr(associative, "_HOT_FTS_READY", False)
    monkeypatch.setattr(associative, "_HOT_FTS_WARMING", False)

    assert associative.warm_hot_fts() is True
    assert associative._HOT_FTS_READY is True
    assert associative._HOT_FTS_WARMING is False


def test_planned_recall_returns_trace_citations_and_hard_budgets(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.execute(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) VALUES (?,?,?,?,?,?,?)",
            ("m1", "semantic", "a concise verified fact", now, now, "{}", 20),
        )
        cognitive.register_memory(conn, "m1", "semantic", {
            "owner_id": "jim", "verified": True,
            "memory_tier": "M1", "verification_status": "verified", "review_status": "active",
            "source_id": "source:1", "root_source_id": "root:1",
            "evidence": [{"type": "note", "locator": "obsidian://source/1",
                          "excerpt": "verified fact", "observed_at": "2026-08-10"}],
        }, trusted_governance=True)
    monkeypatch.setattr(cognitive, "_rrf_candidates", lambda *args, **kwargs: ([{
        "id": "m1", "kind": "semantic", "score": 0.03, "rrf_score": 0.03,
        "snippet": "a concise verified fact", "token_count": 20,
        "channel_ranks": {"lexical": 1, "vector": 2, "graph": None},
        "via": "lexical+vector", "source": "",
    }], {"channels": {"lexical": 1, "vector": 1, "graph": 0},
         "partial": False, "stop_reason": "complete", "elapsed_ms": 1.0}))
    result = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="fact", query_depth="Q1", enforce_acl=True,
        principal_id="jim", max_chars=120, timeout_ms=200,
    ))
    assert result["query_depth"] == "Q1"
    assert result["items"][0]["memory_tier"] == "M1"
    assert result["citation_pack"][0]["locators"] == ["obsidian://source/1"]
    assert result["retrieval_path"][0] == "acl_filter"
    assert result["budget"]["timeout_ms"] == 200


def _register_governed_memory(conn, memory_id, *, tier="M0", verified=False,
                              evidence=None, activation_count=0):
    now = cognitive._now()
    conn.execute(
        "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) "
        "VALUES (?,?,?,?,?,?,?)",
        (memory_id, "semantic", f"governed memory {memory_id}", now, now, "{}", 20),
    )
    cognitive.register_memory(conn, memory_id, "semantic", {
        "memory_tier": tier,
        "verification_status": "verified" if verified else "unreviewed",
        "review_status": "active" if verified else "quarantine",
        "evidence": list(evidence or []),
        "activation_count": activation_count,
        "owner_id": "jim",
        "source": "test",
        "source_id": f"test:{memory_id}",
        "observed_at": "2026-08-08",
        "valid_from": "2026-08-08",
        "valid_to": "2027-08-08",
    }, trusted_governance=True)
    for index in range(activation_count):
        occurred_date = "2026-08-08" if index == 0 else "2026-08-09"
        conn.execute(
            """INSERT OR IGNORE INTO memory_activations(
                 event_id,memory_id,activation_kind,session_id,occurred_date,evidence_locator,
                 principal_type,principal_id,created_ts) VALUES (?,?,?,?,?,?,?,?,?)""",
            (f"act-{memory_id}-{index}", memory_id, "adopted", f"session-{index}",
             occurred_date, f"turn://{memory_id}/{index}", "user", "jim", now + index),
        )


def test_M1_promotion_requires_and_persists_evidence(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    with memory._conn() as conn:
        _register_governed_memory(conn, "candidate")
    denied = cognitive.promote_governed_memory(cognitive.GovernedPromotionRequest(
        memory_id="candidate", to_tier="M1", evidence=[],
    ))
    assert denied["promoted"] is False
    assert "complete_evidence_required" in denied["reasons"]

    promoted = cognitive.promote_governed_memory(cognitive.GovernedPromotionRequest(
        memory_id="candidate", to_tier="M1",
        evidence=[{"type": "episode", "locator": "turn://verified/1",
                   "excerpt": "user-confirmed statement", "observed_at": "2026-08-10"}],
        run_id="night-test-1",
        source="chat", source_id="turn:verified:1", observed_at="2026-08-10",
        valid_from="2026-08-10", valid_to="2027-08-10",
    ))
    assert promoted["promoted"] is True
    with memory._conn() as conn:
        assert conn.execute(
            "SELECT memory_tier,verification_status,evidence_count FROM memory_cognitive WHERE memory_id='candidate'"
        ).fetchone() == ("M1", "verified", 1)
        assert conn.execute(
            "SELECT locator FROM memory_evidence WHERE memory_id='candidate'"
        ).fetchone()[0] == "turn://verified/1"


def test_M2_promotion_uses_verified_parents_and_rejects_stale_state(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    evidence = [{"type": "episode", "locator": "turn://parent",
                 "excerpt": "parent evidence", "observed_at": "2026-08-08"}]
    with memory._conn() as conn:
        _register_governed_memory(conn, "parent-1", tier="M1", verified=True,
                                  evidence=evidence)
        _register_governed_memory(conn, "parent-2", tier="M1", verified=True,
                                  evidence=evidence)
        _register_governed_memory(conn, "summary", tier="M1", verified=True,
                                  evidence=[{"type": "episode", "locator": "turn://summary",
                                             "excerpt": "summary evidence", "observed_at": "2026-08-08"}])

    promoted = cognitive.promote_governed_memory(cognitive.GovernedPromotionRequest(
        memory_id="summary", to_tier="M2",
        parent_memory_ids=["parent-1", "parent-2"],
        run_id="night-test-2", expected_view_version=1,
    ))
    assert promoted["promoted"] is True
    assert promoted["shared_state_version"] == 2
    with pytest.raises(HTTPException) as error:
        cognitive.promote_governed_memory(cognitive.GovernedPromotionRequest(
            memory_id="summary", to_tier="M3", expected_view_version=1,
            evidence=[{"type": "review", "locator": "review://second",
                       "excerpt": "second review", "observed_at": "2026-08-10"}],
        ))
    assert error.value.status_code == 409
    with memory._conn() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_lineage WHERE child_memory_id='summary' AND relation_type='summarizes'"
        ).fetchone()[0] == 2


def test_nightly_governance_promotes_only_eligible_M2(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    evidence = [
        {"type": "episode", "locator": "turn://one", "excerpt": "one", "observed_at": "2026-08-08"},
        {"type": "review", "locator": "review://two", "excerpt": "two", "observed_at": "2026-08-09"},
    ]
    with memory._conn() as conn:
        _register_governed_memory(conn, "eligible", tier="M2", verified=True,
                                  evidence=evidence, activation_count=3)
        _register_governed_memory(conn, "too-cold", tier="M2", verified=True,
                                  evidence=evidence, activation_count=2)
    result = cognitive.run_nightly_governance(cognitive.NightlyGovernanceRequest(
        run_id="night-test-3", limit=100,
    ))
    assert result["promoted_to_M3"] == ["eligible"]
    with memory._conn() as conn:
        rows = dict(conn.execute(
            "SELECT memory_id,memory_tier FROM memory_cognitive WHERE memory_id IN ('eligible','too-cold')"
        ).fetchall())
    assert rows == {"eligible": "M3", "too-cold": "M2"}


def test_ordinary_store_ignores_forged_tier_identity_and_acl(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(associative, "index_one_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(memory.semantic, "embed_and_store", lambda *args, **kwargs: None)
    result = memory.store(memory.StoreRequest(
        kind="semantic", content="ordinary write must remain quarantined",
        meta={
            "memory_tier": "M3", "verification_status": "verified",
            "owner_id": "alice", "visibility": "team",
            "acl": [{"type": "user", "id": "alice", "permission": "admin"}],
        },
    ))
    with memory._conn() as conn:
        row = conn.execute(
            "SELECT memory_tier,verification_status,review_status,owner_id,visibility FROM memory_cognitive WHERE memory_id=?",
            (result["memory_id"],),
        ).fetchone()
        acl = conn.execute(
            "SELECT principal_type,principal_id,permission FROM memory_acl WHERE memory_id=?",
            (result["memory_id"],),
        ).fetchall()
    assert row == ("M0", "unreviewed", "quarantine", "jim", "private")
    assert acl == [("user", "jim", "admin")]


def test_first_result_obeys_character_and_token_budgets(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.execute(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) VALUES (?,?,?,?,?,?,?)",
            ("big", "semantic", "长" * 500, now, now, "{}", 500),
        )
        cognitive.register_memory(conn, "big", "semantic", {})
    monkeypatch.setattr(cognitive, "_rrf_candidates", lambda *args, **kwargs: ([{
        "id": "big", "kind": "semantic", "rrf_score": 0.03,
        "snippet": "长" * 360, "token_count": 500,
        "channel_ranks": {"lexical": 1, "vector": None, "graph": None},
        "via": "lexical", "source": "",
    }], {"channels": {"lexical": 1, "vector": 0, "graph": 0},
         "partial": False, "stop_reason": "complete", "candidate_audit": []}))
    result = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="长内容", query_depth="Q1", token_budget=100, max_chars=30,
    ))
    assert result["result_count"] == 1
    assert result["char_count"] <= 30
    assert result["token_count"] <= 100
    assert len(result["items"][0]["snippet"]) <= 30


def test_seen_does_not_count_as_adoption(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    now = cognitive._now()
    with memory._conn() as conn:
        conn.execute(
            "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta) VALUES (?,?,?,?,?,?)",
            ("seen", "semantic", "visible memory", now, now, "{}"),
        )
        cognitive.register_memory(conn, "seen", "semantic", {})
    cognitive._enqueue_retrieval_event({
        "created_ts": now,
        "chosen_ids": ["seen"],
        "row": ("run-seen", "", "query", "L1", 1, 1, 1, 1.0, "{}", now),
        "item_rows": [],
    })
    cognitive._RETRIEVAL_EVENTS.join()
    with memory._conn() as conn:
        row = conn.execute(
            "SELECT seen_count,activation_count,adopted_count,confirmed_count FROM memory_cognitive WHERE memory_id='seen'"
        ).fetchone()
    assert row == (1, 0, 0, 0)


def test_governed_supersede_closes_lifecycle(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    evidence = [{"type": "note", "locator": "turn://fact", "excerpt": "fact", "observed_at": "2026-08-10"}]
    with memory._conn() as conn:
        _register_governed_memory(conn, "old", tier="M1", verified=True, evidence=evidence)
        _register_governed_memory(conn, "new", tier="M1", verified=True, evidence=evidence)
        conn.execute(
            "INSERT INTO memory_conflicts(conflict_id,left_memory_id,right_memory_id,reason,status,created_ts) VALUES (?,?,?,?,?,?)",
            ("conf-old-new", "old", "new", "facts differ", "open", cognitive._now()),
        )
    result = cognitive.supersede_governed_memory(cognitive.GovernedSupersedeRequest(
        old_memory_id="old", new_memory_id="new", evidence_locator="review://approved",
    ))
    assert result["ok"] is True
    with memory._conn() as conn:
        old = conn.execute("SELECT invalid_at,superseded_by FROM memories WHERE id='old'").fetchone()
        old_review = conn.execute("SELECT review_status FROM memory_cognitive WHERE memory_id='old'").fetchone()[0]
        new_parent = conn.execute("SELECT supersedes_memory_id FROM memory_cognitive WHERE memory_id='new'").fetchone()[0]
        conflict = conn.execute("SELECT status,winning_memory_id FROM memory_conflicts WHERE conflict_id='conf-old-new'").fetchone()
    assert old[0] is not None and old[1] == "new"
    assert old_review == "retired"
    assert new_parent == "old"
    assert conflict == ("resolved", "new")


def test_dream_accessibility_is_idempotent_and_never_counts_for_promotion(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    evidence = [{"type": "note", "locator": "turn://verified", "excerpt": "fact"}]
    with memory._conn() as conn:
        _register_governed_memory(conn, "stable", tier="M1", verified=True, evidence=evidence)
        _register_governed_memory(conn, "unstable", tier="M0", verified=False)
    req = cognitive.DreamAccessibilityRequest(
        dream_note_id="dream-test-1",
        dream_date="2026-09-02",
        mentions=[
            {"memory_id": "stable", "phase_count": 3, "segment_count": 3},
            {"memory_id": "unstable", "phase_count": 1, "segment_count": 1},
        ],
    )

    first = cognitive.record_dream_accessibility(req)
    second = cognitive.record_dream_accessibility(req)

    assert first["accepted"][0]["base_delta"] == 0.035
    assert first["accepted"][0]["deduplicated"] is False
    assert first["rejected"] == [{"memory_id": "unstable", "reason": "stable_tier_required"}]
    assert second["accepted"][0]["deduplicated"] is True
    with memory._conn() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_dream_accessibility_events WHERE memory_id='stable'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT activation_count,adopted_count,confirmed_count,memory_tier,confidence "
            "FROM memory_cognitive WHERE memory_id='stable'"
        ).fetchone() == (0, 0, 0, "M1", 0.7)


def test_dream_accessibility_decays_caps_and_only_reranks_when_enabled(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    evidence = [{"type": "note", "locator": "turn://verified", "excerpt": "fact"}]
    with memory._conn() as conn:
        _register_governed_memory(conn, "dreamed", tier="M1", verified=True, evidence=evidence)
        _register_governed_memory(conn, "plain", tier="M1", verified=True, evidence=evidence)
    for index in range(6):
        result = cognitive.record_dream_accessibility(cognitive.DreamAccessibilityRequest(
            dream_note_id=f"dream-cap-{index}",
            dream_date="2026-09-02",
            mentions=[{"memory_id": "dreamed", "phase_count": 3, "segment_count": 3}],
        ))
        assert result["ok"] is True
    with memory._conn() as conn:
        now_boost = cognitive._dream_accessibility_map(conn, ["dreamed"], on_date="2026-09-02")
        half_boost = cognitive._dream_accessibility_map(conn, ["dreamed"], on_date="2026-10-02")
    assert now_boost["dreamed"] == 0.12
    assert half_boost["dreamed"] == pytest.approx(0.105, abs=0.002)

    candidates = [
        {"id": "plain", "kind": "semantic", "score": 0.03, "rrf_score": 0.03,
         "snippet": "plain", "token_count": 20, "via": "lexical"},
        {"id": "dreamed", "kind": "semantic", "score": 0.03, "rrf_score": 0.03,
         "snippet": "dreamed", "token_count": 20, "via": "lexical"},
    ]
    monkeypatch.setattr(cognitive, "_rrf_candidates", lambda *args, **kwargs: (
        candidates, {"channels": {"test": 2}, "partial": False,
                     "stop_reason": "complete", "candidate_audit": []},
    ))
    enabled = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="same", depth="L2", apply_dream_accessibility=True,
    ))
    disabled = cognitive.recall_planned(cognitive.PlannedRecallRequest(
        query="same", depth="L2", apply_dream_accessibility=False,
    ))

    assert enabled["items"][0]["id"] == "dreamed"
    assert enabled["items"][0]["dream_accessibility_boost"] > 0
    assert "梦境可达性" in enabled["items"][0]["why_recalled"]
    assert disabled["items"][0]["id"] == "plain"
    assert all(item["dream_accessibility_boost"] == 0 for item in disabled["items"])


def test_quiet_dream_seed_recall_is_acl_filtered_stable_deterministic_and_read_only(
    monkeypatch, tmp_path,
):
    _use_temp_db(monkeypatch, tmp_path)
    evidence = [{"type": "note", "locator": "turn://verified", "excerpt": "fact"}]
    with memory._conn() as conn:
        _register_governed_memory(conn, "stable-a", tier="M1", verified=True, evidence=evidence)
        _register_governed_memory(conn, "stable-b", tier="M3", verified=True, evidence=evidence)
        _register_governed_memory(conn, "unstable", tier="M0", verified=False)
        _register_governed_memory(conn, "other-owner", tier="M2", verified=True, evidence=evidence)
        conn.execute(
            "UPDATE memory_cognitive SET owner_id='someone-else' WHERE memory_id='other-owner'"
        )
        conn.execute("DELETE FROM memory_acl WHERE memory_id='other-owner'")
        before = conn.execute(
            "SELECT id,last_recall_ts,recall_count FROM memories ORDER BY id"
        ).fetchall()

    request = cognitive.DreamSeedRecallRequest(seed="night-dream:2026-09-02", limit=4)
    first = cognitive.recall_dream_seeds(request)
    second = cognitive.recall_dream_seeds(request)

    assert [item["id"] for item in first["items"]] == [item["id"] for item in second["items"]]
    assert {item["id"] for item in first["items"]} == {"stable-a", "stable-b"}
    assert all(item["verification_status"] == "verified" for item in first["items"])
    assert all(item["memory_tier"] in {"M1", "M2", "M3"} for item in first["items"])
    assert first["governance_pool"] == "verified_stable"
    assert first["reinforcement_eligible"] is True
    assert all(item["dream_reinforcement_eligible"] is True for item in first["items"])
    assert first["touch"] is False
    assert first["apply_dream_accessibility"] is False
    assert first["retrieval"]["acl_enforced"] is True
    with memory._conn() as conn:
        after = conn.execute(
            "SELECT id,last_recall_ts,recall_count FROM memories ORDER BY id"
        ).fetchall()
        accessibility_events = conn.execute(
            "SELECT COUNT(*) FROM memory_dream_accessibility_events"
        ).fetchone()[0]
    assert after == before
    assert accessibility_events == 0


def test_quiet_dream_seed_recall_uses_acl_filtered_legacy_pool_without_reinforcement(
    monkeypatch, tmp_path,
):
    _use_temp_db(monkeypatch, tmp_path)
    with memory._conn() as conn:
        _register_governed_memory(conn, "legacy-owned", tier="M0", verified=False)
        _register_governed_memory(conn, "legacy-other", tier="M0", verified=False)
        conn.execute(
            "UPDATE memory_cognitive SET owner_id='someone-else' WHERE memory_id='legacy-other'"
        )
        conn.execute("DELETE FROM memory_acl WHERE memory_id='legacy-other'")
        before = conn.execute(
            "SELECT id,last_recall_ts,recall_count FROM memories ORDER BY id"
        ).fetchall()

    result = cognitive.recall_dream_seeds(
        cognitive.DreamSeedRecallRequest(seed="night-dream:legacy", limit=4)
    )

    assert [item["id"] for item in result["items"]] == ["legacy-owned"]
    assert result["governance_pool"] == "legacy_unverified_dream_only"
    assert result["reinforcement_eligible"] is False
    assert result["items"][0]["dream_reinforcement_eligible"] is False
    assert result["items"][0]["verification_status"] == "unreviewed"
    with memory._conn() as conn:
        after = conn.execute(
            "SELECT id,last_recall_ts,recall_count FROM memories ORDER BY id"
        ).fetchall()
        accessibility_events = conn.execute(
            "SELECT COUNT(*) FROM memory_dream_accessibility_events"
        ).fetchone()[0]
    assert after == before
    assert accessibility_events == 0
