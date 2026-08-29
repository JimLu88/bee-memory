from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import cognitive, memory, semantic  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "chinese_business_gold_v1.json"


def _dcg(grades: list[int]) -> float:
    return sum(grade / math.log2(index + 2) for index, grade in enumerate(grades))


def _terms(text: str) -> set[str]:
    compact = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.casefold()))
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def test_curated_chinese_business_recall_and_acl(monkeypatch, tmp_path):
    """Human-curated acceptance set; unlike scale fixtures, this measures ambiguity."""
    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))
    db = tmp_path / "chinese-business-gold.sqlite"
    monkeypatch.setattr(memory, "DB_PATH", db)
    record_terms = {row["id"]: _terms(row["content"]) for row in corpus["records"]}

    def stable_semantic_channel(_conn, query, k=20, **_kwargs):
        query_terms = _terms(query)
        scored = []
        for memory_id, terms in record_terms.items():
            if not query_terms or not terms:
                continue
            score = len(query_terms & terms) / math.sqrt(len(query_terms) * len(terms))
            if score > 0:
                scored.append((memory_id, score))
        return sorted(scored, key=lambda row: row[1], reverse=True)[:k]

    monkeypatch.setattr(semantic, "vector_search", stable_semantic_channel)

    with memory._conn() as conn:
        now = cognitive._now()
        for record in corpus["records"]:
            conn.execute(
                "INSERT INTO memories(id,kind,content,created_ts,last_recall_ts,meta,token_count) VALUES (?,?,?,?,?,?,?)",
                (record["id"], record["kind"], record["content"], now, now, "{}", len(record["content"])),
            )
            cognitive.register_memory(
                conn, record["id"], record["kind"],
                {
                    "domains": record.get("domains", []),
                    "entities": record.get("entities", []),
                    "owner_id": record.get("owner_id", "jim"),
                    "memory_tier": "M1",
                    "verification_status": "verified",
                    "review_status": "active",
                    "source": "curated-gold",
                    "source_id": f"gold:{record['id']}",
                    "observed_at": "2026-08-10T00:00:00+08:00",
                    "valid_from": "2026-08-10",
                    "valid_to": "2027-08-10",
                    "evidence": [{
                        "type": "curated_fixture",
                        "locator": f"gold://{record['id']}",
                        "excerpt": record["content"],
                        "observed_at": "2026-08-10T00:00:00+08:00",
                    }],
                    "conflicts_with": ["color-conflict-new"] if record.get("conflicted") else [],
                },
                trusted_governance=True,
            )

    recalls = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    leakage: set[str] = set()
    diagnostics = []
    for case in corpus["queries"]:
        result = cognitive.recall_planned(cognitive.PlannedRecallRequest(
            query=case["query"], query_depth="Q2", limit=5,
            token_budget=1200, max_chars=1600, touch=False,
        ))
        ranked = [item["id"] for item in result["items"]]
        diagnostics.append((case["query"], ranked))
        forbidden = set(case.get("forbidden", []))
        leakage.update(forbidden & set(ranked))
        relevant = case.get("relevant", {})
        if not relevant:
            continue
        hit_ranks = [rank for rank, memory_id in enumerate(ranked, 1) if memory_id in relevant]
        recalls += int(bool(hit_ranks))
        reciprocal_ranks.append(1.0 / min(hit_ranks) if hit_ranks else 0.0)
        grades = [int(relevant.get(memory_id, 0)) for memory_id in ranked]
        ideal = sorted((int(value) for value in relevant.values()), reverse=True)[:5]
        ndcgs.append(_dcg(grades) / _dcg(ideal) if ideal else 1.0)

    measured = len([case for case in corpus["queries"] if case.get("relevant")])
    assert leakage == set(), f"ACL leakage: {sorted(leakage)}"
    assert recalls / measured >= 0.75, diagnostics
    assert sum(reciprocal_ranks) / measured >= 0.65, diagnostics
    assert sum(ndcgs) / measured >= 0.60, diagnostics
