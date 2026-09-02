"""Additive cognitive architecture for bee-memory.

The existing ``memories`` table remains authoritative.  These tables add
typed memory forms, lifecycle stages, a digital hippocampus and bounded
retrieval telemetry without moving or deleting old data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()

MEMORY_FORMS = {
    "episodic", "semantic", "procedural", "preference", "prospective",
    "social", "evidence", "hypothesis",
}
LIFECYCLE_STAGES = {"sensory", "working", "hippocampal", "consolidated", "deep"}
DEPTHS = {"L0", "L1", "L2", "L3"}
QUERY_DEPTH_ALIASES = {
    "L0": "Q0", "L1": "Q1", "L2": "Q2", "L3": "Q3",
    "Q0": "Q0", "Q1": "Q1", "Q2": "Q2", "Q3": "Q3",
}
MEMORY_TIERS = {"M0", "M1", "M2", "M3"}
GOVERNANCE_VERSION = "governance-v2"
DEFAULT_OWNER_ID = os.environ.get("BEE_DEFAULT_OWNER_ID", "jim").strip() or "jim"
RRF_K = 60
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:api[_ -]?key|access[_ -]?token|password|secret)\s*[:=]\s*\S+", re.I),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)

_RETRIEVAL_EVENTS: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=4096)
_RETRIEVAL_WORKER_LOCK = threading.Lock()
_RETRIEVAL_WORKER_STARTED = False


def _request_principal(request: Request | None) -> dict[str, Any]:
    """Return the server-authenticated principal, never a request-body claim."""
    principal = getattr(getattr(request, "state", None), "bee_principal", None)
    if isinstance(principal, dict) and principal.get("principal_id"):
        return {
            "principal_type": str(principal.get("principal_type") or "user").strip().casefold()[:40],
            "principal_id": str(principal.get("principal_id") or "").strip()[:160],
            "roles": [str(value).strip()[:160] for value in principal.get("roles", []) if str(value).strip()][:20],
            "identity_source": str(principal.get("identity_source") or "server")[:80],
        }
    # Direct Python calls in maintenance/tests do not pass a Request.  They use
    # the same server default, not caller-provided model fields.
    return {
        "principal_type": os.environ.get("BEE_DEFAULT_PRINCIPAL_TYPE", "user").strip().casefold() or "user",
        "principal_id": os.environ.get("BEE_DEFAULT_PRINCIPAL_ID", DEFAULT_OWNER_ID).strip() or DEFAULT_OWNER_ID,
        "roles": [],
        "identity_source": "server_default",
    }


def _start_retrieval_worker() -> None:
    """Persist recall strengthening and telemetry without delaying answers."""
    global _RETRIEVAL_WORKER_STARTED
    with _RETRIEVAL_WORKER_LOCK:
        if _RETRIEVAL_WORKER_STARTED:
            return
        _RETRIEVAL_WORKER_STARTED = True

    def run() -> None:
        while True:
            first = _RETRIEVAL_EVENTS.get()
            batch = [first]
            while len(batch) < 32:
                try:
                    batch.append(_RETRIEVAL_EVENTS.get_nowait())
                except queue.Empty:
                    break
            try:
                from .memory import _conn
                conn = _conn()
                try:
                    conn.execute("PRAGMA synchronous=NORMAL")
                    for event in batch:
                        chosen_ids = list(event.get("chosen_ids") or [])
                        if chosen_ids:
                            conn.executemany(
                                "UPDATE memories SET recall_count=recall_count+1,last_recall_ts=? WHERE id=?",
                                [(event["created_ts"], memory_id) for memory_id in chosen_ids],
                            )
                            conn.executemany(
                                "UPDATE memory_cognitive SET seen_count=seen_count+1,updated_ts=? WHERE memory_id=?",
                                [(event["created_ts"], memory_id) for memory_id in chosen_ids],
                            )
                        conn.execute(
                            "INSERT OR REPLACE INTO retrieval_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                            event["row"],
                        )
                        trace_rows = list(event.get("item_rows") or [])
                        if trace_rows:
                            conn.executemany(
                                "INSERT OR REPLACE INTO retrieval_run_items("
                                "run_id,memory_id,lexical_rank,vector_rank,graph_rank,rrf_score,"
                                "permission_result,filter_reason,final_rank,source_id,created_ts"
                                ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                trace_rows,
                            )
                    conn.execute(
                        "DELETE FROM retrieval_runs WHERE run_id IN "
                        "(SELECT run_id FROM retrieval_runs ORDER BY created_ts DESC LIMIT -1 OFFSET 5000)"
                    )
                    conn.execute(
                        "DELETE FROM retrieval_run_items WHERE created_ts < ?",
                        (_now() - 30 * 86400,),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                # Retrieval results are already safely returned.  A temporary
                # telemetry failure must not take the memory brain offline.
                pass
            finally:
                for _ in batch:
                    _RETRIEVAL_EVENTS.task_done()

    threading.Thread(target=run, name="bee-retrieval-writer", daemon=True).start()


def _enqueue_retrieval_event(event: dict[str, Any]) -> None:
    _start_retrieval_worker()
    try:
        _RETRIEVAL_EVENTS.put_nowait(event)
    except queue.Full:
        # Keep recall latency bounded under an abnormal event storm.  Durable
        # memories are untouched; only this activation/telemetry sample drops.
        return


def _now() -> int:
    return int(time.time())


def _json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "")
    except Exception:
        return fallback


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[,，、;；|]", value)
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name") or item.get("value") or item.get("id")
        text = str(item or "").strip()[:120]
        if text and text not in out:
            out.append(text)
    return out[:40]


def _form_for(kind: str, meta: dict[str, Any]) -> str:
    declared = str(meta.get("memory_form") or "").strip().casefold()
    if declared in MEMORY_FORMS:
        return declared
    normalized = str(kind or "semantic").casefold()
    if normalized in MEMORY_FORMS:
        return normalized
    if normalized in {"knowledge_book", "book", "knowledge"}:
        return "semantic"
    if normalized in {"decision", "lesson", "workflow", "self_upgrade"}:
        return "procedural"
    if normalized in {"dream", "idea", "inference"}:
        return "hypothesis"
    return "semantic"


def _stage_for(kind: str, meta: dict[str, Any]) -> str:
    declared = str(meta.get("lifecycle_stage") or "").strip().casefold()
    if declared in LIFECYCLE_STAGES:
        return declared
    if str(kind or "").casefold() in {"knowledge_book", "book"}:
        return "deep"
    return "consolidated"


def ensure_cognitive_schema(conn: sqlite3.Connection) -> None:
    """Create additive tables without blocking startup on legacy backfill.

    New memories are classified synchronously by :func:`register_memory`.
    Existing memories are deliberately migrated in bounded batches through
    ``/cognitive/backfill`` so a large library can never stall service startup.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_cognitive (
          memory_id TEXT PRIMARY KEY,
          memory_form TEXT NOT NULL DEFAULT 'semantic',
          lifecycle_stage TEXT NOT NULL DEFAULT 'consolidated',
          status TEXT NOT NULL DEFAULT 'active',
          confidence REAL NOT NULL DEFAULT 0.7,
          sensitivity TEXT NOT NULL DEFAULT 'internal',
          source_id TEXT NOT NULL DEFAULT '',
          source TEXT NOT NULL DEFAULT '',
          scope TEXT NOT NULL DEFAULT '',
          observed_at TEXT NOT NULL DEFAULT '',
          valid_from TEXT NOT NULL DEFAULT '',
          valid_to TEXT NOT NULL DEFAULT '',
          review_after TEXT NOT NULL DEFAULT '',
          classification_version TEXT NOT NULL DEFAULT 'cognitive-v1',
          updated_ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cognitive_form_stage
          ON memory_cognitive(memory_form,lifecycle_stage,status);
        CREATE INDEX IF NOT EXISTS idx_cognitive_source
          ON memory_cognitive(source_id,source);
        CREATE TABLE IF NOT EXISTS memory_facets (
          memory_id TEXT NOT NULL,
          facet_type TEXT NOT NULL,
          facet_value TEXT NOT NULL,
          weight REAL NOT NULL DEFAULT 1.0,
          PRIMARY KEY(memory_id,facet_type,facet_value)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_facets_lookup
          ON memory_facets(facet_type,facet_value,memory_id);
        CREATE TABLE IF NOT EXISTS memory_evidence (
          memory_id TEXT NOT NULL,
          evidence_type TEXT NOT NULL,
          locator TEXT NOT NULL,
          excerpt TEXT NOT NULL DEFAULT '',
          observed_at TEXT NOT NULL DEFAULT '',
          PRIMARY KEY(memory_id,evidence_type,locator)
        );
        CREATE TABLE IF NOT EXISTS memory_conflicts (
          conflict_id TEXT PRIMARY KEY,
          left_memory_id TEXT NOT NULL,
          right_memory_id TEXT NOT NULL,
          reason TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'open',
          created_ts INTEGER NOT NULL,
          resolved_ts INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_memory_conflicts_status
          ON memory_conflicts(status,created_ts DESC);
        CREATE TABLE IF NOT EXISTS memory_activations (
          event_id TEXT PRIMARY KEY,
          memory_id TEXT NOT NULL,
          activation_kind TEXT NOT NULL,
          session_id TEXT NOT NULL DEFAULT '',
          occurred_date TEXT NOT NULL,
          evidence_locator TEXT NOT NULL,
          principal_type TEXT NOT NULL,
          principal_id TEXT NOT NULL,
          created_ts INTEGER NOT NULL,
          UNIQUE(memory_id,activation_kind,session_id,occurred_date,evidence_locator)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_activations_gate
          ON memory_activations(memory_id,activation_kind,occurred_date,session_id);
        CREATE TABLE IF NOT EXISTS memory_dream_accessibility_events (
          event_id TEXT PRIMARY KEY,
          memory_id TEXT NOT NULL,
          dream_note_id TEXT NOT NULL,
          occurred_date TEXT NOT NULL,
          phase_count INTEGER NOT NULL,
          segment_count INTEGER NOT NULL,
          base_delta REAL NOT NULL,
          evidence_locator TEXT NOT NULL,
          principal_type TEXT NOT NULL,
          principal_id TEXT NOT NULL,
          created_ts INTEGER NOT NULL,
          UNIQUE(memory_id,dream_note_id)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_dream_accessibility
          ON memory_dream_accessibility_events(memory_id,occurred_date);
        CREATE TABLE IF NOT EXISTS hippocampal_inbox (
          episode_id TEXT PRIMARY KEY,
          idempotency_key TEXT NOT NULL UNIQUE,
          source_id TEXT NOT NULL DEFAULT '',
          summary TEXT NOT NULL,
          evidence_json TEXT NOT NULL DEFAULT '[]',
          memory_form TEXT NOT NULL DEFAULT 'episodic',
          domains_json TEXT NOT NULL DEFAULT '[]',
          entities_json TEXT NOT NULL DEFAULT '[]',
          importance INTEGER NOT NULL DEFAULT 2,
          confidence REAL NOT NULL DEFAULT 0.7,
          sensitivity TEXT NOT NULL DEFAULT 'internal',
          observed_at TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'staged',
          memory_id TEXT NOT NULL DEFAULT '',
          created_ts INTEGER NOT NULL,
          updated_ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_hippocampal_status
          ON hippocampal_inbox(status,updated_ts DESC);
        CREATE TABLE IF NOT EXISTS retrieval_runs (
          run_id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL DEFAULT '',
          query_hash TEXT NOT NULL,
          depth TEXT NOT NULL,
          candidate_count INTEGER NOT NULL,
          result_count INTEGER NOT NULL,
          token_count INTEGER NOT NULL,
          latency_ms REAL NOT NULL,
          source_counts_json TEXT NOT NULL DEFAULT '{}',
          created_ts INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_retrieval_runs_created
          ON retrieval_runs(created_ts DESC);
        CREATE TABLE IF NOT EXISTS cognitive_migrations (
          name TEXT PRIMARY KEY,
          completed_ts INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_lineage (
          child_memory_id TEXT NOT NULL,
          parent_memory_id TEXT NOT NULL,
          relation_type TEXT NOT NULL,
          source_locator TEXT NOT NULL DEFAULT '',
          created_ts INTEGER NOT NULL,
          PRIMARY KEY(child_memory_id,parent_memory_id,relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_lineage_parent
          ON memory_lineage(parent_memory_id,relation_type,child_memory_id);
        CREATE TABLE IF NOT EXISTS memory_acl (
          memory_id TEXT NOT NULL,
          principal_type TEXT NOT NULL,
          principal_id TEXT NOT NULL,
          permission TEXT NOT NULL DEFAULT 'read',
          expires_at INTEGER,
          PRIMARY KEY(memory_id,principal_type,principal_id,permission)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_acl_principal
          ON memory_acl(principal_type,principal_id,permission,memory_id);
        CREATE TABLE IF NOT EXISTS memory_promotion_log (
          promotion_id TEXT PRIMARY KEY,
          memory_id TEXT NOT NULL,
          from_tier TEXT NOT NULL,
          to_tier TEXT NOT NULL,
          evidence_json TEXT NOT NULL DEFAULT '[]',
          rule_version TEXT NOT NULL,
          run_id TEXT NOT NULL DEFAULT '',
          reviewer TEXT NOT NULL DEFAULT '',
          created_ts INTEGER NOT NULL,
          UNIQUE(memory_id,to_tier,rule_version,run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_promotion_memory
          ON memory_promotion_log(memory_id,created_ts DESC);
        CREATE TABLE IF NOT EXISTS shared_state_versions (
          scope_id TEXT PRIMARY KEY,
          version INTEGER NOT NULL DEFAULT 1,
          content_hash TEXT NOT NULL DEFAULT '',
          updated_ts INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS retrieval_run_items (
          run_id TEXT NOT NULL,
          memory_id TEXT NOT NULL,
          lexical_rank INTEGER,
          vector_rank INTEGER,
          graph_rank INTEGER,
          rrf_score REAL NOT NULL DEFAULT 0,
          permission_result TEXT NOT NULL DEFAULT 'allowed',
          filter_reason TEXT NOT NULL DEFAULT '',
          final_rank INTEGER,
          source_id TEXT NOT NULL DEFAULT '',
          created_ts INTEGER NOT NULL,
          PRIMARY KEY(run_id,memory_id)
        );
        CREATE INDEX IF NOT EXISTS idx_retrieval_run_items_created
          ON retrieval_run_items(created_ts DESC);
        """
    )
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(memory_cognitive)")}
    additive_columns = {
        "memory_tier": "TEXT NOT NULL DEFAULT 'M0'",
        "verification_status": "TEXT NOT NULL DEFAULT 'unreviewed'",
        "review_status": "TEXT NOT NULL DEFAULT 'quarantine'",
        "owner_type": "TEXT NOT NULL DEFAULT 'user'",
        "owner_id": "TEXT NOT NULL DEFAULT 'jim'",
        "visibility": "TEXT NOT NULL DEFAULT 'private'",
        "acl_json": "TEXT NOT NULL DEFAULT '[]'",
        "parent_memory_id": "TEXT NOT NULL DEFAULT ''",
        "root_source_id": "TEXT NOT NULL DEFAULT ''",
        "supersedes_memory_id": "TEXT NOT NULL DEFAULT ''",
        "evidence_count": "INTEGER NOT NULL DEFAULT 0",
        "activation_count": "INTEGER NOT NULL DEFAULT 0",
        "seen_count": "INTEGER NOT NULL DEFAULT 0",
        "adopted_count": "INTEGER NOT NULL DEFAULT 0",
        "confirmed_count": "INTEGER NOT NULL DEFAULT 0",
        "last_verified_ts": "INTEGER NOT NULL DEFAULT 0",
        "governance_version": "TEXT NOT NULL DEFAULT 'governance-v1'",
    }
    for column, declaration in additive_columns.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE memory_cognitive ADD COLUMN {column} {declaration}")
    conflict_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(memory_conflicts)")}
    for column, declaration in {
        "resolution": "TEXT NOT NULL DEFAULT ''",
        "resolved_by": "TEXT NOT NULL DEFAULT ''",
        "winning_memory_id": "TEXT NOT NULL DEFAULT ''",
        "superseding_memory_id": "TEXT NOT NULL DEFAULT ''",
    }.items():
        if column not in conflict_columns:
            conn.execute(f"ALTER TABLE memory_conflicts ADD COLUMN {column} {declaration}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cognitive_acl "
        "ON memory_cognitive(visibility,owner_type,owner_id,review_status,memory_tier)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO cognitive_migrations(name,completed_ts) VALUES (?,?)",
        ("lazy_existing_v1", _now()),
    )


def _sync_hot_fts_memory(conn: sqlite3.Connection, memory_id: str) -> None:
    """Keep the bounded conversational index aligned with lifecycle state."""
    conn.execute("DELETE FROM memories_hot_fts WHERE memory_id=?", (memory_id,))
    row = conn.execute(
        """SELECT m.content
           FROM memories AS m
           JOIN memory_cognitive AS c ON c.memory_id=m.id
           WHERE m.id=? AND m.invalid_at IS NULL
             AND c.status='active' AND c.lifecycle_stage!='deep'
             AND c.review_status!='retired'""",
        (memory_id,),
    ).fetchone()
    if row:
        conn.execute(
            "INSERT INTO memories_hot_fts(memory_id,content) VALUES (?,?)",
            (memory_id, str(row[0] or "")),
        )


def rebuild_hot_fts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Rebuild the compact Q1/Q2 index deterministically from source rows."""
    started = time.perf_counter()
    conn.execute("DELETE FROM memories_hot_fts")
    conn.execute(
        """INSERT INTO memories_hot_fts(memory_id,content)
           SELECT m.id,m.content
           FROM memories AS m
           JOIN memory_cognitive AS c ON c.memory_id=m.id
           WHERE m.invalid_at IS NULL AND c.status='active'
             AND c.lifecycle_stage!='deep' AND c.review_status!='retired'"""
    )
    count = int(conn.execute("SELECT COUNT(*) FROM memories_hot_fts").fetchone()[0])
    return {"ok": True, "rows": count,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def _valid_evidence_count(conn: sqlite3.Connection, memory_id: str) -> int:
    return int(conn.execute(
        """SELECT COUNT(*) FROM memory_evidence
           WHERE memory_id=? AND TRIM(locator)!='' AND TRIM(excerpt)!=''
             AND TRIM(observed_at)!=''""",
        (memory_id,),
    ).fetchone()[0])


def register_memory(conn: sqlite3.Connection, memory_id: str, kind: str,
                    meta: dict[str, Any] | None, *,
                    principal: dict[str, Any] | None = None,
                    trusted_governance: bool = False) -> None:
    """Attach typed metadata while forcing ordinary writes into M0 quarantine.

    ``trusted_governance`` exists only for internal migrations/tests.  HTTP
    store routes never set it; M1-M3 are produced exclusively by governance
    endpoints after evidence and lifecycle gates pass.
    """
    meta = dict(meta or {})
    caller = principal or _request_principal(None)
    now = _now()
    owner_type = str(caller.get("principal_type") or "user").strip().casefold()[:40]
    owner_id = str(caller.get("principal_id") or DEFAULT_OWNER_ID).strip()[:160] or DEFAULT_OWNER_ID
    if trusted_governance:
        owner_type = str(meta.get("owner_type") or owner_type).strip().casefold()[:40]
        owner_id = str(meta.get("owner_id") or owner_id).strip()[:160] or owner_id
    source_id = str(meta.get("source_id") or "").strip()[:300]
    source = str(meta.get("source") or meta.get("source_type") or "").strip()[:120]
    observed_at = str(meta.get("observed_at") or meta.get("happened_at") or "").strip()[:80]
    valid_from = str(meta.get("valid_from") or "").strip()[:40]
    valid_to = str(meta.get("valid_to") or "").strip()[:40]
    conn.execute(
        """
        INSERT INTO memory_cognitive(
          memory_id,memory_form,lifecycle_stage,status,confidence,sensitivity,
          source_id,source,scope,observed_at,valid_from,valid_to,review_after,
          classification_version,updated_ts
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(memory_id) DO UPDATE SET
          memory_form=excluded.memory_form,lifecycle_stage=excluded.lifecycle_stage,
          status=excluded.status,confidence=excluded.confidence,
          sensitivity=excluded.sensitivity,source_id=excluded.source_id,
          source=excluded.source,scope=excluded.scope,
          observed_at=excluded.observed_at,valid_from=excluded.valid_from,
          valid_to=excluded.valid_to,review_after=excluded.review_after,
          classification_version=excluded.classification_version,
          updated_ts=excluded.updated_ts
        """,
        (
            memory_id, _form_for(kind, meta), _stage_for(kind, meta),
            str(meta.get("status") or "active")[:40],
            max(0.0, min(1.0, float(meta.get("confidence", 0.7) or 0.7))),
            str(meta.get("sensitivity") or "internal")[:40],
            source_id, source,
            str(meta.get("scope") or meta.get("project") or meta.get("mode_id") or "")[:200],
            observed_at, valid_from, valid_to,
            str(meta.get("review_after") or "")[:40],
            str(meta.get("classification_version") or "cognitive-v2")[:80], now,
        ),
    )
    raw_evidence = meta.get("evidence") or []
    if isinstance(raw_evidence, dict):
        raw_evidence = [raw_evidence]
    evidence_rows = [row for row in raw_evidence if isinstance(row, dict)]
    for row in evidence_rows:
        locator = str(row.get("locator") or row.get("url") or row.get("source_ref") or "").strip()[:1000]
        if locator:
            conn.execute(
                "INSERT OR REPLACE INTO memory_evidence(memory_id,evidence_type,locator,excerpt,observed_at) VALUES (?,?,?,?,?)",
                (memory_id, str(row.get("type") or "source")[:80], locator,
                 str(row.get("excerpt") or row.get("statement") or "")[:2000],
                 str(row.get("observed_at") or "")[:80]),
            )

    verification = "unreviewed"
    memory_tier = "M0"
    review_status = "quarantine"
    visibility = "private"
    parent_memory_id = ""
    supersedes_memory_id = ""
    activation_count = 0
    if trusted_governance:
        requested_verification = str(meta.get("verification_status") or "").strip().casefold()
        if requested_verification in {"unreviewed", "verified", "rejected", "conflicted"}:
            verification = requested_verification
        elif bool(meta.get("verified")):
            verification = "verified"
        requested_tier = str(meta.get("memory_tier") or "M0").strip().upper()
        memory_tier = requested_tier if requested_tier in MEMORY_TIERS else "M0"
        requested_review = str(meta.get("review_status") or ("active" if verification == "verified" else "quarantine")).strip().casefold()
        review_status = requested_review if requested_review in {"quarantine", "active", "needs_review", "retired"} else "quarantine"
        requested_visibility = str(meta.get("visibility") or "private").strip().casefold()
        visibility = requested_visibility if requested_visibility in {"private", "team", "restricted", "agent"} else "private"
        parent_memory_id = str(meta.get("parent_memory_id") or "").strip()[:160]
        supersedes_memory_id = str(meta.get("supersedes_memory_id") or "").strip()[:160]
        activation_count = max(0, int(meta.get("activation_count") or 0))
    root_source_id = str(meta.get("root_source_id") or source_id or memory_id).strip()[:300]
    evidence_count = _valid_evidence_count(conn, memory_id)
    conn.execute(
        """
        UPDATE memory_cognitive SET memory_tier=?,verification_status=?,review_status=?,
          owner_type=?,owner_id=?,visibility=?,acl_json='[]',parent_memory_id=?,
          root_source_id=?,supersedes_memory_id=?,evidence_count=?,activation_count=?,
          last_verified_ts=?,governance_version='governance-v2',updated_ts=?
        WHERE memory_id=?
        """,
        (memory_tier, verification, review_status, owner_type, owner_id, visibility,
         parent_memory_id, root_source_id, supersedes_memory_id, evidence_count,
         activation_count, now if verification == "verified" else 0, now, memory_id),
    )
    # Every governed row is private and readable only by its authenticated
    # owner unless a later governance action adds an explicit ACL grant.
    conn.execute(
        "INSERT OR REPLACE INTO memory_acl(memory_id,principal_type,principal_id,permission,expires_at) VALUES (?,?,?,?,NULL)",
        (memory_id, owner_type, owner_id, "admin"),
    )
    if trusted_governance:
        acl = meta.get("acl") or []
        if isinstance(acl, dict):
            acl = [acl]
        for entry in acl if isinstance(acl, list) else []:
            if not isinstance(entry, dict):
                continue
            entry_type = str(entry.get("type") or entry.get("principal_type") or "user").strip().casefold()[:40]
            entry_id = str(entry.get("id") or entry.get("principal_id") or "").strip()[:160]
            permission = str(entry.get("permission") or "read").strip().casefold()[:40]
            if entry_id and permission in {"read", "write", "admin"}:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_acl(memory_id,principal_type,principal_id,permission,expires_at) VALUES (?,?,?,?,?)",
                    (memory_id, entry_type, entry_id, permission, entry.get("expires_at")),
                )
    if parent_memory_id:
        conn.execute(
            "INSERT OR IGNORE INTO memory_lineage(child_memory_id,parent_memory_id,relation_type,source_locator,created_ts) VALUES (?,?,?,?,?)",
            (memory_id, parent_memory_id, str(meta.get("parent_relation") or "derived_from")[:40], source_id, now),
        )
    if supersedes_memory_id:
        conn.execute(
            "INSERT OR IGNORE INTO memory_lineage(child_memory_id,parent_memory_id,relation_type,source_locator,created_ts) VALUES (?,?,?,?,?)",
            (memory_id, supersedes_memory_id, "supersedes", source_id, now),
        )
    mappings = {
        "domain": _list(meta.get("domains") or meta.get("domain")),
        "entity": _list(meta.get("entities")),
        "tag": _list(meta.get("tags")),
        "project": _list(meta.get("project") or meta.get("scope")),
    }
    facets = [
        (memory_id, facet_type, value.casefold(), 1.0)
        for facet_type, values in mappings.items() for value in values
    ]
    if facets:
        conn.executemany(
            "INSERT OR REPLACE INTO memory_facets(memory_id,facet_type,facet_value,weight) VALUES (?,?,?,?)",
            facets,
        )
    # Conflicts remain explicit and unresolved until a later evidence-backed
    # decision.  A new memory never silently overwrites an older fact.
    for other_id in _list(meta.get("conflicts_with")):
        if other_id == memory_id:
            continue
        seed = "|".join(sorted((memory_id, other_id)))
        conflict_id = "conf-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        conn.execute(
            "INSERT OR IGNORE INTO memory_conflicts(conflict_id,left_memory_id,right_memory_id,reason,status,created_ts) VALUES (?,?,?,?,?,?)",
            (conflict_id, other_id, memory_id,
             str(meta.get("conflict_reason") or "source statements disagree")[:1000],
             "open", now),
        )
    _sync_hot_fts_memory(conn, memory_id)


class HippocampalStageRequest(BaseModel):
    summary: str = Field(min_length=1, max_length=6000)
    idempotency_key: str = Field(min_length=3, max_length=300)
    source_id: str = ""
    memory_form: str = "episodic"
    domains: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    importance: int = Field(default=2, ge=0, le=5)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    sensitivity: str = "internal"
    observed_at: str = ""


class HippocampalPromoteRequest(BaseModel):
    episode_id: str = ""
    source_id: str = ""
    memory_id: str = ""
    status: str = "promoted"


class CognitiveBackfillRequest(BaseModel):
    limit: int = Field(default=500, ge=1, le=5000)


class HotFtsRebuildRequest(BaseModel):
    confirm: bool = False


class GovernedPromotionRequest(BaseModel):
    memory_id: str = Field(min_length=1, max_length=160)
    to_tier: str = Field(default="M1", min_length=2, max_length=2)
    parent_memory_ids: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    run_id: str = ""
    reviewer: str = "night-sync"
    scope_id: str = "global"
    expected_view_version: int = Field(default=0, ge=0)
    rule_version: str = "governance-v2"
    source: str = ""
    source_id: str = ""
    observed_at: str = ""
    valid_from: str = ""
    valid_to: str = ""


class ActivationRequest(BaseModel):
    memory_id: str = Field(min_length=1, max_length=160)
    activation_kind: str = "adopted"
    session_id: str = ""
    occurred_at: str = ""
    evidence_locator: str = Field(min_length=1, max_length=1000)
    idempotency_key: str = ""


class DreamAccessibilityMention(BaseModel):
    memory_id: str = Field(min_length=1, max_length=160)
    phase_count: int = Field(default=1, ge=1, le=3)
    segment_count: int = Field(default=1, ge=1, le=3)


class DreamAccessibilityRequest(BaseModel):
    dream_note_id: str = Field(min_length=1, max_length=160)
    dream_date: str = Field(min_length=10, max_length=10)
    mentions: list[DreamAccessibilityMention] = Field(default_factory=list)


class DreamSeedRecallRequest(BaseModel):
    seed: str = Field(min_length=1, max_length=160)
    limit: int = Field(default=4, ge=1, le=8)
    max_chars: int = Field(default=2800, ge=400, le=4000)


class ConflictReviewRequest(BaseModel):
    conflict_id: str = Field(min_length=1, max_length=160)
    resolution: str = "needs_more_evidence"
    winning_memory_id: str = ""
    evidence_locator: str = ""
    notes: str = ""


class GovernedSupersedeRequest(BaseModel):
    old_memory_id: str = Field(min_length=1, max_length=160)
    new_memory_id: str = Field(min_length=1, max_length=160)
    evidence_locator: str = Field(min_length=1, max_length=1000)
    scope_id: str = "global"
    expected_view_version: int = Field(default=0, ge=0)


class NightlyGovernanceRequest(BaseModel):
    run_id: str = ""
    limit: int = Field(default=100, ge=1, le=1000)
    scope_id: str = "global"


@router.post("/governance/promote")
def promote_governed_memory(req: GovernedPromotionRequest,
                            request: Request = None) -> dict[str, Any]:
    """Promote one memory only when deterministic evidence gates pass."""
    from .memory import _conn

    to_tier = req.to_tier.strip().upper()
    if to_tier not in MEMORY_TIERS - {"M0"}:
        raise HTTPException(400, "to_tier must be M1, M2 or M3")
    parents = list(dict.fromkeys(str(value).strip()[:160] for value in req.parent_memory_ids if str(value).strip()))[:32]
    valid_evidence = [value for value in req.evidence if isinstance(value, dict)
                      and str(value.get("locator") or value.get("url") or value.get("source_ref") or "").strip()
                      and str(value.get("excerpt") or value.get("statement") or "").strip()
                      and str(value.get("observed_at") or "").strip()]
    principal = _request_principal(request)
    now = _now()
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM memory_cognitive WHERE memory_id=?", (req.memory_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "memory governance row not found")
        current = dict(row)
        state = conn.execute(
            "SELECT version FROM shared_state_versions WHERE scope_id=?", (req.scope_id,)
        ).fetchone()
        current_version = int(state[0]) if state else 1
        if req.expected_view_version and req.expected_view_version != current_version:
            raise HTTPException(409, f"stale shared state: expected {req.expected_view_version}, current {current_version}")
        if req.source:
            current["source"] = req.source.strip()[:120]
        if req.source_id:
            current["source_id"] = req.source_id.strip()[:300]
        if req.observed_at:
            current["observed_at"] = req.observed_at.strip()[:80]
        if req.valid_from:
            current["valid_from"] = req.valid_from.strip()[:40]
        if req.valid_to:
            current["valid_to"] = req.valid_to.strip()[:40]
        conn.execute(
            """UPDATE memory_cognitive SET source=?,source_id=?,observed_at=?,valid_from=?,valid_to=?,updated_ts=?
               WHERE memory_id=?""",
            (str(current.get("source") or ""), str(current.get("source_id") or ""),
             str(current.get("observed_at") or ""), str(current.get("valid_from") or ""),
             str(current.get("valid_to") or ""), now, req.memory_id),
        )
        for evidence in valid_evidence:
            locator = str(evidence.get("locator") or evidence.get("url") or evidence.get("source_ref") or "")[:1000]
            conn.execute(
                "INSERT OR REPLACE INTO memory_evidence(memory_id,evidence_type,locator,excerpt,observed_at) VALUES (?,?,?,?,?)",
                (req.memory_id, str(evidence.get("type") or "source")[:80], locator,
                 str(evidence.get("excerpt") or evidence.get("statement") or "")[:2000],
                 str(evidence.get("observed_at") or "")[:80]),
            )
        evidence_count = _valid_evidence_count(conn, req.memory_id)
        conn.execute("UPDATE memory_cognitive SET evidence_count=? WHERE memory_id=?",
                     (evidence_count, req.memory_id))
        conflict_count = conn.execute(
            "SELECT COUNT(*) FROM memory_conflicts WHERE status='open' AND (left_memory_id=? OR right_memory_id=?)",
            (req.memory_id, req.memory_id),
        ).fetchone()[0]
        parent_rows: list[dict[str, Any]] = []
        if parents:
            placeholders = ",".join("?" for _ in parents)
            parent_rows = [dict(value) for value in conn.execute(
                f"SELECT * FROM memory_cognitive WHERE memory_id IN ({placeholders})",
                parents,
            )]
        reasons: list[str] = []
        if conflict_count:
            reasons.append("open_conflict")
        if to_tier == "M1":
            for field in ("source", "source_id", "observed_at", "valid_from", "valid_to"):
                if not str(current.get(field) or "").strip():
                    reasons.append(f"{field}_required")
            if evidence_count < 1:
                reasons.append("complete_evidence_required")
        elif to_tier == "M2":
            if str(current.get("memory_tier") or "M0") != "M1" or str(current.get("verification_status") or "") != "verified":
                reasons.append("current_memory_must_be_verified_M1")
            if len(parent_rows) < 2 or any(str(value.get("memory_tier") or "M0") not in {"M1", "M2", "M3"} for value in parent_rows):
                reasons.append("two_verified_M1_parents_required")
            if any(str(value.get("verification_status") or "unreviewed") != "verified" for value in parent_rows):
                reasons.append("parent_verification_required")
        elif to_tier == "M3":
            if str(current.get("memory_tier") or "M0") != "M2":
                reasons.append("current_memory_must_be_M2")
            activation = conn.execute(
                """SELECT COUNT(*),COUNT(DISTINCT occurred_date),
                          COUNT(DISTINCT CASE WHEN TRIM(session_id)!='' THEN session_id END)
                   FROM memory_activations
                   WHERE memory_id=? AND activation_kind IN ('adopted','confirmed')""",
                (req.memory_id,),
            ).fetchone()
            activation_count = int(activation[0] or 0)
            distinct_dates = int(activation[1] or 0)
            distinct_sessions = int(activation[2] or 0)
            if activation_count < 3 or (distinct_dates < 2 and distinct_sessions < 2):
                reasons.append("independent_adoption_required")
            if evidence_count < 2:
                reasons.append("two_evidence_items_required")
        if reasons:
            return {"ok": False, "promoted": False, "memory_id": req.memory_id,
                    "current_tier": current.get("memory_tier") or "M0", "reasons": reasons,
                    "shared_state_version": current_version}
        from_tier = str(current.get("memory_tier") or "M0")
        if int(to_tier[-1]) < int(from_tier[-1]):
            raise HTTPException(400, "governance promotion cannot move a memory backwards")
        conn.execute(
            "UPDATE memory_cognitive SET memory_tier=?,verification_status='verified',review_status='active',"
            "evidence_count=?,last_verified_ts=?,governance_version=?,updated_ts=? WHERE memory_id=?",
            (to_tier, evidence_count, now, req.rule_version[:80], now, req.memory_id),
        )
        for parent in parents:
            conn.execute(
                "INSERT OR IGNORE INTO memory_lineage(child_memory_id,parent_memory_id,relation_type,source_locator,created_ts) VALUES (?,?,?,?,?)",
                (req.memory_id, parent, "summarizes" if to_tier == "M2" else "derived_from", req.run_id[:300], now),
            )
        promotion_seed = "|".join([req.memory_id, to_tier, req.rule_version, req.run_id])
        promotion_id = "prom-" + hashlib.sha256(promotion_seed.encode("utf-8")).hexdigest()[:20]
        conn.execute(
            "INSERT OR IGNORE INTO memory_promotion_log(promotion_id,memory_id,from_tier,to_tier,evidence_json,rule_version,run_id,reviewer,created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (promotion_id, req.memory_id, from_tier, to_tier,
             json.dumps(valid_evidence, ensure_ascii=False), req.rule_version[:80],
             req.run_id[:160], f"{principal['principal_type']}:{principal['principal_id']}"[:120], now),
        )
        new_version = current_version + 1
        conn.execute(
            "INSERT INTO shared_state_versions(scope_id,version,content_hash,updated_ts) VALUES (?,?,?,?) "
            "ON CONFLICT(scope_id) DO UPDATE SET version=excluded.version,content_hash=excluded.content_hash,updated_ts=excluded.updated_ts",
            (req.scope_id[:160], new_version, promotion_id, now),
        )
    return {"ok": True, "promoted": True, "memory_id": req.memory_id,
            "from_tier": from_tier, "to_tier": to_tier,
            "promotion_id": promotion_id, "shared_state_version": new_version}


@router.post("/governance/activate")
def record_governed_activation(req: ActivationRequest,
                               request: Request = None) -> dict[str, Any]:
    """Record adoption/confirmation; retrieval visibility is only ``seen``."""
    from .memory import _conn

    kind = req.activation_kind.strip().casefold()
    if kind not in {"adopted", "confirmed"}:
        raise HTTPException(400, "activation_kind must be adopted or confirmed")
    locator = req.evidence_locator.strip()[:1000]
    occurred = (req.occurred_at.strip() or time.strftime("%Y-%m-%d", time.localtime()))[:40]
    occurred_date = occurred[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", occurred_date):
        raise HTTPException(400, "occurred_at must begin with YYYY-MM-DD")
    session_id = req.session_id.strip()[:200]
    principal = _request_principal(request)
    seed = req.idempotency_key.strip() or "|".join(
        [req.memory_id, kind, session_id, occurred_date, locator,
         principal["principal_type"], principal["principal_id"]]
    )
    event_id = "act-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    now = _now()
    with _conn() as conn:
        if not conn.execute("SELECT 1 FROM memory_cognitive WHERE memory_id=?", (req.memory_id,)).fetchone():
            raise HTTPException(404, "memory governance row not found")
        before = conn.total_changes
        conn.execute(
            """INSERT OR IGNORE INTO memory_activations(
                 event_id,memory_id,activation_kind,session_id,occurred_date,
                 evidence_locator,principal_type,principal_id,created_ts
               ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (event_id, req.memory_id, kind, session_id, occurred_date, locator,
             principal["principal_type"], principal["principal_id"], now),
        )
        inserted = conn.total_changes > before
        counts = conn.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN activation_kind='adopted' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN activation_kind='confirmed' THEN 1 ELSE 0 END),
                      COUNT(DISTINCT occurred_date),
                      COUNT(DISTINCT CASE WHEN TRIM(session_id)!='' THEN session_id END)
               FROM memory_activations WHERE memory_id=?""",
            (req.memory_id,),
        ).fetchone()
        conn.execute(
            """UPDATE memory_cognitive SET activation_count=?,adopted_count=?,confirmed_count=?,updated_ts=?
               WHERE memory_id=?""",
            (int(counts[0] or 0), int(counts[1] or 0), int(counts[2] or 0), now, req.memory_id),
        )
    return {
        "ok": True, "event_id": event_id, "deduplicated": not inserted,
        "memory_id": req.memory_id, "activation_kind": kind,
        "activation_count": int(counts[0] or 0),
        "distinct_dates": int(counts[3] or 0),
        "distinct_sessions": int(counts[4] or 0),
    }


_DREAM_ACCESSIBILITY_DELTA = {1: 0.020, 2: 0.028, 3: 0.035}
_DREAM_ACCESSIBILITY_HALF_LIFE_DAYS = 30.0
_DREAM_ACCESSIBILITY_CAP = 0.12


@router.post("/recall-dream-seeds")
def recall_dream_seeds(req: DreamSeedRecallRequest,
                       request: Request = None) -> dict[str, Any]:
    """Return a deterministic random sample of stable memories for a quiet night.

    Candidate IDs are filtered by governance and ACL before content is fetched.
    The endpoint is deliberately read-only: it neither records recall nor applies
    prior dream accessibility while selecting the next dream's source material.
    """
    from .memory import _conn

    principal = _request_principal(request)
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        candidate_ids = {
            str(row[0]) for row in conn.execute(
                """SELECT c.memory_id
                   FROM memory_cognitive AS c
                   JOIN memories AS m ON m.id=c.memory_id
                   WHERE m.invalid_at IS NULL
                     AND c.memory_tier IN ('M1','M2','M3')
                     AND c.verification_status='verified'
                     AND c.review_status='active'
                     AND c.status='active'
                     AND c.memory_form!='hypothesis'
                   ORDER BY COALESCE(m.last_recall_ts,0),c.memory_id
                   LIMIT 512"""
            ).fetchall()
        }
        allowed = _allowed_memory_ids(
            conn,
            principal_type=principal["principal_type"],
            principal_id=principal["principal_id"],
            roles=principal["roles"],
            candidate_ids=candidate_ids,
        )
        if not allowed:
            return {
                "ok": True, "items": [], "candidate_count": 0,
                "selection": "deterministic_seeded_stable_memory",
                "touch": False, "apply_dream_accessibility": False,
            }
        placeholders = ",".join("?" for _ in allowed)
        rows = [dict(row) for row in conn.execute(
            f"""SELECT m.id,m.kind,m.content,m.importance,m.meta,
                       c.memory_form,c.lifecycle_stage,c.memory_tier,
                       c.verification_status,c.review_status,c.source,c.source_id
                FROM memories AS m
                JOIN memory_cognitive AS c ON c.memory_id=m.id
                WHERE m.id IN ({placeholders})""",
            sorted(allowed),
        ).fetchall()]

    rows.sort(key=lambda row: hashlib.sha256(
        f"{req.seed}|{row['id']}".encode("utf-8")
    ).hexdigest())
    items: list[dict[str, Any]] = []
    used_chars = 0
    for row in rows:
        remaining = req.max_chars - used_chars
        if remaining <= 0 or len(items) >= req.limit:
            break
        snippet = str(row.get("content") or "").strip()[: min(700, remaining)]
        if not snippet:
            continue
        items.append({
            "id": str(row["id"]),
            "memory_id": str(row["id"]),
            "kind": str(row.get("kind") or ""),
            "snippet": snippet,
            "content": snippet,
            "importance": int(row.get("importance") or 0),
            "memory_form": str(row.get("memory_form") or ""),
            "lifecycle_stage": str(row.get("lifecycle_stage") or ""),
            "memory_tier": str(row.get("memory_tier") or "M0"),
            "verification_status": str(row.get("verification_status") or "unreviewed"),
            "review_status": str(row.get("review_status") or "quarantine"),
            "source": str(row.get("source") or ""),
            "source_id": str(row.get("source_id") or ""),
            "dream_accessibility_boost": 0.0,
        })
        used_chars += len(snippet)
    return {
        "ok": True,
        "items": items,
        "candidate_count": len(allowed),
        "char_count": used_chars,
        "selection": "deterministic_seeded_stable_memory",
        "touch": False,
        "apply_dream_accessibility": False,
        "retrieval": {
            "principal_type": principal["principal_type"],
            "principal_id": principal["principal_id"],
            "identity_source": principal["identity_source"],
            "acl_enforced": True,
        },
    }


@router.post("/governance/dream-accessibility")
def record_dream_accessibility(req: DreamAccessibilityRequest,
                               request: Request = None) -> dict[str, Any]:
    """Record bounded dream salience without changing truth or promotion state."""
    from .memory import _conn

    try:
        occurred_date = date.fromisoformat(req.dream_date).isoformat()
    except ValueError as exc:
        raise HTTPException(400, "dream_date must use YYYY-MM-DD") from exc
    if len(req.mentions) > 16:
        raise HTTPException(400, "at most 16 dream mentions are accepted")
    principal = _request_principal(request)
    now = _now()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    with _conn() as conn:
        candidate_ids = {item.memory_id for item in req.mentions}
        allowed = _allowed_memory_ids(
            conn,
            principal_type=principal["principal_type"],
            principal_id=principal["principal_id"],
            roles=principal["roles"],
            candidate_ids=candidate_ids,
        )
        for item in req.mentions:
            row = conn.execute(
                "SELECT memory_tier,verification_status,review_status,status "
                "FROM memory_cognitive WHERE memory_id=?",
                (item.memory_id,),
            ).fetchone()
            reason = ""
            if item.memory_id not in allowed:
                reason = "acl_denied"
            elif not row:
                reason = "governance_row_missing"
            elif str(row[0] or "M0") not in {"M1", "M2", "M3"}:
                reason = "stable_tier_required"
            elif str(row[1] or "unreviewed") != "verified":
                reason = "verified_memory_required"
            elif str(row[2] or "quarantine") != "active" or str(row[3] or "active") != "active":
                reason = "active_memory_required"
            if reason:
                rejected.append({"memory_id": item.memory_id, "reason": reason})
                continue
            delta = _DREAM_ACCESSIBILITY_DELTA[item.phase_count]
            seed = f"{req.dream_note_id}|{item.memory_id}"
            event_id = "dream-access-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO memory_dream_accessibility_events(
                     event_id,memory_id,dream_note_id,occurred_date,phase_count,
                     segment_count,base_delta,evidence_locator,principal_type,
                     principal_id,created_ts
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event_id, item.memory_id, req.dream_note_id, occurred_date,
                    item.phase_count, item.segment_count, delta,
                    f"dream:{req.dream_note_id}", principal["principal_type"],
                    principal["principal_id"], now,
                ),
            )
            accepted.append({
                "memory_id": item.memory_id,
                "event_id": event_id,
                "base_delta": delta,
                "deduplicated": conn.total_changes == before,
            })
    return {
        "ok": True,
        "dream_note_id": req.dream_note_id,
        "dream_date": occurred_date,
        "accepted": accepted,
        "rejected": rejected,
        "policy": {
            "half_life_days": int(_DREAM_ACCESSIBILITY_HALF_LIFE_DAYS),
            "cumulative_cap": _DREAM_ACCESSIBILITY_CAP,
            "changes_truth_state": False,
            "counts_for_promotion": False,
        },
    }


@router.post("/governance/conflicts/review")
def review_memory_conflict(req: ConflictReviewRequest,
                           request: Request = None) -> dict[str, Any]:
    from .memory import _conn

    resolution = req.resolution.strip().casefold()
    if resolution not in {"resolved", "dismissed", "needs_more_evidence"}:
        raise HTTPException(400, "unsupported conflict resolution")
    if resolution == "resolved" and (not req.winning_memory_id.strip() or not req.evidence_locator.strip()):
        raise HTTPException(400, "resolved conflicts require winner and evidence_locator")
    principal = _request_principal(request)
    now = _now()
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM memory_conflicts WHERE conflict_id=?", (req.conflict_id,)).fetchone()
        if not row:
            raise HTTPException(404, "memory conflict not found")
        if str(row["status"]) != "open" and resolution != "needs_more_evidence":
            raise HTTPException(409, "memory conflict is already closed")
        status_value = "open" if resolution == "needs_more_evidence" else resolution
        conn.execute(
            """UPDATE memory_conflicts SET status=?,resolution=?,resolved_by=?,
                      winning_memory_id=?,resolved_ts=? WHERE conflict_id=?""",
            (status_value, (req.notes.strip() or resolution)[:1000],
             f"{principal['principal_type']}:{principal['principal_id']}"[:200],
             req.winning_memory_id.strip()[:160], None if status_value == "open" else now,
             req.conflict_id),
        )
        for memory_id in {str(row["left_memory_id"]), str(row["right_memory_id"])}:
            open_count = int(conn.execute(
                "SELECT COUNT(*) FROM memory_conflicts WHERE status='open' AND (left_memory_id=? OR right_memory_id=?)",
                (memory_id, memory_id),
            ).fetchone()[0])
            conn.execute(
                "UPDATE memory_cognitive SET verification_status=?,review_status=?,updated_ts=? WHERE memory_id=?",
                (("conflicted" if open_count else "verified"),
                 ("needs_review" if open_count else "active"), now, memory_id),
            )
    return {"ok": True, "conflict_id": req.conflict_id, "status": status_value,
            "resolution": resolution}


@router.post("/governance/supersede")
def supersede_governed_memory(req: GovernedSupersedeRequest,
                              request: Request = None) -> dict[str, Any]:
    """Complete a traceable replacement lifecycle; never silently overwrite."""
    from .memory import _conn

    if req.old_memory_id == req.new_memory_id:
        raise HTTPException(400, "old and new memory must differ")
    principal = _request_principal(request)
    now = _now()
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        old = conn.execute(
            "SELECT m.invalid_at,c.* FROM memories m JOIN memory_cognitive c ON c.memory_id=m.id WHERE m.id=?",
            (req.old_memory_id,),
        ).fetchone()
        new = conn.execute(
            "SELECT m.invalid_at,c.* FROM memories m JOIN memory_cognitive c ON c.memory_id=m.id WHERE m.id=?",
            (req.new_memory_id,),
        ).fetchone()
        if not old or not new:
            raise HTTPException(404, "old or new memory not found")
        if old["invalid_at"] is not None:
            raise HTTPException(409, "old memory is already inactive")
        if str(new["memory_tier"] or "M0") == "M0" or str(new["verification_status"] or "") != "verified":
            raise HTTPException(409, "new memory must be verified M1 or higher")
        state = conn.execute("SELECT version FROM shared_state_versions WHERE scope_id=?", (req.scope_id,)).fetchone()
        version = int(state[0]) if state else 1
        if req.expected_view_version and req.expected_view_version != version:
            raise HTTPException(409, f"stale shared state: expected {req.expected_view_version}, current {version}")
        conn.execute("UPDATE memories SET invalid_at=?,superseded_by=? WHERE id=?",
                     (now, req.new_memory_id, req.old_memory_id))
        conn.execute(
            "UPDATE memory_cognitive SET review_status='retired',updated_ts=? WHERE memory_id=?",
            (now, req.old_memory_id),
        )
        conn.execute(
            "UPDATE memory_cognitive SET supersedes_memory_id=?,updated_ts=? WHERE memory_id=?",
            (req.old_memory_id, now, req.new_memory_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO memory_lineage(child_memory_id,parent_memory_id,relation_type,source_locator,created_ts) VALUES (?,?,?,?,?)",
            (req.new_memory_id, req.old_memory_id, "supersedes", req.evidence_locator[:1000], now),
        )
        try:
            conn.execute("INSERT OR REPLACE INTO edges(src,dst,weight,kind) VALUES (?,?,?,'supersede')",
                         (req.new_memory_id, req.old_memory_id, 1.0))
        except sqlite3.OperationalError:
            conn.execute("INSERT OR REPLACE INTO edges(src,dst,weight) VALUES (?,?,?)",
                         (req.new_memory_id, req.old_memory_id, 1.0))
        conn.execute(
            """UPDATE memory_conflicts SET status='resolved',resolution='superseded',
                      resolved_by=?,winning_memory_id=?,superseding_memory_id=?,resolved_ts=?
               WHERE status='open' AND (left_memory_id IN (?,?) OR right_memory_id IN (?,?))""",
            (f"{principal['principal_type']}:{principal['principal_id']}"[:200],
             req.new_memory_id, req.new_memory_id, now,
             req.old_memory_id, req.new_memory_id, req.old_memory_id, req.new_memory_id),
        )
        _sync_hot_fts_memory(conn, req.old_memory_id)
        _sync_hot_fts_memory(conn, req.new_memory_id)
        new_version = version + 1
        content_hash = hashlib.sha256(
            f"{req.old_memory_id}|{req.new_memory_id}|{req.evidence_locator}".encode("utf-8")
        ).hexdigest()[:24]
        conn.execute(
            """INSERT INTO shared_state_versions(scope_id,version,content_hash,updated_ts) VALUES (?,?,?,?)
               ON CONFLICT(scope_id) DO UPDATE SET version=excluded.version,content_hash=excluded.content_hash,updated_ts=excluded.updated_ts""",
            (req.scope_id[:160], new_version, content_hash, now),
        )
    return {"ok": True, "old_memory_id": req.old_memory_id,
            "new_memory_id": req.new_memory_id, "shared_state_version": new_version}


@router.post("/governance/nightly")
def run_nightly_governance(req: NightlyGovernanceRequest,
                           request: Request = None) -> dict[str, Any]:
    """Promote independently adopted, well-evidenced M2 memories to M3."""
    from .memory import _conn

    now = _now()
    principal = _request_principal(request)
    promoted: list[str] = []
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.memory_id FROM memory_cognitive c
            WHERE c.memory_tier='M2' AND c.verification_status='verified'
              AND c.review_status='active'
              AND c.evidence_count>=2
              AND (SELECT COUNT(*) FROM memory_activations a
                   WHERE a.memory_id=c.memory_id AND a.activation_kind IN ('adopted','confirmed'))>=3
              AND (
                (SELECT COUNT(DISTINCT occurred_date) FROM memory_activations a
                 WHERE a.memory_id=c.memory_id AND a.activation_kind IN ('adopted','confirmed'))>=2
                OR
                (SELECT COUNT(DISTINCT CASE WHEN TRIM(session_id)!='' THEN session_id END)
                 FROM memory_activations a WHERE a.memory_id=c.memory_id
                   AND a.activation_kind IN ('adopted','confirmed'))>=2
              )
              AND NOT EXISTS (
                SELECT 1 FROM memory_conflicts x WHERE x.status='open'
                  AND (x.left_memory_id=c.memory_id OR x.right_memory_id=c.memory_id)
              )
            ORDER BY c.activation_count DESC,c.last_verified_ts ASC
            LIMIT ?
            """,
            (req.limit,),
        ).fetchall()
        state = conn.execute(
            "SELECT version FROM shared_state_versions WHERE scope_id=?", (req.scope_id,)
        ).fetchone()
        version = int(state[0]) if state else 1
        for row in rows:
            memory_id = str(row["memory_id"])
            promotion_seed = "|".join([memory_id, "M3", "governance-v1", req.run_id])
            promotion_id = "prom-" + hashlib.sha256(promotion_seed.encode("utf-8")).hexdigest()[:20]
            conn.execute(
                "UPDATE memory_cognitive SET memory_tier='M3',governance_version='governance-v1',updated_ts=? WHERE memory_id=?",
                (now, memory_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_promotion_log(promotion_id,memory_id,from_tier,to_tier,evidence_json,rule_version,run_id,reviewer,created_ts) VALUES (?,?,?,'M3','[]','governance-v2',?,?,?)",
                (promotion_id, memory_id, "M2", req.run_id[:160],
                 f"{principal['principal_type']}:{principal['principal_id']}"[:120], now),
            )
            promoted.append(memory_id)
            version += 1
        if promoted:
            content_hash = hashlib.sha256("|".join(promoted).encode("utf-8")).hexdigest()[:24]
            conn.execute(
                "INSERT INTO shared_state_versions(scope_id,version,content_hash,updated_ts) VALUES (?,?,?,?) "
                "ON CONFLICT(scope_id) DO UPDATE SET version=excluded.version,content_hash=excluded.content_hash,updated_ts=excluded.updated_ts",
                (req.scope_id[:160], version, content_hash, now),
            )
    return {"ok": True, "run_id": req.run_id, "promoted_to_M3": promoted,
            "count": len(promoted), "shared_state_version": version}


@router.post("/cognitive/backfill")
def cognitive_backfill(req: CognitiveBackfillRequest) -> dict[str, Any]:
    """Classify one bounded batch of legacy memories.

    The endpoint is idempotent.  It intentionally avoids a full-table
    ``COUNT`` or ``INSERT SELECT`` so it remains responsive on NAS storage.
    """
    from .memory import _conn

    started = time.perf_counter()
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id,m.kind
            FROM memories AS m
            LEFT JOIN memory_cognitive AS c ON c.memory_id=m.id
            WHERE c.memory_id IS NULL
            ORDER BY m.rowid
            LIMIT ?
            """,
            (req.limit,),
        ).fetchall()
        now = _now()
        if rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO memory_cognitive(
                  memory_id,memory_form,lifecycle_stage,status,updated_ts
                ) VALUES (?,?,?,?,?)
                """,
                [
                    (
                        str(memory_id), _form_for(str(kind or ""), {}),
                        _stage_for(str(kind or ""), {}), "active", now,
                    )
                    for memory_id, kind in rows
                ],
            )
            for memory_id, _ in rows:
                _sync_hot_fts_memory(conn, str(memory_id))
        complete = len(rows) < req.limit
        if complete:
            conn.execute(
                "INSERT OR REPLACE INTO cognitive_migrations(name,completed_ts) VALUES (?,?)",
                ("register_existing_v1", now),
            )
    return {
        "ok": True,
        "processed": len(rows),
        "complete": complete,
        "last_memory_id": str(rows[-1][0]) if rows else "",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@router.post("/governance/rebuild-hot-index")
def rebuild_hot_index(req: HotFtsRebuildRequest) -> dict[str, Any]:
    """Explicit maintenance endpoint; never performs a hidden full rebuild."""
    if not req.confirm:
        raise HTTPException(400, "confirm=true is required")
    from .memory import _conn
    with _conn() as conn:
        return rebuild_hot_fts(conn)


@router.post("/hippocampus/stage")
def stage_hippocampal(req: HippocampalStageRequest) -> dict[str, Any]:
    from .memory import _conn

    memory_form = req.memory_form.casefold()
    if memory_form not in MEMORY_FORMS:
        raise HTTPException(400, "unsupported memory_form")
    secret_surface = req.summary + "\n" + json.dumps(req.evidence, ensure_ascii=False)
    if any(pattern.search(secret_surface) for pattern in _SECRET_PATTERNS):
        raise HTTPException(400, "hippocampal episode may contain a secret")
    now = _now()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT episode_id,status FROM hippocampal_inbox WHERE idempotency_key=?",
            (req.idempotency_key,),
        ).fetchone()
        if existing:
            return {"ok": True, "episode_id": existing[0], "status": existing[1], "deduplicated": True}
        episode_id = "hip-" + uuid.uuid4().hex[:16]
        conn.execute(
            """
            INSERT INTO hippocampal_inbox(
              episode_id,idempotency_key,source_id,summary,evidence_json,memory_form,
              domains_json,entities_json,importance,confidence,sensitivity,observed_at,
              status,memory_id,created_ts,updated_ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (episode_id, req.idempotency_key, req.source_id[:300], req.summary,
             json.dumps(req.evidence, ensure_ascii=False), memory_form,
             json.dumps(_list(req.domains), ensure_ascii=False),
             json.dumps(_list(req.entities), ensure_ascii=False), req.importance,
             req.confidence, req.sensitivity[:40], req.observed_at[:80], "staged", "", now, now),
        )
    return {"ok": True, "episode_id": episode_id, "status": "staged", "deduplicated": False}


@router.post("/hippocampus/promote")
def promote_hippocampal(req: HippocampalPromoteRequest) -> dict[str, Any]:
    from .memory import _conn

    status = req.status if req.status in {"promoted", "dismissed", "archived", "processed"} else "promoted"
    with _conn() as conn:
        if req.episode_id:
            where, target = "episode_id=?", req.episode_id
        elif req.source_id:
            where, target = "source_id=?", req.source_id
        else:
            raise HTTPException(400, "episode_id or source_id is required")
        if not conn.execute(f"SELECT 1 FROM hippocampal_inbox WHERE {where}", (target,)).fetchone():
            raise HTTPException(404, "hippocampal episode not found")
        cursor = conn.execute(
            f"UPDATE hippocampal_inbox SET status=?,memory_id=?,updated_ts=? WHERE {where}",
            (status, req.memory_id[:100], _now(), target),
        )
    return {"ok": True, "episode_id": req.episode_id, "source_id": req.source_id,
            "status": status, "memory_id": req.memory_id, "updated": int(cursor.rowcount or 0)}


@router.get("/hippocampus")
def hippocampal_status(status: str = "", limit: int = Query(default=30, ge=1, le=200)) -> dict[str, Any]:
    from .memory import _conn

    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT * FROM hippocampal_inbox WHERE status=? ORDER BY updated_ts DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hippocampal_inbox ORDER BY updated_ts DESC LIMIT ?", (limit,)
            ).fetchall()
        counts = {str(k): int(v) for k, v in conn.execute(
            "SELECT status,COUNT(*) FROM hippocampal_inbox GROUP BY status"
        )}
    items = []
    for row in rows:
        item = dict(row)
        item["domains"] = _json(item.pop("domains_json", "[]"), [])
        item["entities"] = _json(item.pop("entities_json", "[]"), [])
        item["evidence"] = _json(item.pop("evidence_json", "[]"), [])
        items.append(item)
    return {"ok": True, "counts": counts, "items": items}


def _query_depth(value: str) -> str:
    return QUERY_DEPTH_ALIASES.get(str(value or "Q2").strip().upper(), "Q2")


def _depth_budget(depth: str) -> dict[str, int]:
    return {
        "Q0": {"timeout_ms": 0, "limit": 0, "max_chars": 0, "lexical": 0, "vector": 0, "graph": 0},
        "Q1": {"timeout_ms": 250, "limit": 4, "max_chars": 2400, "lexical": 24, "vector": 24, "graph": 0},
        "Q2": {"timeout_ms": 900, "limit": 8, "max_chars": 6400, "lexical": 48, "vector": 48, "graph": 40},
        "Q3": {"timeout_ms": 2500, "limit": 16, "max_chars": 16000, "lexical": 96, "vector": 96, "graph": 80},
    }[depth]


def _allowed_memory_ids(conn: sqlite3.Connection, *, principal_type: str,
                        principal_id: str, roles: list[str],
                        candidate_ids: set[str] | None = None) -> set[str]:
    """Resolve ACLs before any lexical/vector candidate is exposed.

    Legacy rows without a governance sidecar belong only to the configured
    default owner.  Restricted and agent memories always require an explicit
    ACL row.  Expired ACL entries fail closed.
    """
    principal_type = str(principal_type or "user").strip().casefold()[:40]
    principal_id = str(principal_id or DEFAULT_OWNER_ID).strip()[:160] or DEFAULT_OWNER_ID
    role_values = [str(value).strip()[:160] for value in roles if str(value).strip()][:20]
    now = _now()
    candidate_values = list(dict.fromkeys(str(value) for value in (candidate_ids or set()) if value))
    if candidate_ids is not None and not candidate_values:
        return set()
    # SQLite builds poor plans for a large OR expression over the complete
    # memory table.  Recall therefore resolves permissions only for the small
    # ID-only pool produced by FTS/vector ranking.  No content is fetched until
    # this check has passed.  The unbounded form remains for maintenance/tests.
    candidate_clause = ""
    candidate_params: list[Any] = []
    if candidate_values:
        candidate_clause = f" AND m.id IN ({','.join('?' for _ in candidate_values)})"
        candidate_params = candidate_values
    params: list[Any] = [
        principal_id, DEFAULT_OWNER_ID,
        principal_type, principal_id,
        principal_id,
        principal_type, principal_id, now,
    ]
    role_clause = ""
    if role_values:
        role_clause = (
            " OR EXISTS (SELECT 1 FROM memory_acl ar WHERE ar.memory_id=m.id "
            f"AND ar.principal_type='role' AND ar.principal_id IN ({','.join('?' for _ in role_values)}) "
            "AND ar.permission IN ('read','write','admin') AND (ar.expires_at IS NULL OR ar.expires_at>?))"
        )
        params.extend(role_values)
        params.append(now)
    sql = f"""
        SELECT m.id
        FROM memories AS m
        LEFT JOIN memory_cognitive AS c ON c.memory_id=m.id
        WHERE m.invalid_at IS NULL {candidate_clause} AND (
          (c.memory_id IS NULL AND ?=?)
          OR (COALESCE(c.visibility,'private')='private' AND c.owner_type=? AND c.owner_id=?)
          OR (COALESCE(c.visibility,'private')='team' AND c.owner_id=?)
          OR EXISTS (
              SELECT 1 FROM memory_acl a WHERE a.memory_id=m.id
              AND a.principal_type=? AND a.principal_id=?
              AND a.permission IN ('read','write','admin')
              AND (a.expires_at IS NULL OR a.expires_at>?)
          )
          {role_clause}
        )
    """
    return {str(row[0]) for row in conn.execute(sql, [*candidate_params, *params])}


def _bounded_graph_ranks(conn: sqlite3.Connection, seeds: list[str], allowed: set[str] | None,
                         cap: int, deadline: float,
                         acl_resolver: Any | None = None) -> list[str]:
    if not seeds or cap <= 0 or time.perf_counter() >= deadline:
        return []
    frontier = list(dict.fromkeys(seeds[:16]))
    seen = set(frontier)
    scored: dict[str, float] = {}
    for _ in range(2):
        if not frontier or len(scored) >= cap or time.perf_counter() >= deadline:
            break
        placeholders = ",".join("?" for _ in frontier)
        rows = conn.execute(
            f"SELECT src,dst,weight FROM edges WHERE src IN ({placeholders}) ORDER BY weight DESC LIMIT ?",
            [*frontier, cap * 4],
        ).fetchall()
        raw_scores: dict[str, float] = {}
        for _, dst, weight in rows:
            memory_id = str(dst)
            if memory_id not in seen:
                raw_scores[memory_id] = max(raw_scores.get(memory_id, 0.0), float(weight or 0.0))
        step_allowed = (
            set(acl_resolver(set(raw_scores))) if acl_resolver is not None
            else (set(raw_scores) if allowed is None else allowed)
        )
        next_frontier: list[str] = []
        for memory_id, weight in sorted(raw_scores.items(), key=lambda row: row[1], reverse=True):
            if memory_id in step_allowed and memory_id not in seen:
                scored[memory_id] = max(scored.get(memory_id, 0.0), weight)
                seen.add(memory_id)
                next_frontier.append(memory_id)
                if len(scored) >= cap:
                    break
        frontier = next_frontier[:16]
    return [memory_id for memory_id, _ in sorted(scored.items(), key=lambda row: row[1], reverse=True)]


def _rrf_candidates(conn: sqlite3.Connection, query: str, *, allowed: set[str] | None = None,
                    budget: dict[str, int], timeout_ms: int,
                    principal_type: str = "user", principal_id: str = DEFAULT_OWNER_ID,
                    roles: list[str] | None = None,
                    enforce_acl: bool = True,
                    lexical_scope: str = "all") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ACL-bounded candidates using rank fusion, never raw-score mixing."""
    from . import associative, semantic

    started = time.perf_counter()
    deadline = started + max(0.05, timeout_ms / 1000.0)
    lexical_cap = int(budget["lexical"])
    vector_cap = int(budget["vector"])
    graph_cap = int(budget["graph"])

    lexical_interrupted = False
    # A broad FTS5 trigram OR can spend seconds ranking inside a virtual-table
    # opcode.  The progress handler alone is not a dependable wall-clock
    # deadline there.  Connection.interrupt() is thread-safe and also stops the
    # active virtual-table statement, so fixed-budget recall remains fixed.
    remaining = max(0.001, deadline - time.perf_counter())
    hard_interrupted = threading.Event()

    def interrupt_lexical() -> None:
        hard_interrupted.set()
        conn.interrupt()

    lexical_timer = threading.Timer(remaining, interrupt_lexical)
    lexical_timer.daemon = True
    conn.set_progress_handler(
        lambda: 1 if time.perf_counter() >= deadline else 0,
        2_000,
    )
    lexical_timer.start()
    try:
        lexical_all = associative.fts_search(
            conn, query, k=min(900, max(lexical_cap * 4, lexical_cap)),
            candidate_pool=min(900, max(lexical_cap * 4, lexical_cap)),
            scope=lexical_scope,
        )
    except sqlite3.OperationalError as exc:
        if "interrupt" not in str(exc).casefold():
            raise
        lexical_all = []
        lexical_interrupted = True
    finally:
        lexical_timer.cancel()
        conn.set_progress_handler(None, 0)
    lexical_interrupted = lexical_interrupted or hard_interrupted.is_set()
    vector_all: list[str] = []
    vector_scores: dict[str, float] = {}
    vector_timed_out = False
    if vector_cap and time.perf_counter() < deadline:
        remaining = max(0.05, deadline - time.perf_counter())
        vector_hits = semantic.vector_search(
            conn, query, k=min(384, max(vector_cap * 8, vector_cap)),
            candidate_ids=None, timeout=remaining, warm_only=True,
        )
        vector_all = [memory_id for memory_id, _ in vector_hits]
        vector_scores = {memory_id: float(score) for memory_id, score in vector_hits}
        vector_timed_out = time.perf_counter() >= deadline
    raw_pool = set(lexical_all) | set(vector_all)
    raw_lexical_ranks = {memory_id: rank for rank, memory_id in enumerate(lexical_all, 1)}
    raw_vector_ranks = {memory_id: rank for rank, memory_id in enumerate(vector_all, 1)}
    if allowed is None:
        if enforce_acl:
            allowed = _allowed_memory_ids(
                conn, principal_type=principal_type, principal_id=principal_id,
                roles=list(roles or []), candidate_ids=raw_pool,
            )
        else:
            allowed = raw_pool
    denied_ids = sorted(raw_pool - set(allowed or set()))
    lexical = [memory_id for memory_id in lexical_all if memory_id in allowed][:lexical_cap]
    vector = [memory_id for memory_id in vector_all if memory_id in allowed][:vector_cap]
    seeds = list(dict.fromkeys([*lexical[:12], *vector[:12]]))
    acl_resolver = None
    if enforce_acl:
        acl_resolver = lambda ids: _allowed_memory_ids(
            conn, principal_type=principal_type, principal_id=principal_id,
            roles=list(roles or []), candidate_ids=ids,
        )
    graph = _bounded_graph_ranks(
        conn, seeds, allowed if not enforce_acl else None, graph_cap, deadline,
        acl_resolver=acl_resolver,
    )
    ranks = {
        "lexical": {memory_id: rank for rank, memory_id in enumerate(lexical, 1)},
        "vector": {memory_id: rank for rank, memory_id in enumerate(vector, 1)},
        "graph": {memory_id: rank for rank, memory_id in enumerate(graph, 1)},
    }
    pool = set(lexical) | set(vector) | set(graph)
    if not pool:
        timed_out = lexical_interrupted or vector_timed_out or time.perf_counter() >= deadline
        return [], {
            "channels": {name: 0 for name in ranks}, "partial": timed_out,
            "stop_reason": "timeout" if timed_out else "no_match",
            "candidate_audit": [{
                "memory_id": memory_id,
                "permission_result": "denied",
                "filter_reason": "acl_denied",
                "channel_ranks": {
                    "lexical": raw_lexical_ranks.get(memory_id),
                    "vector": raw_vector_ranks.get(memory_id),
                    "graph": None,
                },
            } for memory_id in denied_ids],
        }
    placeholders = ",".join("?" for _ in pool)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT * FROM memories WHERE id IN ({placeholders}) AND invalid_at IS NULL",
        list(pool),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        memory = dict(row)
        memory_id = str(memory["id"])
        channel_ranks = {name: mapping.get(memory_id) for name, mapping in ranks.items()}
        rrf_score = sum(1.0 / (RRF_K + rank) for rank in channel_ranks.values() if rank)
        content = str(memory.get("content") or "").strip()
        title = (content.split("\n", 1)[0].strip() or content)[:80]
        source = ""
        try:
            raw_meta = json.loads(memory.get("meta") or "{}")
            source = str(raw_meta.get("title") or raw_meta.get("author") or raw_meta.get("source") or "").strip()
        except Exception:
            raw_meta = {}
        via = [name for name, rank in channel_ranks.items() if rank]
        items.append({
            "id": memory_id,
            "kind": memory.get("kind"),
            "importance": memory.get("importance"),
            "title": title,
            "snippet": content[:360],
            "token_count": memory.get("token_count"),
            "source": source,
            "score": rrf_score,
            "rrf_score": round(rrf_score, 8),
            "channel_ranks": channel_ranks,
            "vector_similarity": round(vector_scores.get(memory_id, 0.0), 5),
            "via": "+".join(via),
        })
    items.sort(key=lambda item: (float(item["rrf_score"]), float(item["vector_similarity"])), reverse=True)
    partial = lexical_interrupted or time.perf_counter() >= deadline
    return items, {
        "channels": {name: len(mapping) for name, mapping in ranks.items()},
        "raw_channels": {"lexical": len(lexical_all), "vector": len(vector_all), "graph": len(graph)},
        "partial": partial,
        "stop_reason": "timeout" if partial else "complete",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "candidate_audit": [{
            "memory_id": memory_id,
            "permission_result": "denied",
            "filter_reason": "acl_denied",
            "channel_ranks": {
                "lexical": raw_lexical_ranks.get(memory_id),
                "vector": raw_vector_ranks.get(memory_id),
                "graph": None,
            },
        } for memory_id in denied_ids],
    }


class PlannedRecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    depth: str = "L2"
    query_depth: str = ""
    limit: int = Field(default=8, ge=1, le=30)
    token_budget: int = Field(default=1600, ge=100, le=12000)
    max_chars: int = Field(default=0, ge=0, le=100000)
    timeout_ms: int = Field(default=0, ge=0, le=10000)
    domains: list[str] = Field(default_factory=list)
    memory_forms: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    project: str = ""
    session_id: str = ""
    include_books: bool = False
    include_hypotheses: bool = False
    apply_dream_accessibility: bool = True
    touch: bool = True


def _facet_map(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict[str, list[str]]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    result: dict[str, dict[str, list[str]]] = {}
    for memory_id, facet_type, facet_value in conn.execute(
        f"SELECT memory_id,facet_type,facet_value FROM memory_facets WHERE memory_id IN ({placeholders})",
        ids,
    ):
        result.setdefault(str(memory_id), {}).setdefault(str(facet_type), []).append(str(facet_value))
    return result


def _dream_accessibility_map(conn: sqlite3.Connection, ids: list[str],
                             *, on_date: str = "") -> dict[str, float]:
    """Return the decayed and capped accessibility boost for candidate IDs."""
    if not ids:
        return {}
    selected_date = date.fromisoformat(on_date) if on_date else date.today()
    placeholders = ",".join("?" for _ in ids)
    totals: dict[str, float] = {}
    for memory_id, occurred_date, base_delta in conn.execute(
        f"SELECT memory_id,occurred_date,base_delta "
        f"FROM memory_dream_accessibility_events WHERE memory_id IN ({placeholders})",
        ids,
    ):
        try:
            age_days = (selected_date - date.fromisoformat(str(occurred_date))).days
        except ValueError:
            continue
        if age_days < 0:
            continue
        decayed = float(base_delta) * math.pow(0.5, age_days / _DREAM_ACCESSIBILITY_HALF_LIFE_DAYS)
        key = str(memory_id)
        totals[key] = totals.get(key, 0.0) + decayed
    return {key: min(_DREAM_ACCESSIBILITY_CAP, value) for key, value in totals.items()}


@router.post("/recall-planned")
def recall_planned(req: PlannedRecallRequest,
                   request: Request = None) -> dict[str, Any]:
    """Return a bounded evidence pack using deterministic depth and facet rules."""
    from . import associative
    from .memory import _conn

    started = time.perf_counter()
    principal = _request_principal(request)
    query_depth = _query_depth(req.query_depth or req.depth)
    depth = "L" + query_depth[-1]
    server_budget = _depth_budget(query_depth)
    timeout_ms = min(int(req.timeout_ms or server_budget["timeout_ms"]), server_budget["timeout_ms"])
    result_limit = min(req.limit, server_budget["limit"])
    max_chars = min(int(req.max_chars or server_budget["max_chars"]), server_budget["max_chars"])
    if query_depth == "Q0":
        return {"ok": True, "query": req.query, "depth": depth,
                "query_depth": query_depth, "items": [],
                "candidate_count": 0, "token_count": 0, "latency_ms": 0.0,
                "reason": "Q0 uses only current conversation context",
                "budget": server_budget, "partial": False, "stop_reason": "conversation_only",
                "retrieval": {"principal_type": principal["principal_type"],
                              "principal_id": principal["principal_id"],
                              "identity_source": principal["identity_source"],
                              "acl_enforced": True}}
    with _conn() as acl_conn:
        candidates, channel_trace = _rrf_candidates(
            acl_conn, req.query, allowed=None,
            budget=server_budget, timeout_ms=timeout_ms,
            principal_type=principal["principal_type"], principal_id=principal["principal_id"],
            roles=principal["roles"], enforce_acl=True,
            lexical_scope="all" if query_depth == "Q3" else "hot",
        )
    # No legacy fallback: an unfiltered endpoint must never bypass ACL.
    ids = [str(item.get("id") or "") for item in candidates if item.get("id")]
    wanted_domains = {x.casefold() for x in _list(req.domains)}
    wanted_entities = {x.casefold() for x in _list(req.entities)}
    wanted_forms = {x.casefold() for x in _list(req.memory_forms)} & MEMORY_FORMS
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        cognitive: dict[str, dict[str, Any]] = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            cognitive = {str(row["memory_id"]): dict(row) for row in conn.execute(
                f"SELECT * FROM memory_cognitive WHERE memory_id IN ({placeholders})", ids
            )}
        facets = _facet_map(conn, ids)
        dream_accessibility = (
            _dream_accessibility_map(conn, ids)
            if req.apply_dream_accessibility else {}
        )
        evidence_map: dict[str, list[dict[str, Any]]] = {}
        conflict_map: dict[str, list[dict[str, Any]]] = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            for row in conn.execute(
                f"SELECT * FROM memory_evidence WHERE memory_id IN ({placeholders})", ids
            ):
                evidence_map.setdefault(str(row["memory_id"]), []).append(dict(row))
            for row in conn.execute(
                f"SELECT * FROM memory_conflicts WHERE status='open' AND (left_memory_id IN ({placeholders}) OR right_memory_id IN ({placeholders}))",
                [*ids, *ids],
            ):
                data = dict(row)
                conflict_map.setdefault(str(row["left_memory_id"]), []).append(data)
                conflict_map.setdefault(str(row["right_memory_id"]), []).append(data)
        ranked: list[dict[str, Any]] = []
        audit: dict[str, dict[str, Any]] = {
            str(row.get("memory_id")): dict(row)
            for row in channel_trace.get("candidate_audit", [])
            if row.get("memory_id")
        }
        for item in candidates:
            memory_id = str(item.get("id") or "")
            audit.setdefault(memory_id, {
                "memory_id": memory_id,
                "permission_result": "allowed",
                "filter_reason": "",
                "channel_ranks": item.get("channel_ranks", {}),
            })
            meta = cognitive.get(memory_id, {})
            memory_form = str(meta.get("memory_form") or _form_for(str(item.get("kind") or ""), {}))
            stage = str(meta.get("lifecycle_stage") or "consolidated")
            memory_tier = str(meta.get("memory_tier") or "M0")
            verification_status = str(meta.get("verification_status") or "unreviewed")
            review_status = str(meta.get("review_status") or "quarantine")
            if meta.get("status", "active") != "active":
                audit[memory_id]["filter_reason"] = "inactive"
                continue
            if verification_status == "rejected" or review_status == "retired":
                audit[memory_id]["filter_reason"] = "governance_rejected_or_retired"
                continue
            if memory_form == "hypothesis" and not req.include_hypotheses:
                audit[memory_id]["filter_reason"] = "hypothesis_excluded"
                continue
            is_book = str(item.get("kind") or "").casefold() in {"knowledge_book", "book"} or meta.get("source") == "books"
            if is_book and not (req.include_books or depth == "L3"):
                audit[memory_id]["filter_reason"] = "book_excluded"
                continue
            if depth == "L1" and stage == "deep":
                audit[memory_id]["filter_reason"] = "deep_memory_excluded"
                continue
            row_facets = facets.get(memory_id, {})
            domain_hits = wanted_domains & set(row_facets.get("domain", []))
            entity_hits = wanted_entities & set(row_facets.get("entity", []))
            base_score = float(item.get("rrf_score") or item.get("score") or 0.0)
            multiplier = 1.0
            multiplier += min(0.06, 0.03 * len(domain_hits))
            multiplier += min(0.06, 0.03 * len(entity_hits))
            form_hit = bool(wanted_forms and memory_form in wanted_forms)
            if form_hit:
                multiplier += 0.03
            confidence = max(0.0, min(1.0, float(meta.get("confidence") or 0.7)))
            multiplier += (confidence - 0.5) * 0.04
            if req.project and req.project.casefold() in set(row_facets.get("project", [])):
                multiplier += 0.04
            dream_boost = float(dream_accessibility.get(memory_id, 0.0))
            multiplier += dream_boost
            open_conflicts = conflict_map.get(memory_id, [])
            if open_conflicts:
                multiplier *= 0.90
                audit[memory_id]["filter_reason"] = "conflict_downgraded"
            score = base_score * multiplier
            explanation = list(item.get("via", "").split("+")) if item.get("via") else []
            explanation += [f"领域:{x}" for x in sorted(domain_hits)]
            explanation += [f"实体:{x}" for x in sorted(entity_hits)]
            if form_hit:
                explanation.append(f"记忆类型:{memory_form}")
            if dream_boost > 0:
                explanation.append("梦境可达性")
            ranked.append({**item, "score": round(score, 4), "memory_form": memory_form,
                           "lifecycle_stage": stage,
                           "memory_tier": memory_tier,
                           "verification_status": verification_status,
                           "review_status": review_status,
                           "confidence": round(float(meta.get("confidence") or 0.7), 3),
                           "facets": row_facets,
                           "why_recalled": [x for x in explanation if x],
                           "source_id": str(meta.get("source_id") or ""),
                           "root_source_id": str(meta.get("root_source_id") or meta.get("source_id") or ""),
                           "parent_memory_id": str(meta.get("parent_memory_id") or ""),
                           "evidence": evidence_map.get(memory_id, []),
                            "open_conflicts": open_conflicts,
                             "dream_accessibility_boost": round(dream_boost, 6),
                             "dream_accessibility_multiplier": round(1.0 + dream_boost, 6),
                             "score_multiplier": round(multiplier, 4)})
        ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        chosen: list[dict[str, Any]] = []
        used_tokens = 0
        used_chars = 0
        covered: set[str] = set()
        selection_stop = "complete"
        for item in ranked:
            memory_id = str(item.get("id") or "")
            snippet = str(item.get("snippet") or "")
            remaining_chars = max_chars - used_chars
            remaining_tokens = req.token_budget - used_tokens
            if remaining_chars <= 0:
                audit[memory_id]["filter_reason"] = "char_budget_dropped"
                selection_stop = "char_budget"
                continue
            if remaining_tokens <= 0:
                audit[memory_id]["filter_reason"] = "token_budget_dropped"
                selection_stop = "token_budget"
                continue
            # One Unicode code point is a conservative upper bound for the
            # mixed Chinese/English snippets returned by this endpoint.
            allowed_length = min(len(snippet), remaining_chars, remaining_tokens)
            if allowed_length <= 0:
                audit[memory_id]["filter_reason"] = "budget_dropped"
                selection_stop = "budget"
                continue
            if allowed_length < len(snippet):
                snippet = snippet[:allowed_length]
                item["snippet"] = snippet
                item["truncated_by_budget"] = True
            estimated = len(snippet)
            item_chars = len(snippet)
            coverage = {
                *[f"domain:{value}" for value in item.get("facets", {}).get("domain", [])],
                *[f"entity:{value}" for value in item.get("facets", {}).get("entity", [])],
                f"source:{item.get('root_source_id') or item.get('source_id') or item.get('id')}",
                f"form:{item.get('memory_form')}",
            }
            if chosen and not (coverage - covered) and float(item.get("rrf_score") or 0.0) < 0.02:
                audit[memory_id]["filter_reason"] = "coverage_duplicate"
                continue
            item["estimated_tokens"] = estimated
            chosen.append(item)
            used_tokens += estimated
            used_chars += item_chars
            covered.update(coverage)
            if len(chosen) >= result_limit:
                selection_stop = "item_budget"
                break
        chosen_ranks = {str(item["id"]): rank for rank, item in enumerate(chosen, 1)}
        for memory_id, rank in chosen_ranks.items():
            audit[memory_id]["final_rank"] = rank
        if len(chosen) >= result_limit:
            for item in ranked:
                memory_id = str(item.get("id") or "")
                if memory_id not in chosen_ranks and not audit[memory_id].get("filter_reason"):
                    audit[memory_id]["filter_reason"] = "item_budget_dropped"
        source_counts = Counter(str(item.get("source") or item.get("kind") or "memory") for item in chosen)
        state_row = conn.execute(
            "SELECT version FROM shared_state_versions WHERE scope_id=?",
            ((req.project or "global")[:160],),
        ).fetchone()
        shared_state_version = int(state_row[0]) if state_row else 1
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        run_id = "rr-" + uuid.uuid4().hex[:16]
        created_ts = _now()
    _enqueue_retrieval_event({
        "created_ts": created_ts,
        "chosen_ids": [item["id"] for item in chosen] if req.touch else [],
        "row": (
            run_id, req.session_id[:200], hashlib.sha256(req.query.encode("utf-8")).hexdigest()[:24],
            depth, len(candidates), len(chosen), used_tokens, latency_ms,
            json.dumps(source_counts, ensure_ascii=False), created_ts,
        ),
        "item_rows": [(
            run_id, memory_id, row.get("channel_ranks", {}).get("lexical"),
            row.get("channel_ranks", {}).get("vector"),
            row.get("channel_ranks", {}).get("graph"),
            float(next((item.get("rrf_score") or 0.0 for item in candidates if str(item.get("id")) == memory_id), 0.0)),
            str(row.get("permission_result") or "allowed"), str(row.get("filter_reason") or ""),
            row.get("final_rank"),
            str(next((item.get("source_id") or "" for item in ranked if str(item.get("id")) == memory_id), "")),
            created_ts,
        ) for memory_id, row in audit.items()],
    })
    evidence_pack = [{
        "memory_id": item["id"], "source_id": item.get("source_id", ""),
        "source": item.get("source", ""), "evidence": item.get("evidence", []),
        "open_conflicts": item.get("open_conflicts", []),
    } for item in chosen]
    citation_pack = [{
        "memory_id": item["id"],
        "source_id": item.get("source_id", ""),
        "root_source_id": item.get("root_source_id", ""),
        "locators": [row.get("locator", "") for row in item.get("evidence", []) if row.get("locator")],
    } for item in chosen]
    partial = bool(channel_trace.get("partial")) or selection_stop != "complete"
    stop_reason = str(channel_trace.get("stop_reason") or "complete")
    if selection_stop != "complete":
        stop_reason = selection_stop
    return {"ok": True, "run_id": run_id, "query": req.query, "depth": depth,
            "query_depth": query_depth,
            "items": chosen, "candidate_count": len(candidates), "result_count": len(chosen),
            "token_count": used_tokens, "char_count": used_chars, "latency_ms": latency_ms,
            "source_counts": dict(source_counts),
            "evidence_pack": evidence_pack,
            "citation_pack": citation_pack,
            "budget": {**server_budget, "limit": result_limit, "max_chars": max_chars,
                       "token_budget": req.token_budget, "timeout_ms": timeout_ms},
            "partial": partial, "stop_reason": stop_reason,
            "shared_state_version": shared_state_version,
            "retrieval_path": [
                "acl_filter",
                "fts5_bm25", "vector", "rrf", "bounded_graph",
                *(["dream_accessibility"] if req.apply_dream_accessibility else []),
                "governance", "coverage_and_budget",
            ],
            "retrieval": {**channel_trace,
                          "filters": {"domains": sorted(wanted_domains),
                                      "entities": sorted(wanted_entities),
                                      "forms": sorted(wanted_forms),
                                      "principal_type": principal["principal_type"],
                                      "principal_id": principal["principal_id"],
                                      "identity_source": principal["identity_source"]}}}


@router.get("/cognitive/stats")
def cognitive_stats() -> dict[str, Any]:
    from . import ppr, semantic
    from .memory import _conn

    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        # Force narrow covering indexes.  Counting the vector table directly
        # may scan its large embedding BLOBs on SQLite/NAS storage.
        totals = int(conn.execute(
            "SELECT COUNT(id) FROM memories INDEXED BY sqlite_autoindex_memories_1"
        ).fetchone()[0])
        vector_rows = int(conn.execute(
            "SELECT COUNT(memory_id) FROM memories_vec INDEXED BY sqlite_autoindex_memories_vec_1"
        ).fetchone()[0])
        hot_fts_rows = int(conn.execute(
            "SELECT COUNT(*) FROM memories_hot_fts"
        ).fetchone()[0])
        forms = {str(k): int(v) for k, v in conn.execute(
            "SELECT memory_form,COUNT(*) FROM memory_cognitive GROUP BY memory_form"
        )}
        stages = {str(k): int(v) for k, v in conn.execute(
            "SELECT lifecycle_stage,COUNT(*) FROM memory_cognitive GROUP BY lifecycle_stage"
        )}
        domains = [{"name": str(k), "count": int(v)} for k, v in conn.execute(
            "SELECT facet_value,COUNT(*) FROM memory_facets WHERE facet_type='domain' GROUP BY facet_value ORDER BY COUNT(*) DESC LIMIT 20"
        )]
        recent = [dict(row) for row in conn.execute(
            "SELECT * FROM retrieval_runs ORDER BY created_ts DESC LIMIT 20"
        )]
        inbox = {str(k): int(v) for k, v in conn.execute(
            "SELECT status,COUNT(*) FROM hippocampal_inbox GROUP BY status"
        )}
        open_conflicts = int(conn.execute(
            "SELECT COUNT(*) FROM memory_conflicts WHERE status='open'"
        ).fetchone()[0])
        tiers = {str(k): int(v) for k, v in conn.execute(
            "SELECT memory_tier,COUNT(*) FROM memory_cognitive GROUP BY memory_tier"
        )}
        verification = {str(k): int(v) for k, v in conn.execute(
            "SELECT verification_status,COUNT(*) FROM memory_cognitive GROUP BY verification_status"
        )}
        state_rows = {str(k): int(v) for k, v in conn.execute(
            "SELECT scope_id,version FROM shared_state_versions ORDER BY scope_id LIMIT 100"
        )}
        activations = {str(k): int(v) for k, v in conn.execute(
            "SELECT activation_kind,COUNT(*) FROM memory_activations GROUP BY activation_kind"
        )}
        dream_accessibility_events = int(conn.execute(
            "SELECT COUNT(*) FROM memory_dream_accessibility_events"
        ).fetchone()[0])
    for row in recent:
        row["source_counts"] = _json(row.pop("source_counts_json", "{}"), {})
    classified = sum(forms.values())
    return {"ok": True, "total_memories": totals,
            "classified_memories": classified,
            "unclassified_memories": max(0, totals - classified),
            "migration_mode": "lazy_batched",
            "forms": forms,
            "lifecycle": stages, "domains": domains, "hippocampus": inbox,
            "open_conflicts": open_conflicts, "recent_retrievals": recent,
            "governance": {
                "version": GOVERNANCE_VERSION,
                "tiers": tiers,
                "verification": verification,
                "shared_state_versions": state_rows,
                "retrieval_fusion": "rrf-v1",
                "dream_accessibility_events": dream_accessibility_events,
                "dream_accessibility_counts_for_promotion": False,
                "acl_before_recall": True,
                "hot_fts_rows": hot_fts_rows,
                "hot_fts_depths": ["Q1", "Q2"],
                "ordinary_write_tier": "M0",
                "acl_enforced": True,
                "activations": activations,
            },
            "readiness": {
                "vector_cache": "ready" if semantic._CACHE.get("mat") is not None else "cold",
                "graph_cache": "ready" if ppr._CACHE.get("P") is not None else "cold",
            },
            "vector_index": {
                "backend": "exact_numpy",
                "rows": vector_rows,
                "migration_threshold": 100000,
                "migration_required": vector_rows >= 100000,
                "recommended_next_backend": "hnsw_or_sqlite_vec",
            },
            "architecture_version": "cognitive-v2"}
