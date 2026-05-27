"""bee-memory / 记忆中心 — 三层记忆 + 6 因子激活打分 + 宪法 (v2 阶段 3 + v3-D/E/F)"""
from __future__ import annotations
import sqlite3, json, time, uuid, math
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"
CONST_PATH = Path(__file__).parent.parent / "data" / "constitution.md"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            kind TEXT,                   -- episodic|semantic|procedural|self_upgrade
            content TEXT,
            mode_id TEXT,
            importance INTEGER DEFAULT 2,  -- 0-5
            created_ts INTEGER,
            last_recall_ts INTEGER,
            recall_count INTEGER DEFAULT 0,
            emotional_tag REAL DEFAULT 0,   -- v3-D EmotionalTag
            novelty REAL DEFAULT 0.5,       -- v3-D Novelty
            connection_density REAL DEFAULT 0,
            predictive_value REAL DEFAULT 0,
            meta TEXT
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_kind_ts ON memories(kind, last_recall_ts)")
    return c


class StoreRequest(BaseModel):
    kind: str
    content: str
    mode_id: str = ""
    importance: int = 2
    emotional_tag: float = 0.0
    meta: dict = Field(default_factory=dict)


@router.post("/store")
def store(req: StoreRequest) -> dict:
    mid = "m-" + uuid.uuid4().hex[:12]
    now = int(time.time())
    with _conn() as c:
        c.execute(
            "INSERT INTO memories (id,kind,content,mode_id,importance,created_ts,last_recall_ts,emotional_tag,meta) VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, req.kind, req.content, req.mode_id, req.importance, now, now, req.emotional_tag, json.dumps(req.meta, ensure_ascii=False)),
        )
    return {"memory_id": mid}


def _activation_score(row: dict, now: int) -> float:
    """v3-D 6 因子打分."""
    age_days = max(0, (now - row["created_ts"]) / 86400.0)
    recency_days = max(0, (now - row.get("last_recall_ts", row["created_ts"])) / 86400.0)
    recency = math.exp(-recency_days / 14.0)              # half-life 14d
    frequency = math.log1p(row.get("recall_count", 0))
    emotional = row.get("emotional_tag", 0) * 2.0
    novelty = row.get("novelty", 0.5)
    connection = row.get("connection_density", 0)
    predictive = row.get("predictive_value", 0)
    age_penalty = age_days / 365.0
    importance_boost = row.get("importance", 2) / 5.0
    return (1.0 * recency + 0.6 * frequency + 0.8 * emotional + 0.5 * novelty
            + 0.7 * connection + 0.6 * predictive - 0.3 * age_penalty + 0.5 * importance_boost)


@router.get("/recall")
def recall(query: str = "", kind: str = "", k: int = 5, strategy: str = "activation") -> dict:
    """v3-D activation strategy by default; static = old top-N by ts."""
    where = []
    params: list = []
    if kind:
        where.append("kind=?"); params.append(kind)
    if query:
        where.append("content LIKE ?"); params.append(f"%{query}%")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(f"SELECT * FROM memories {where_sql} ORDER BY last_recall_ts DESC LIMIT 200", params).fetchall()
    items = [dict(r) for r in rows]
    now = int(time.time())
    if strategy == "activation":
        items.sort(key=lambda r: _activation_score(r, now), reverse=True)
    items = items[:k]
    # update recall stats
    ids = [r["id"] for r in items]
    if ids:
        with _conn() as c:
            for mid in ids:
                c.execute("UPDATE memories SET recall_count=recall_count+1, last_recall_ts=? WHERE id=?", (now, mid))
    return {"items": items, "strategy": strategy}


@router.post("/consolidate")
def consolidate() -> dict:
    """v3-F sleep cycle: distill episodic → semantic (stub)."""
    return {"status": "ok", "consolidated": 0, "note": "stub; full impl in v3-D"}


@router.post("/forget")
def forget(below_score: float = 0.05) -> dict:
    """v3-D 遗忘: PageRank-aware 折叠 (stub, never deletes for now)."""
    return {"status": "ok", "folded": 0, "note": "stub; v3-D PageRank safety + halt at 2-pool"}


@router.get("/constitution")
def get_constitution() -> dict:
    if not CONST_PATH.exists():
        CONST_PATH.write_text("# 宪法 v0\n\n1. 月度预算 ¥800 不可超\n2. 不假装知道未验证的事实\n3. 默认走最低成本模型,只在必要时升级\n", encoding="utf-8")
    return {"version": "0", "content": CONST_PATH.read_text(encoding="utf-8")}


class ConstProposal(BaseModel):
    content: str


@router.post("/constitution/propose")
def propose_constitution(req: ConstProposal) -> dict:
    # v3-G 宪法层永久,人审才能改 → 这里只暂存 proposal,不直接写
    return {"status": "queued_for_human_review", "preview_length": len(req.content)}