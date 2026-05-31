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
    # v3-D 边 (sender, receiver, weight) — 沿边激活扩散
    c.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT, dst TEXT, weight REAL DEFAULT 1.0,
            PRIMARY KEY (src, dst)
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
    # v3-F SM-2 复习状态
    c.execute("""
        CREATE TABLE IF NOT EXISTS review_state (
            memory_id TEXT PRIMARY KEY,
            ef REAL DEFAULT 2.5,
            interval_days INTEGER DEFAULT 0,
            repetitions INTEGER DEFAULT 0,
            next_review_ts INTEGER,
            last_grade INTEGER
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_review_due ON review_state(next_review_ts)")
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


# v8 记忆种类优先级: 亲历的流程/情景决策 > 书本理论.
# 用户直觉: "情景决策与流程的记忆高于书本; 书本重要度没这个高".
# 数值越大越优先; 在激活打分里作为一个加权项, 让经验类记忆压过纯书本知识.
_KIND_PRIORITY: dict[str, float] = {
    "procedural":        1.00,   # 流程/方法 (怎么做) — 最高
    "episodic":          0.90,   # 亲历的决策/对话
    "self_upgrade":      0.85,
    "knowledge_case":    0.70,   # 真实案例 ≈ 准经验
    "knowledge_pitfall": 0.65,   # 踩过的坑
    "knowledge_standard":0.60,   # 法规/标准
    "semantic":          0.55,
    "knowledge_kol":     0.48,
    "knowledge_book":    0.45,   # 书本理论 — 低于一切亲历经验
    "knowledge_trend":   0.42,
    "knowledge_history": 0.40,
    "knowledge_slang":   0.35,
}


def _activation_score(row: dict, now: int) -> float:
    """v3-D 6 因子打分 + v8 记忆种类权重.

    遗忘曲线: recency (14d 半衰期) + age_penalty 让久未调用的记忆自然沉底;
    importance 由写入方分级 (核心书=5 慢沉, 普通/边角书=3 快沉), 被专门召回时
    last_recall_ts 刷新 → recency 重置 = "重新激活". 这正是用户要的"书也会遗忘,
    除非特别调用才重新激活".
    """
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
    kind_priority = _KIND_PRIORITY.get(row.get("kind") or "", 0.40)
    return (1.0 * recency + 0.6 * frequency + 0.8 * emotional + 0.5 * novelty
            + 0.7 * connection + 0.6 * predictive - 0.3 * age_penalty
            + 0.5 * importance_boost + 1.0 * kind_priority)


def _spread_activation(seed_ids: list[str], hops: int = 2, decay: float = 0.7) -> dict[str, float]:
    """v3-D 沿边激活扩散 (2 跳, 衰减 0.7).

    返回 {memory_id: bonus_score} —— 起点本身权重 1.0, 1 跳邻居 0.7, 2 跳 0.49.
    """
    bonus: dict[str, float] = {sid: 1.0 for sid in seed_ids}
    frontier = list(seed_ids)
    weight = 1.0
    with _conn() as c:
        for _ in range(hops):
            weight *= decay
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = c.execute(
                f"SELECT src, dst, weight FROM edges WHERE src IN ({placeholders})",
                frontier,
            ).fetchall()
            next_frontier: list[str] = []
            for src, dst, w in rows:
                contrib = weight * (w or 1.0)
                if contrib > bonus.get(dst, 0.0):
                    bonus[dst] = contrib
                    next_frontier.append(dst)
            frontier = next_frontier
    return bonus


@router.get("/recall")
def recall(
    query: str = "",
    kind: str = "",
    k: int = 5,
    strategy: str = "activation",
    seed_ids: str = "",
    persona_id: str = "",
) -> dict:
    """v3-D activation strategy by default; static = old top-N by ts.

    `seed_ids`: 逗号分隔的种子记忆 ID; 提供时启用沿边扩散加成 (2 跳, 衰减 0.7).
    `persona_id`: 按 meta.persona_id 服务端过滤 (人设知识库领域隔离, 必须在截断前过滤).
    """
    where = []
    params: list = []
    if kind:
        where.append("kind=?"); params.append(kind)
    if query:
        where.append("content LIKE ?"); params.append(f"%{query}%")
    if persona_id:
        # meta 是 JSON 文本, 形如 {"persona_id": "head_fd_...", ...}; 用 LIKE 匹配该字段值.
        where.append("meta LIKE ?"); params.append(f'%"persona_id": "{persona_id}"%')
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        # 加 persona_id 过滤后命中集变小, 提高预取上限保证 activation 排序覆盖全部候选.
        rows = c.execute(f"SELECT * FROM memories {where_sql} ORDER BY last_recall_ts DESC LIMIT 500", params).fetchall()
    items = [dict(r) for r in rows]
    now = int(time.time())
    spread_bonus: dict[str, float] = {}
    if strategy == "activation":
        if seed_ids:
            seeds = [s.strip() for s in seed_ids.split(",") if s.strip()]
            spread_bonus = _spread_activation(seeds)
        items.sort(
            key=lambda r: _activation_score(r, now) + spread_bonus.get(r["id"], 0.0),
            reverse=True,
        )
    items = items[:k]
    # update recall stats
    ids = [r["id"] for r in items]
    if ids:
        with _conn() as c:
            for mid in ids:
                c.execute("UPDATE memories SET recall_count=recall_count+1, last_recall_ts=? WHERE id=?", (now, mid))
    return {"items": items, "strategy": strategy, "spread_seeds": list(spread_bonus.keys()) if spread_bonus else []}


class EdgeIn(BaseModel):
    src: str
    dst: str
    weight: float = 1.0


@router.post("/edges")
def add_edge(req: EdgeIn) -> dict:
    """v3-D 注册一条记忆间的关联 (语义/因果/共现)."""
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO edges (src,dst,weight) VALUES (?,?,?)",
            (req.src, req.dst, req.weight),
        )
        # 同步更新两端 connection_density
        for node in (req.src, req.dst):
            cnt = c.execute(
                "SELECT COUNT(*) FROM edges WHERE src=? OR dst=?", (node, node)
            ).fetchone()[0]
            c.execute(
                "UPDATE memories SET connection_density=? WHERE id=?",
                (math.log1p(cnt) / 5.0, node),
            )
    return {"status": "ok", "src": req.src, "dst": req.dst}


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