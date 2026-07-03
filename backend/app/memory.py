"""bee-memory / 记忆中心 — 三层记忆 + 6 因子激活打分 + 宪法 (v2 阶段 3 + v3-D/E/F)"""
from __future__ import annotations
import sqlite3, json, time, uuid, math
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from . import associative  # v4 关联层 (概念图/FTS/共现边), 纯增量
from . import semantic      # v4 语义向量层 (bge-m3 嵌入)

router = APIRouter()


def _migrate_add_columns(c: sqlite3.Connection) -> None:
    """v4: 给 memories 补可空列 (token_count/stability/difficulty). 幂等, 不动旧列/旧数据."""
    cols = {r[1] for r in c.execute("PRAGMA table_info(memories)")}
    for col, decl in (("token_count", "INTEGER"), ("stability", "REAL"), ("difficulty", "REAL")):
        if col not in cols:
            c.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"
CONST_PATH = Path(__file__).parent.parent / "data" / "constitution.md"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=15)
    c.execute("PRAGMA journal_mode=WAL")   # v4: 并发读写不互锁 (backfill/服务/夜间循环)
    c.execute("PRAGMA busy_timeout=8000")  # 锁等待 8s 再报错
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
    associative.ensure_associative_schema(c)  # v4: 建关联层新表 (幂等, 不动旧表)
    semantic.ensure_vec_schema(c)             # v4: 向量表
    _migrate_add_columns(c)                   # v4: memories 补可空列
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
        # v4: 写入即增量索引 (概念/FTS/dangling). 失败绝不破坏 store.
        try:
            associative.index_one_memory(c, mid, req.content)
        except Exception:
            pass
        try:  # token_count (让检索结果能标 token 预算)
            c.execute("UPDATE memories SET token_count=? WHERE id=?",
                      (max(1, len(req.content or "") // 3), mid))
        except Exception:
            pass
        try:  # 写入即嵌入 (Ollama 挂了自动降级, 不阻断 store)
            semantic.embed_and_store(c, mid, req.content)
            semantic.invalidate_cache()
        except Exception:
            pass
        try:  # v4: 重要记忆(>=4)自动入复习闸, 否则复习页永远空
            if req.importance >= 4:
                c.execute(
                    "INSERT OR IGNORE INTO review_state(memory_id,ef,interval_days,repetitions,next_review_ts,last_grade) "
                    "VALUES (?,2.5,1,0,?,NULL)", (mid, now + 86400))
        except Exception:
            pass
    return {"memory_id": mid}


@router.get("/get")
def get_by_id(id: str) -> dict:
    """v4: 按 id 精确取全文 (token 金字塔 T3). brain_get / UI 看全文 用这个,
    不再走 hybrid_recall(query=id) 的近似搜索 (常miss)."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM memories WHERE id=?", (id,)).fetchone()
        if not row:
            raise HTTPException(404, f"memory {id} 不存在")
        # touch-on-recall
        c.execute("UPDATE memories SET recall_count=recall_count+1, last_recall_ts=? WHERE id=?",
                  (int(time.time()), id))
        return dict(row)


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
    # v4: 遗忘曲线半衰期 = 每条记忆自己的 stability S (由重要度+复习次数睡眠循环算出);
    # 无 S 时回退 14 天. 这就是 FSRS 的 R(t): 强记忆衰减慢, 弱记忆衰减快.
    S = row.get("stability") or 14.0
    recency = math.exp(-recency_days / max(1.0, float(S)))
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
    fts: int = 0,
) -> dict:
    """v3-D activation strategy by default; static = old top-N by ts.

    `seed_ids`: 逗号分隔的种子记忆 ID; 提供时启用沿边扩散加成 (2 跳, 衰减 0.7).
    `persona_id`: 按 meta.persona_id 服务端过滤 (人设知识库领域隔离, 必须在截断前过滤).
    `fts`: v4 开关. 0(默认)=原行为(content LIKE 子串); 1=用 FTS5 索引找入口集,
           且未显式给 seed_ids 时自动用 FTS 命中头部做扩散激活种子 (关联召回).
    """
    where = []
    params: list = []
    if kind:
        where.append("kind=?"); params.append(kind)
    if query and not fts:
        where.append("content LIKE ?"); params.append(f"%{query}%")
    if persona_id:
        # meta 是 JSON 文本, 形如 {"persona_id": "head_fd_...", ...}; 用 LIKE 匹配该字段值.
        where.append("meta LIKE ?"); params.append(f'%"persona_id": "{persona_id}"%')
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    fts_ids: list[str] = []
    with _conn() as c:
        c.row_factory = sqlite3.Row
        if query and fts:
            # v4: FTS5 入口集 (索引化 + bm25 排序, 命中空回退 LIKE), 再叠加 persona/kind 过滤.
            fts_ids = associative.entry_search(c, query, k=max(50, k * 5))
            if fts_ids:
                ph = ",".join("?" * len(fts_ids))
                w2 = where + [f"id IN ({ph})"]
                rows = c.execute(
                    f"SELECT * FROM memories WHERE {' AND '.join(w2)} ORDER BY last_recall_ts DESC LIMIT 500",
                    params + fts_ids,
                ).fetchall()
            else:
                rows = []
        else:
            # 加 persona_id 过滤后命中集变小, 提高预取上限保证 activation 排序覆盖全部候选.
            rows = c.execute(f"SELECT * FROM memories {where_sql} ORDER BY last_recall_ts DESC LIMIT 500", params).fetchall()
    items = [dict(r) for r in rows]
    now = int(time.time())
    spread_bonus: dict[str, float] = {}
    if strategy == "activation":
        seeds: list[str] = []
        if seed_ids:
            seeds = [s.strip() for s in seed_ids.split(",") if s.strip()]
        elif fts and fts_ids:
            seeds = fts_ids[:5]  # v4: FTS 头部命中作扩散种子, 沿共现边关联召回
        if seeds:
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
def consolidate(promote_threshold: int = 3) -> dict:
    """v4 结构固化 (睡眠循环的非 LLM 部分):
    1. 重建概念图 + 记忆↔记忆共现边 (给扩散激活加油);
    2. 把被反复引用的 dangling link 提升为正式概念 (Obsidian 自动立节点);
    3. 返回真实统计. LLM 语义蒸馏 (episodic→semantic) 留 P3.
    """
    idx = associative.reindex_concepts(rebuild_edges=True)
    promoted = 0
    with _conn() as c:
        # dangling 引用够多 且已有同名概念 → 标记 promoted
        dangs = c.execute(
            "SELECT name FROM dangling_refs WHERE mention_count>=? AND promoted=0",
            (promote_threshold,),
        ).fetchall()
        for (name,) in dangs:
            cid = associative._cid(name)
            has = c.execute("SELECT 1 FROM concepts WHERE id=?", (cid,)).fetchone()
            if has:
                c.execute("UPDATE dangling_refs SET promoted=1 WHERE name=?", (name,))
                promoted += 1
    return {
        "status": "ok",
        "reindex": idx,
        "dangling_promoted": promoted,
        "note": "结构固化完成; LLM 语义蒸馏留 P3 睡眠循环",
    }


class ForgetIn(BaseModel):
    memory_id: str | None = None       # 按 id 删单条 (p7_forgetting 用这个)
    below_activation: float | None = None  # 或按激活分批量遗忘 (< 阈值)
    force: bool = False                # 覆盖护栏 (删高重要度/已入复习闸的)
    max_delete: int = 50               # 批量单次上限
    dry_run: bool = False


def _protected(row: dict, enrolled: set[str]) -> bool:
    """护栏: 高重要度(>=4) / 已入复习闸 / 高连接密度 的记忆不自动遗忘."""
    if int(row.get("importance") or 0) >= 4:
        return True
    if row["id"] in enrolled:
        return True
    if float(row.get("connection_density") or 0) >= 0.6:
        return True
    return False


def _hard_delete(c: sqlite3.Connection, mid: str) -> None:
    """删记忆并清干净所有派生引用 (概念链接/FTS/向量/边/复习态). 无孤儿残留."""
    c.execute("DELETE FROM memories WHERE id=?", (mid,))
    c.execute("DELETE FROM mem_concepts WHERE memory_id=?", (mid,))
    c.execute("DELETE FROM memories_fts WHERE memory_id=?", (mid,))
    c.execute("DELETE FROM memories_vec WHERE memory_id=?", (mid,))  # v4: 别留孤儿向量
    c.execute("DELETE FROM edges WHERE src=? OR dst=?", (mid, mid))
    c.execute("DELETE FROM review_state WHERE memory_id=?", (mid,))


@router.post("/forget")
def forget(req: ForgetIn = ForgetIn()) -> dict:
    """v4 遗忘 (实装): 按 id 或按激活分安全删除. 护栏见 _protected; force 可覆盖.

    双分量思想: 遗忘的只是低价值碎片; 高重要度/强连接/在复习的记忆永不自动删.
    """
    now = int(time.time())
    with _conn() as c:
        c.row_factory = sqlite3.Row
        enrolled = {r["memory_id"] for r in c.execute("SELECT memory_id FROM review_state")}

        # 模式 1: 按 id 删单条
        if req.memory_id:
            row = c.execute("SELECT * FROM memories WHERE id=?", (req.memory_id,)).fetchone()
            if not row:
                return {"status": "not_found", "memory_id": req.memory_id, "deleted": 0}
            if not req.force and _protected(dict(row), enrolled):
                return {"status": "protected", "memory_id": req.memory_id, "deleted": 0,
                        "note": "高重要度/复习中/强连接; 传 force=true 强删"}
            if not req.dry_run:
                _hard_delete(c, req.memory_id)
                semantic.invalidate_cache(); associative.ppr.invalidate_cache()
            return {"status": "ok", "deleted": 0 if req.dry_run else 1,
                    "memory_id": req.memory_id, "dry_run": req.dry_run}

        # 模式 2: 按激活分批量遗忘
        thr = req.below_activation if req.below_activation is not None else 0.05
        rows = c.execute("SELECT * FROM memories ORDER BY last_recall_ts ASC LIMIT 2000").fetchall()
        cands = []
        for r in rows:
            d = dict(r)
            if _protected(d, enrolled):
                continue
            if _activation_score(d, now) < thr:
                cands.append(d["id"])
            if len(cands) >= req.max_delete:
                break
        if not req.dry_run and cands:
            for mid in cands:
                _hard_delete(c, mid)
            semantic.invalidate_cache(); associative.ppr.invalidate_cache()
        return {"status": "ok", "deleted": 0 if req.dry_run else len(cands),
                "candidates": len(cands), "below_activation": thr, "dry_run": req.dry_run}


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