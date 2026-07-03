"""关联层 (v4 记忆大脑 P0) — 概念图 + FTS5 检索 + 记忆↔记忆共现边.

设计原则 (兼容承诺): 全部为新增, 不改动 memory.py 已有的
memories / edges / review_state 表结构与默认行为.

补的三处断路:
1. **图是空的** — 现有 `_spread_activation` 走 edges(记忆↔记忆), 但从没有人写过边.
   这里从"共享概念"生成记忆↔记忆边 (KAG 互索引思想), 给扩散激活加油.
2. **检索只认子串** — 加 FTS5(trigram) 索引, 词法检索可排序 (BM25) 且索引化.
3. **概念无处落** — concepts/mem_concepts/concept_edges 三表承载 L2 概念图;
   dangling_refs 承载 Obsidian 式"未解析链接一等公民"(先记引用, 节点后成型).

护栏 (防 12k 书本记忆共享通用概念导致 O(n²) 边爆炸):
- hub 概念 (文档频率 > BEE_HUB_DF_RATIO 的全库占比) 不参与共现建边 (不判别);
- 每条记忆最多留 BEE_MAX_EDGES_PER_MEM 条最强邻居;
- 只有共享判别性概念数 ≥ BEE_MIN_SHARED 才连边;
- 单概念桶超过 BEE_BUCKET_CAP 的配对被跳过并计数上报 (不静默截断).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from . import ppr, semantic  # v4 P1: 语义向量 + 个性化 PageRank

router = APIRouter()

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"

# ---- 护栏参数 (env 可调) ----
HUB_DF_RATIO = float(os.environ.get("BEE_HUB_DF_RATIO", "0.05"))   # 概念出现在 >5% 记忆里 = hub, 不建边
HUB_DF_MIN = int(os.environ.get("BEE_HUB_DF_MIN", "200"))          # 或绝对 >200 条也算 hub
HUB_DF_FLOOR = int(os.environ.get("BEE_HUB_DF_FLOOR", "8"))        # 但至少要出现 8 次才可能算 hub (防小库误判)
MAX_EDGES_PER_MEM = int(os.environ.get("BEE_MAX_EDGES_PER_MEM", "8"))
MIN_SHARED = int(os.environ.get("BEE_MIN_SHARED", "2"))
BUCKET_CAP = int(os.environ.get("BEE_BUCKET_CAP", "60"))
MAX_ENTITIES = int(os.environ.get("BEE_MAX_ENTITIES", "15"))

# ---- 实体抽取 (jieba 中文分词优先, 无 jieba 时降级 regex+2gram) ----
_CH_RUN = re.compile(r"[一-龥]{2,}")
_CH_NOUN = re.compile(r"[一-龥]{2,6}")
_EN_NOUN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b")
_WIKILINK = re.compile(r"\[\[([^\]|]{1,40})(?:\|[^\]]*)?\]\]")

# 通用停用概念 (中文常见虚高词, 避免噪音节点)
_STOP = {
    "可以", "没有", "这个", "那个", "什么", "怎么", "因为", "所以", "但是",
    "如果", "已经", "自己", "他们", "我们", "你们", "现在", "时候", "问题",
    "方法", "情况", "需要", "进行", "通过", "由于", "对于", "一个", "一些",
    "这样", "那样", "并且", "以及", "或者", "不是", "就是", "还是", "这些",
    "那些", "然后", "还有", "而且", "只是", "一直", "一定", "非常", "比较",
}

# jieba: 中文分词标准库 (纯 Python, 离线). 缓存写 D 盘 (遵守 C 盘禁写).
try:
    import jieba  # type: ignore
    jieba.setLogLevel(60)  # 静默
    try:
        jieba.dt.tmp_dir = str(DB_PATH.parent)  # cache 落 data/ 而非系统 temp
    except Exception:
        pass
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def extract_wikilinks(text: str) -> list[str]:
    """抽 [[概念]] 显式链接 (Claude 写记忆时用). 显式链接权重更高."""
    if not text:
        return []
    return [m.group(1).strip() for m in _WIKILINK.finditer(text) if m.group(1).strip()]


def _zh_tokens(text: str) -> list[str]:
    """中文分词. 有 jieba 用之 (词更准), 否则退化: regex 2-6 + 每个长串的 2-gram."""
    if _HAS_JIEBA:
        out = []
        for tok in jieba.cut(text):
            tok = tok.strip()
            if 2 <= len(tok) <= 8 and re.fullmatch(r"[一-龥]+", tok) and tok not in _STOP:
                out.append(tok)
        return out
    # 降级: regex 词 + 2-gram (捕捉被切碎的常用词如 '利润')
    out = []
    for m in _CH_NOUN.finditer(text):
        s = m.group(0)
        if not s.isdigit() and s not in _STOP:
            out.append(s)
    for run in _CH_RUN.findall(text):
        for i in range(len(run) - 1):
            g = run[i:i + 2]
            if g not in _STOP:
                out.append(g)
    return out


def extract_entities(text: str) -> list[str]:
    """抽概念候选: [[wikilink]] + 中文分词词 + 英文标识符. 保序去重."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for name in extract_wikilinks(text):
        seen.setdefault(name, None)
    for tok in _zh_tokens(text):
        seen.setdefault(tok, None)
    for m in _EN_NOUN.finditer(text):
        seen.setdefault(m.group(0), None)
    return list(seen.keys())


# ---- schema (全部 IF NOT EXISTS, 幂等) ----
def ensure_associative_schema(c: sqlite3.Connection) -> None:
    """建关联层新表. 由 memory.py 的 _conn() 在建完旧表后调用."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,          -- 'c-<name hash>' 稳定, name 归一后 md5 前 12
            name TEXT UNIQUE,
            kind TEXT DEFAULT 'entity',   -- entity | wikilink
            mention_count INTEGER DEFAULT 0,
            doc_freq INTEGER DEFAULT 0,   -- 出现在多少条记忆 (判 hub 用)
            created_ts INTEGER,
            is_hub INTEGER DEFAULT 0
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)")
    c.execute("""
        CREATE TABLE IF NOT EXISTS mem_concepts (
            memory_id TEXT, concept_id TEXT, weight REAL DEFAULT 1.0,
            PRIMARY KEY (memory_id, concept_id)
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mc_concept ON mem_concepts(concept_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mc_memory ON mem_concepts(memory_id)")
    c.execute("""
        CREATE TABLE IF NOT EXISTS concept_edges (
            src TEXT, dst TEXT, weight REAL DEFAULT 1.0, edge_type TEXT DEFAULT 'cooccur',
            PRIMARY KEY (src, dst)
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ce_src ON concept_edges(src)")
    # v5 P2: 类型化边 (记忆↔记忆), 带关系类型 + because 理由 + because 向量 (关系可检索, LightRAG 双层)
    c.execute("""
        CREATE TABLE IF NOT EXISTS typed_edges (
            id TEXT PRIMARY KEY, src TEXT, dst TEXT,
            rel_type TEXT, because TEXT, weight REAL DEFAULT 1.0,
            embedding BLOB, created_ts INTEGER
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_te_src ON typed_edges(src)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_te_dst ON typed_edges(dst)")
    # Obsidian 式未解析链接: 先记引用, 积累够了自动立节点
    c.execute("""
        CREATE TABLE IF NOT EXISTS dangling_refs (
            name TEXT PRIMARY KEY, mention_count INTEGER DEFAULT 0,
            first_ts INTEGER, last_ts INTEGER, promoted INTEGER DEFAULT 0
        )""")
    # FTS5 词法索引 (trigram 支持 CJK 子串). 独立表, memory_id 存 UNINDEXED 列.
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED, content, tokenize='trigram'
        )""")
    # v4 P1: 给 memories 补可空列 (token_count/stability/difficulty). 幂等, 不动旧列.
    # 放这里保证任何连接 (associative._conn / memory._conn) 都先迁移, 避免 SELECT * 缺列.
    if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone():
        cols = {r[1] for r in c.execute("PRAGMA table_info(memories)")}
        # v4 P1: token_count/stability/difficulty; v5 P2 双时序: invalid_at(失效时刻)/superseded_by(被谁取代)
        for col, decl in (("token_count", "INTEGER"), ("stability", "REAL"), ("difficulty", "REAL"),
                          ("invalid_at", "INTEGER"), ("superseded_by", "TEXT")):
            if col not in cols:
                c.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
    # v5 P2: edges 加 kind (cooccur|provenance|supersede), 让 reindex 只清 cooccur, 保住溯源/取代边
    if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='edges'").fetchone():
        ecols = {r[1] for r in c.execute("PRAGMA table_info(edges)")}
        if "kind" not in ecols:
            c.execute("ALTER TABLE edges ADD COLUMN kind TEXT DEFAULT 'cooccur'")


def _cid(name: str) -> str:
    import hashlib
    return "c-" + hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()[:12]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=15)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=8000")
    c.row_factory = sqlite3.Row
    ensure_associative_schema(c)
    return c


# ---- 增量: 单条记忆写入时索引 (供 memory.store 调用) ----
def index_one_memory(c: sqlite3.Connection, memory_id: str, content: str) -> int:
    """把一条记忆的概念/FTS/dangling 落库. 返回抽到的概念数. 幂等 (可重复调)."""
    now = int(time.time())
    # FTS
    c.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
    c.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?,?)", (memory_id, content or ""))
    # dangling: 显式 [[link]] 单独记 (可能还没有对应概念节点)
    wikilinks = set(extract_wikilinks(content))
    for name in wikilinks:
        c.execute(
            """INSERT INTO dangling_refs(name, mention_count, first_ts, last_ts)
               VALUES (?,1,?,?)
               ON CONFLICT(name) DO UPDATE SET mention_count=mention_count+1, last_ts=?""",
            (name, now, now, now),
        )
    # 概念
    ents = extract_entities(content)[:MAX_ENTITIES]
    old = {r["concept_id"] for r in c.execute(
        "SELECT concept_id FROM mem_concepts WHERE memory_id=?", (memory_id,))}
    for name in ents:
        cid = _cid(name)
        kind = "wikilink" if name in wikilinks else "entity"
        c.execute(
            """INSERT INTO concepts(id,name,kind,mention_count,doc_freq,created_ts)
               VALUES (?,?,?,1,0,?)
               ON CONFLICT(id) DO UPDATE SET mention_count=mention_count+1""",
            (cid, name, kind, now),
        )
        if cid not in old:
            c.execute(
                "INSERT OR IGNORE INTO mem_concepts(memory_id,concept_id,weight) VALUES (?,?,?)",
                (memory_id, cid, 2.0 if kind == "wikilink" else 1.0),
            )
    return len(ents)


# ---- 批量: 全库重建概念图 + 记忆↔记忆共现边 (核心断路修复) ----
def reindex_concepts(rebuild_edges: bool = True, limit: int = 0) -> dict[str, Any]:
    """扫全库记忆, 重建概念/FTS/共现边. 幂等. limit>0 时只处理最近 limit 条 (调试用).

    共现边写入**现有 edges 表**(记忆↔记忆), 双向, 让 memory._spread_activation 有边可走.
    """
    t0 = time.time()
    stats: dict[str, Any] = {"scanned": 0, "concepts": 0, "mem_concept_links": 0,
                             "cooccur_edges": 0, "concept_edges": 0, "hub_concepts": 0,
                             "buckets_capped": 0, "fts_rows": 0}
    with _conn() as c:
        q = "SELECT id, content, kind FROM memories WHERE invalid_at IS NULL"  # 失效记忆不进概念图/FTS
        if limit > 0:
            q += f" ORDER BY last_recall_ts DESC LIMIT {int(limit)}"
        rows = c.execute(q).fetchall()
        stats["scanned"] = len(rows)
        if not rows:
            return {**stats, "status": "empty", "elapsed_s": round(time.time() - t0, 2)}

        # 全量重建概念表: 清空派生表. edges 只清 cooccur (保住 provenance/supersede 边), 防陈边永久累积.
        c.execute("DELETE FROM concepts")
        c.execute("DELETE FROM mem_concepts")
        c.execute("DELETE FROM concept_edges")
        c.execute("DELETE FROM memories_fts")
        c.execute("DELETE FROM edges WHERE kind='cooccur' OR kind IS NULL")

        # pass 1: 抽实体 + 建倒排索引 concept -> [memory_ids].
        # 内容去重: 12k 书库里同一本书按 persona 存了 ~81 份, 内容全同. FTS 仍逐行建(每行可搜),
        # 但概念图/边只用**每种内容一个代表**, 否则 hub 判定被污染、边爆炸. 保留所有行给 persona 召回.
        import hashlib as _hl
        inverted: dict[str, list[str]] = defaultdict(list)
        concept_name: dict[str, str] = {}
        concept_kind: dict[str, str] = {}
        now = int(time.time())
        rep_of_hash: dict[str, str] = {}
        for r in rows:
            mid, content = r["id"], (r["content"] or "")
            c.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?,?)", (mid, content))
            # MOC 是派生的导航索引 (概念地图), 只建 FTS 供召回, 不再抽概念 (否则"地图/概念"污染概念图)
            if r["kind"] == "moc":
                continue
            chash = _hl.md5(content.strip().encode("utf-8")).hexdigest()
            if chash in rep_of_hash:
                continue  # 非代表: 只建 FTS, 不进概念图 (内容与代表完全相同)
            rep_of_hash[chash] = mid
            wl = set(extract_wikilinks(content))
            for name in extract_entities(content)[:MAX_ENTITIES]:
                cid = _cid(name)
                concept_name[cid] = name
                concept_kind[cid] = "wikilink" if name in wl else "entity"
                inverted[cid].append(mid)
        stats["fts_rows"] = len(rows)
        stats["unique_contents"] = len(rep_of_hash)
        stats["concepts"] = len(inverted)

        # pass 2: 落概念表 + mem_concepts + 判 hub. total = 去重后的代表数 (hub 比例才准).
        total = len(rep_of_hash)
        hub_ids: set[str] = set()
        for cid, mids in inverted.items():
            df = len(set(mids))
            # hub = 出现太广的通用概念 (不判别). 需同时够高比例且过绝对下限, 或超硬上限.
            is_hub = 1 if ((df > total * HUB_DF_RATIO and df >= HUB_DF_FLOOR) or df > HUB_DF_MIN) else 0
            if is_hub:
                hub_ids.add(cid)
            c.execute(
                """INSERT INTO concepts(id,name,kind,mention_count,doc_freq,created_ts,is_hub)
                   VALUES (?,?,?,?,?,?,?)""",
                (cid, concept_name[cid], concept_kind[cid], len(mids), df, now, is_hub),
            )
            for mid in set(mids):
                w = 2.0 if concept_kind[cid] == "wikilink" else 1.0
                c.execute("INSERT OR IGNORE INTO mem_concepts(memory_id,concept_id,weight) VALUES (?,?,?)",
                          (mid, cid, w))
        stats["hub_concepts"] = len(hub_ids)
        stats["mem_concept_links"] = c.execute("SELECT COUNT(*) FROM mem_concepts").fetchone()[0]

        # 记忆↔记忆共现: 判别性概念桶内两两 +1 共享; 桶太大跳过配对 (计数上报)
        pair_w: dict[tuple[str, str], float] = defaultdict(float)
        mem_concepts_disc: dict[str, set[str]] = defaultdict(set)
        for cid, mids in inverted.items():
            if cid in hub_ids:
                continue
            umids = list(set(mids))
            if len(umids) > BUCKET_CAP:
                stats["buckets_capped"] += 1
                continue
            for i in range(len(umids)):
                for j in range(i + 1, len(umids)):
                    a, b = umids[i], umids[j]
                    key = (a, b) if a < b else (b, a)
                    pair_w[key] += 1.0
            for mid in umids:
                mem_concepts_disc[mid].add(cid)

        # concept↔concept 共现 (判别性概念在同一记忆内共同出现)
        cc_w: dict[tuple[str, str], float] = defaultdict(float)
        for mid, cids in mem_concepts_disc.items():
            cl = sorted(cids)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    cc_w[(cl[i], cl[j])] += 1.0
        for (a, b), w in cc_w.items():
            c.execute("INSERT OR REPLACE INTO concept_edges(src,dst,weight,edge_type) VALUES (?,?,?,'cooccur')", (a, b, w))
            c.execute("INSERT OR REPLACE INTO concept_edges(src,dst,weight,edge_type) VALUES (?,?,?,'cooccur')", (b, a, w))
        stats["concept_edges"] = len(cc_w) * 2

        if rebuild_edges:
            # 每条记忆只留最强 MAX_EDGES_PER_MEM 条邻居, 且共享 >= MIN_SHARED
            by_mem: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for (a, b), w in pair_w.items():
                if w >= MIN_SHARED:
                    by_mem[a].append((b, w))
                    by_mem[b].append((a, w))
            written: set[tuple[str, str]] = set()
            edge_count = 0
            deg: dict[str, int] = defaultdict(int)
            for mid, nbrs in by_mem.items():
                nbrs.sort(key=lambda x: x[1], reverse=True)
                for nb, w in nbrs[:MAX_EDGES_PER_MEM]:
                    for s, d in ((mid, nb), (nb, mid)):
                        if (s, d) in written:
                            continue
                        # OR IGNORE: 若该对已有 provenance/supersede 边则不覆盖 (它们更权威)
                        c.execute("INSERT OR IGNORE INTO edges(src,dst,weight,kind) VALUES (?,?,?,'cooccur')", (s, d, w))
                        written.add((s, d))
                        deg[s] += 1
                        edge_count += 1
            stats["cooccur_edges"] = edge_count
            # 更新 connection_density (与 memory.add_edge 同公式)
            import math
            for node, cnt in deg.items():
                c.execute("UPDATE memories SET connection_density=? WHERE id=?",
                          (math.log1p(cnt) / 5.0, node))

    ppr.invalidate_cache()  # 边变了, PPR 邻接矩阵缓存失效
    stats["status"] = "ok"
    stats["elapsed_s"] = round(time.time() - t0, 2)
    return stats


# ---- FTS 检索 ----
def _fts_query(query: str) -> str:
    """把自然语言 query 转成 trigram FTS5 表达式. trigram 需 >=3 字的连续片段.
    取 query 里 >=3 长度的中英片段作 OR 短语; 都太短则退化为整串短语."""
    q = (query or "").strip()
    if not q:
        return ""
    frags = re.findall(r"[一-龥]{3,}|[A-Za-z0-9]{3,}", q)
    if not frags:
        return '"' + q.replace('"', '') + '"'
    return " OR ".join('"' + f.replace('"', '') + '"' for f in frags)


def fts_search(c: sqlite3.Connection, query: str, k: int = 20) -> list[str]:
    """返回 FTS5 命中的 memory_id, 按 bm25 排序. query 空则返回 []."""
    expr = _fts_query(query)
    if not expr:
        return []
    try:
        rows = c.execute(
            "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?",
            (expr, k),
        ).fetchall()
        return [r[0] for r in rows]  # 位置取值: 不依赖 row_factory (调用方连接可能是 tuple 工厂)
    except sqlite3.OperationalError:
        return []


def entry_search(c: sqlite3.Connection, query: str, k: int = 20) -> list[str]:
    """入口检索: FTS5 优先, 命中为空则回退 LIKE 子串. 保证不比纯 LIKE 差.

    (trigram 对 <3 字的 CJK 查询无能为力, LIKE 兜底; 二者都索引/全扫得到结果.)
    """
    ids = fts_search(c, query, k)
    if ids:
        return ids
    if not query:
        return []
    rows = c.execute(
        "SELECT id FROM memories WHERE content LIKE ? ORDER BY last_recall_ts DESC LIMIT ?",
        (f"%{query}%", k),
    ).fetchall()
    return [r[0] for r in rows]


# ---- A-B 连接 (双端播种找连接者): 走记忆↔记忆 edges 的 BFS 相遇 ----
def connect_path(c: sqlite3.Connection, a_ids: list[str], b_ids: list[str],
                 max_hops: int = 3) -> dict[str, Any]:
    """从 A 集与 B 集双向 BFS, 找最短相遇路径. 返回连接记忆链 (id 序列) 或空."""
    if not a_ids or not b_ids:
        return {"connected": False, "reason": "empty seed"}
    a_set, b_set = set(a_ids), set(b_ids)
    if a_set & b_set:
        common = list(a_set & b_set)[0]
        return {"connected": True, "hops": 0, "path": [common], "connector": common}

    def neighbors(node: str) -> list[str]:
        return [r[0] for r in c.execute("SELECT dst FROM edges WHERE src=?", (node,))]

    frontier = list(a_set)
    parent: dict[str, str | None] = {x: None for x in a_set}
    for _ in range(max_hops):
        nxt = []
        for node in frontier:
            for nb in neighbors(node):
                if nb in parent:
                    continue
                parent[nb] = node
                if nb in b_set:
                    path = [nb]
                    p: str | None = node
                    while p is not None:
                        path.append(p)
                        p = parent.get(p)
                    path.reverse()
                    return {"connected": True, "hops": len(path) - 1, "path": path,
                            "connector": path[len(path) // 2]}
                nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    return {"connected": False, "reason": f"no path within {max_hops} hops"}


# ---- 端点 ----
class ReindexIn(BaseModel):
    rebuild_edges: bool = True
    limit: int = 0


@router.post("/reindex-concepts")
def reindex_endpoint(req: ReindexIn = ReindexIn()) -> dict:
    """全库重建概念图 + FTS + 记忆↔记忆共现边. p6_graph_rebuild / 夜间循环调这个."""
    return reindex_concepts(rebuild_edges=req.rebuild_edges, limit=req.limit)


@router.post("/fts-rebuild")
def fts_rebuild_endpoint() -> dict:
    """只重建 FTS 索引 (不动概念/边). 快."""
    t0 = time.time()
    with _conn() as c:
        c.execute("DELETE FROM memories_fts")
        n = 0
        for r in c.execute("SELECT id, content FROM memories").fetchall():
            c.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?,?)",
                      (r["id"], r["content"] or ""))
            n += 1
    return {"status": "ok", "fts_rows": n, "elapsed_s": round(time.time() - t0, 2)}


@router.get("/concepts")
def list_concepts(limit: int = 30, hub: int = -1) -> dict:
    """列概念 (按提及量). hub=1 只看 hub, hub=0 只看判别性, -1 全部."""
    with _conn() as c:
        where = "" if hub < 0 else f"WHERE is_hub={int(hub)}"
        rows = c.execute(
            f"SELECT name, kind, mention_count, doc_freq, is_hub FROM concepts {where} "
            f"ORDER BY mention_count DESC LIMIT ?", (limit,)).fetchall()
        total = c.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    return {"total": total, "items": [dict(r) for r in rows]}


@router.get("/connect")
def connect_endpoint(a: str, b: str, k: int = 8, max_hops: int = 3) -> dict:
    """回答 'A 和 B 有什么关系': 用 FTS 找 A/B 各自的入口记忆, 再双端 BFS 找连接链."""
    with _conn() as c:
        a_ids = entry_search(c, a, k)
        b_ids = entry_search(c, b, k)
        res = connect_path(c, a_ids, b_ids, max_hops=max_hops)
        if res.get("connected") and res.get("path"):
            snippets = {}
            for mid in res["path"]:
                row = c.execute("SELECT content FROM memories WHERE id=?", (mid,)).fetchone()
                if row:
                    snippets[mid] = (row["content"] or "")[:80]
            res["path_snippets"] = snippets
    res["a_entry_count"] = len(a_ids)
    res["b_entry_count"] = len(b_ids)
    return res


@router.get("/dangling")
def list_dangling(min_mentions: int = 1, limit: int = 50) -> dict:
    """列未解析链接 (mention 越多越该立节点)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT name, mention_count, promoted FROM dangling_refs WHERE mention_count>=? "
            "ORDER BY mention_count DESC LIMIT ?", (min_mentions, limit)).fetchall()
    return {"items": [dict(r) for r in rows]}


# ============ P1: 混合检索 (向量+FTS入口 → PPR扩散 → 排序) ============
def _title_snippet(content: str) -> tuple[str, str]:
    """从内容取标题 (首行/前 40 字) 和摘要 (前 90 字)."""
    content = (content or "").strip()
    first = content.split("\n", 1)[0].strip()
    title = first[:40] if first else content[:40]
    return title, content[:90]


def hybrid_recall(query: str, k: int = 8, persona_id: str = "", kind: str = "",
                  compact: bool = True, boost_mode: str = "") -> dict[str, Any]:
    """向量(语义)+FTS(字面) 找入口 → 个性化 PageRank 沿共现边扩散 → 综合排序.

    综合分 = 0.5*语义相似 + 0.3*PPR扩散(归一) + 0.2*激活分(归一). 这一步负责"问A召回B".
    compact=True: 只回 {id,title,snippet,score,token_count,via}; False: 回完整行.
    """
    from .memory import _activation_score  # 延迟导入避免循环
    now = int(time.time())
    with _conn() as c:
        c.row_factory = sqlite3.Row
        # persona/kind 预过滤集
        allowed: set[str] | None = None
        if persona_id or kind:
            w, p = [], []
            if kind:
                w.append("kind=?"); p.append(kind)
            if persona_id:
                w.append("meta LIKE ?"); p.append(f'%"persona_id": "{persona_id}"%')
            allowed = {r[0] for r in c.execute(f"SELECT id FROM memories WHERE {' AND '.join(w)}", p)}
            if not allowed:
                return {"items": [], "note": "persona/kind 无匹配"}

        # 入口: 向量 topN + FTS topN
        vec_hits = semantic.vector_search(c, query, k=max(30, k * 4), candidate_ids=allowed)
        vec_map = {mid: sim for mid, sim in vec_hits}
        fts_ids = [i for i in fts_search(c, query, k=max(30, k * 4)) if allowed is None or i in allowed]
        entry = list(dict.fromkeys(list(vec_map.keys())[:15] + fts_ids[:15]))

        # PPR 扩散
        ppr_scores = ppr.personalized_pagerank(c, entry) if entry else {}
        max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0

        # 候选池 = 入口 ∪ PPR 可达头部 (PPR 可能触达数千条, 取 top 防 SQLite 变量上限 999).
        POOL_CAP = 400
        top_ppr = [mid for mid, _ in sorted(ppr_scores.items(), key=lambda kv: kv[1], reverse=True)[:POOL_CAP]]
        pool = set(entry) | set(top_ppr)
        if allowed is not None:
            pool &= allowed
        if not pool:
            pool = set(entry_search(c, query, k))
        pool = set(list(pool)[:POOL_CAP])
        if not pool:
            return {"items": [], "note": "无命中"}

        ph = ",".join("?" * len(pool))
        # 双时序: 默认只回仍有效的 (invalid_at IS NULL). 被取代的旧事实不出现在常规召回.
        rows = c.execute(f"SELECT * FROM memories WHERE id IN ({ph}) AND invalid_at IS NULL",
                         list(pool)).fetchall()
        # 激活分归一
        acts = {r["id"]: _activation_score(dict(r), now) for r in rows}
        max_act = max(acts.values()) if acts else 1.0

        scored = []
        for r in rows:
            mid = r["id"]
            vs = vec_map.get(mid, 0.0)
            ps = (ppr_scores.get(mid, 0.0) / max_ppr) if max_ppr else 0.0
            ac = (acts[mid] / max_act) if max_act else 0.0
            final = 0.5 * vs + 0.3 * ps + 0.2 * ac
            if boost_mode and r["mode_id"] == boost_mode:  # 编码特异性: 同项目/同语境的记忆提分
                final += 0.08
            if r["meta"] and '"consolidated": true' in r["meta"]:  # 已蒸馏的源: 略降, 让语义笔记优先
                final -= 0.05
            via = []
            if mid in vec_map:
                via.append("语义")
            if mid in fts_ids:
                via.append("字面")
            if mid not in entry and mid in ppr_scores:
                via.append("关联扩散")
            scored.append((final, r, vs, ps, "+".join(via) or "关联"))
        scored.sort(key=lambda x: x[0], reverse=True)
        # 去重: 同标题只留最高分 (12k 书库有跨 persona 的近重复)
        deduped = []
        seen_titles: set[str] = set()
        for tup in scored:
            t, _ = _title_snippet(dict(tup[1]).get("content"))
            if t in seen_titles:
                continue
            seen_titles.add(t)
            deduped.append(tup)
            if len(deduped) >= k:
                break
        # touch-on-recall
        for _, r, _, _, _ in deduped:
            c.execute("UPDATE memories SET recall_count=recall_count+1, last_recall_ts=? WHERE id=?", (now, r["id"]))

        items = []
        for final, r, vs, ps, via in deduped:
            rd = dict(r)
            title, snip = _title_snippet(rd.get("content"))
            src = ""  # 溯源: 书名/作者 (从 meta), 让召回可引用
            try:
                m = json.loads(rd.get("meta") or "{}")
                src = str(m.get("title") or m.get("author") or "").strip()
            except Exception:
                pass
            if compact:
                items.append({"id": rd["id"], "title": title, "snippet": snip,
                              "score": round(final, 4), "token_count": rd.get("token_count"),
                              "kind": rd.get("kind"), "importance": rd.get("importance"),
                              "source": src, "via": via})
            else:
                rd.update({"score": round(final, 4), "via": via, "title": title, "source": src})
                items.append(rd)
    return {"items": items, "entry_count": len(entry), "ppr_reached": len(ppr_scores),
            "semantic": bool(vec_map)}


@router.get("/recall-hybrid")
def recall_hybrid_endpoint(query: str, k: int = 8, persona_id: str = "", kind: str = "",
                           compact: int = 1, boost_mode: str = "") -> dict:
    """P1 混合检索端点 (语义+字面+关联扩散). 默认紧凑返回 (省 token). boost_mode=同项目提分."""
    return hybrid_recall(query, k=k, persona_id=persona_id, kind=kind,
                         compact=bool(compact), boost_mode=boost_mode)


def _rich_seeds(c, query: str, k: int = 12) -> list[str]:
    """连接用的种子: 语义(向量) + 字面 并集, 排除失效记忆. 比纯 LIKE 更可能命中有边的节点."""
    vec = [m for m, _ in semantic.vector_search(c, query, k=k)]
    seeds = list(dict.fromkeys(vec + entry_search(c, query, k)))
    if not seeds:
        return seeds
    ph = ",".join("?" * len(seeds))
    valid = {r[0] for r in c.execute(
        f"SELECT id FROM memories WHERE id IN ({ph}) AND invalid_at IS NULL", seeds)}
    return [s for s in seeds if s in valid]


@router.get("/connect2")
def connect_ppr_endpoint(a: str, b: str, k: int = 12) -> dict:
    """P1 双端 PPR 找连接者 (比 BFS 鲁棒): 回答 'A 和 B 有什么关系'."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        a_ids = _rich_seeds(c, a, k)
        b_ids = _rich_seeds(c, b, k)
        # 直接类型化关系 (最强信号: A 侧记忆与 B 侧记忆之间已有 LLM 标注的关系)
        a_set, b_set = set(a_ids), set(b_ids)
        typed_rels = []
        allset = list(a_set | b_set)
        if len(allset) >= 2:
            ph = ",".join("?" * len(allset))
            for t in c.execute(f"SELECT src,dst,rel_type,because FROM typed_edges "
                               f"WHERE src IN ({ph}) AND dst IN ({ph})", allset + allset):
                if (t["src"] in a_set and t["dst"] in b_set) or (t["src"] in b_set and t["dst"] in a_set):
                    typed_rels.append({"rel": t["rel_type"], "because": t["because"]})
        res = ppr.connect_ppr(c, a_ids, b_ids, topn=5)
        res["method"] = "ppr"
        if typed_rels:
            res["typed_relations"] = typed_rels[:3]
        if not res.get("connected"):
            # 兜底: 语义桥 — 同时与 A 和 B 都相近的记忆 (图谱按主题聚簇时跨簇连接靠这个)
            bridges = semantic.bridge(c, a, b, k=5)
            if bridges:
                res = {"connected": True, "method": "semantic_bridge",
                       "connectors": [{"memory_id": m, "score": round(s, 4)} for m, s in bridges]}
        if res.get("connected"):
            for conn in res["connectors"]:
                row = c.execute("SELECT content FROM memories WHERE id=?", (conn["memory_id"],)).fetchone()
                conn["snippet"] = (row["content"][:90] if row else "")
    res["a_entry_count"] = len(a_ids)
    res["b_entry_count"] = len(b_ids)
    return res


class VecBackfillIn(BaseModel):
    limit: int = 0
    batch: int = 16


@router.post("/vec/backfill")
def vec_backfill_endpoint(req: VecBackfillIn = VecBackfillIn()) -> dict:
    """给尚无向量的记忆补 bge-m3 嵌入 (可反复跑续传). limit=0 全量."""
    return semantic.backfill(limit=req.limit, batch=req.batch)


@router.get("/vec/stats")
def vec_stats_endpoint() -> dict:
    """向量覆盖率."""
    return semantic.stats()


@router.post("/sleep-cycle")
def sleep_cycle_endpoint(do_forget: int = 0, render_vault: int = 1) -> dict:
    """P3 睡眠循环: reindex+补嵌入+dangling提升+stability+MOC+vault渲染+遗忘(默认dry_run).

    夜间调度调这个 (register_schedule.ps1). do_forget=0 只报告遗忘候选不删.
    """
    from . import sleep_cycle
    return sleep_cycle.run_sleep_cycle(do_forget=bool(do_forget), render_vault=bool(render_vault))


class InvalidateIn(BaseModel):
    memory_id: str


@router.post("/invalidate")
def invalidate_endpoint(req: InvalidateIn) -> dict:
    """双时序: 把一条记忆标为失效 (不删, 历史仍可查). 用于口径/决策过时."""
    now = int(time.time())
    with _conn() as c:
        r = c.execute("UPDATE memories SET invalid_at=? WHERE id=? AND invalid_at IS NULL",
                      (now, req.memory_id))
        ok = r.rowcount > 0
    return {"status": "ok" if ok else "noop", "memory_id": req.memory_id, "invalid_at": now}


class SupersedeIn(BaseModel):
    old_id: str
    new_id: str


@router.post("/supersede")
def supersede_endpoint(req: SupersedeIn) -> dict:
    """双时序: 新事实取代旧事实. 旧的打 invalid_at + superseded_by=新id, 并建一条取代边."""
    now = int(time.time())
    with _conn() as c:
        exists = c.execute("SELECT 1 FROM memories WHERE id=?", (req.new_id,)).fetchone()
        if not exists:
            return {"status": "error", "detail": "new_id 不存在"}
        c.execute("UPDATE memories SET invalid_at=?, superseded_by=? WHERE id=?",
                  (now, req.new_id, req.old_id))
        # 取代边 (新 -> 旧, kind=supersede 不被 reindex 清), 便于回溯"我当时以为什么"
        c.execute("INSERT OR REPLACE INTO edges(src,dst,weight,kind) VALUES (?,?,?,'supersede')",
                  (req.new_id, req.old_id, 3.0))
    ppr.invalidate_cache()
    return {"status": "ok", "old_id": req.old_id, "new_id": req.new_id, "invalid_at": now}


@router.get("/digest")
def digest_endpoint(project: str = "", limit: int = 8) -> dict:
    """P2 会话启动摘要 (硬顶极小 token): 统计 + top 重要记忆一句话.

    project 给定时优先召回该项目相关记忆; 否则给全局最重要/最新的.
    供 SessionStart 钩子与 MCP brain_digest 调用.
    """
    now = int(time.time())
    with _conn() as c:
        c.row_factory = sqlite3.Row
        counts = {
            "memories": c.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "edges": c.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
            "concepts": c.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
            "embedded": c.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0],
            "due_review": c.execute("SELECT COUNT(*) FROM review_state WHERE next_review_ts<=?", (now,)).fetchone()[0],
        }
        top = []
        if project:
            top = hybrid_recall(project, k=limit, compact=True).get("items", [])
        if not top:
            rows = c.execute(
                "SELECT id, content, kind, importance, token_count FROM memories "
                "WHERE invalid_at IS NULL ORDER BY importance DESC, last_recall_ts DESC LIMIT ?", (limit,)).fetchall()
            for r in rows:
                title, snip = _title_snippet(r["content"])
                top.append({"id": r["id"], "title": title, "snippet": snip,
                            "kind": r["kind"], "importance": r["importance"],
                            "token_count": r["token_count"]})
        # themes: 知识网络的主干概念 (按图上连接度) + 各自邻居, 答"有哪些主题"
        themes = []
        deg = {src: cnt for src, cnt in c.execute("SELECT src, COUNT(*) FROM concept_edges GROUP BY src")}
        for cid in sorted(deg, key=lambda x: deg[x], reverse=True)[:12]:
            crow = c.execute("SELECT name, is_hub FROM concepts WHERE id=?", (cid,)).fetchone()
            if not crow or crow["is_hub"]:
                continue
            nbrs = [r["name"] for r in c.execute(
                "SELECT c.name FROM concept_edges e JOIN concepts c ON e.dst=c.id "
                "WHERE e.src=? ORDER BY e.weight DESC LIMIT 5", (cid,)).fetchall()]
            themes.append({"concept": crow["name"], "degree": deg[cid], "related": nbrs})
            if len(themes) >= 6:
                break
    return {"counts": counts, "top": top, "themes": themes, "project": project}
