"""个性化 PageRank (v4 记忆大脑 P1) — 扩散激活的可扩展形式.

从入口记忆 (向量/FTS 命中) 出发, 沿 edges 图多跳扩散, 找到"问 A 召回 B"的 B.
这就是 HippoRAG / fast-graphrag 的核心: 向量只找入口, 图结构负责多跳关联.

- 只读 edges 表 (记忆↔记忆共现边), 不写任何表.
- scipy 稀疏矩阵, 进程内缓存邻接矩阵 (按边数失效).
- 双端 PPR (bidirectional): 分别从 A/B 播种, 取乘积, 找中间连接者.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"

ALPHA = 0.15   # 重启概率 (阻尼 0.85)
ITERS = 30     # 幂迭代次数

_CACHE: dict[str, Any] = {"edge_n": -1, "ids": None, "index": None, "P": None, "ts": 0.0}


def _load_graph(c: sqlite3.Connection) -> tuple[list[str], dict[str, int], sparse.csr_matrix]:
    """载入 edges → 列归一化转移矩阵 P (P[j,i]=从 i 到 j 的概率). 按边数缓存."""
    edge_n = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    if _CACHE["edge_n"] == edge_n and _CACHE["P"] is not None and (time.time() - _CACHE["ts"] < 300):
        return _CACHE["ids"], _CACHE["index"], _CACHE["P"]
    rows = c.execute("SELECT src, dst, weight FROM edges").fetchall()
    node_set: set[str] = set()
    for s, d, _ in rows:
        node_set.add(s); node_set.add(d)
    ids = sorted(node_set)
    index = {mid: i for i, mid in enumerate(ids)}
    n = len(ids)
    if n == 0:
        P = sparse.csr_matrix((0, 0), dtype=np.float32)
        _CACHE.update({"edge_n": edge_n, "ids": ids, "index": index, "P": P, "ts": time.time()})
        return ids, index, P
    src_i = [index[s] for s, d, w in rows]
    dst_i = [index[d] for s, d, w in rows]
    wts = [float(w or 1.0) for s, d, w in rows]
    # A[dst, src] = w  (列 = 来源, 行 = 去向)
    A = sparse.csr_matrix((wts, (dst_i, src_i)), shape=(n, n), dtype=np.float32)
    # 列归一化 → 转移概率
    colsum = np.asarray(A.sum(axis=0)).ravel()
    colsum[colsum == 0] = 1.0
    D = sparse.diags(1.0 / colsum)
    P = (A @ D).tocsr()
    _CACHE.update({"edge_n": edge_n, "ids": ids, "index": index, "P": P, "ts": time.time()})
    return ids, index, P


def invalidate_cache() -> None:
    _CACHE["edge_n"] = -1


def personalized_pagerank(c: sqlite3.Connection, seeds: list[str],
                          seed_weights: dict[str, float] | None = None) -> dict[str, float]:
    """从 seeds 播种的个性化 PageRank. 返回 {memory_id: score} (只含图中节点)."""
    ids, index, P = _load_graph(c)
    n = len(ids)
    if n == 0:
        return {}
    e = np.zeros(n, dtype=np.float32)
    seen = 0
    for s in seeds:
        if s in index:
            e[index[s]] += (seed_weights or {}).get(s, 1.0)
            seen += 1
    if seen == 0 or e.sum() == 0:
        return {}
    e = e / e.sum()
    r = e.copy()
    for _ in range(ITERS):
        r = (1 - ALPHA) * (P @ r) + ALPHA * e
    return {ids[i]: float(r[i]) for i in np.nonzero(r > 1e-6)[0]}


def connect_ppr(c: sqlite3.Connection, a_seeds: list[str], b_seeds: list[str],
                topn: int = 5) -> dict[str, Any]:
    """双端 PPR 找连接者: 从 A 与 B 分别扩散, 乘积高的节点 = 桥. 比 BFS 更鲁棒."""
    ra = personalized_pagerank(c, a_seeds)
    rb = personalized_pagerank(c, b_seeds)
    if not ra or not rb:
        return {"connected": False, "reason": "seed 不在图中或图空", "connectors": []}
    common = set(ra) & set(rb)
    if not common:
        return {"connected": False, "reason": "两端扩散无交集", "connectors": []}
    scored = sorted(((mid, ra[mid] * rb[mid]) for mid in common), key=lambda x: x[1], reverse=True)
    return {"connected": True, "connectors": [{"memory_id": m, "score": round(s, 8)} for m, s in scored[:topn]]}
