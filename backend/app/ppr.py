"""个性化 PageRank (v4 记忆大脑 P1) — 扩散激活的可扩展形式.

从入口记忆 (向量/FTS 命中) 出发, 沿 edges 图多跳扩散, 找到"问 A 召回 B"的 B.
这就是 HippoRAG / fast-graphrag 的核心: 向量只找入口, 图结构负责多跳关联.

- 只读 edges 表 (记忆↔记忆共现边), 不写任何表.
- scipy 稀疏矩阵, 进程内缓存邻接矩阵 (按边数失效).
- 双端 PPR (bidirectional): 分别从 A/B 播种, 取乘积, 找中间连接者.
"""
from __future__ import annotations

import sqlite3
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"

ALPHA = 0.15   # 重启概率 (阻尼 0.85)
ITERS = 30     # 幂迭代次数

_CACHE: dict[str, Any] = {
    "edge_n": -1,
    "ids": None,
    "index": None,
    "P": None,
    "ts": 0.0,
    "refresh_requested_at": 0.0,
    "db_path": "",
}
_CACHE_LOCK = threading.RLock()
_BUILD_LOCK = threading.Lock()
_WARM_LOCK = threading.Lock()
_WARMING = False
STALE_REFRESH_SECONDS = max(5.0, float(os.environ.get("BEE_GRAPH_REFRESH_SECONDS", "60")))


def _build_graph(
    c: sqlite3.Connection,
) -> tuple[list[str], dict[str, int], sparse.csr_matrix]:
    """Build one immutable graph snapshot without holding the cache lock."""
    try:
        rows = c.execute("SELECT src, dst, weight FROM edges").fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).casefold():
            raise
        rows = []
    edge_n = len(rows)
    node_set: set[str] = set()
    for source, destination, _ in rows:
        node_set.add(source)
        node_set.add(destination)
    ids = sorted(node_set)
    index = {memory_id: position for position, memory_id in enumerate(ids)}
    n = len(ids)
    if n == 0:
        matrix = sparse.csr_matrix((0, 0), dtype=np.float32)
    else:
        source_indexes = [index[source] for source, _, _ in rows]
        destination_indexes = [index[destination] for _, destination, _ in rows]
        weights = [float(weight or 1.0) for _, _, weight in rows]
        adjacency = sparse.csr_matrix(
            (weights, (destination_indexes, source_indexes)),
            shape=(n, n),
            dtype=np.float32,
        )
        column_sum = np.asarray(adjacency.sum(axis=0)).ravel()
        column_sum[column_sum == 0] = 1.0
        matrix = (adjacency @ sparse.diags(1.0 / column_sum)).tocsr()
    with _CACHE_LOCK:
        _CACHE.update(
            {
                "edge_n": edge_n,
                "ids": ids,
                "index": index,
                "P": matrix,
                "ts": time.time(),
                "refresh_requested_at": 0.0,
                "db_path": str(DB_PATH),
            }
        )
    return ids, index, matrix


def _load_graph(
    c: sqlite3.Connection,
    *,
    force_refresh: bool = False,
) -> tuple[list[str], dict[str, int], sparse.csr_matrix]:
    """Return a graph snapshot using stale-while-revalidate under write load.

    A 100-book import changes the edge count after nearly every batch. Rebuilding
    the complete sparse matrix on the next user request made occasional recalls
    take tens of seconds. Once a valid snapshot exists, requests keep using it
    while one low-priority worker refreshes at a bounded cadence.
    """
    try:
        edge_n = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).casefold():
            raise
        edge_n = 0
    now = time.time()
    schedule_refresh = False
    stale_snapshot = None
    with _CACHE_LOCK:
        same_database = _CACHE.get("db_path") == str(DB_PATH)
        if same_database and _CACHE["edge_n"] == edge_n and _CACHE["P"] is not None:
            return _CACHE["ids"], _CACHE["index"], _CACHE["P"]
        if same_database and _CACHE["P"] is not None and not force_refresh:
            stale_snapshot = (_CACHE["ids"], _CACHE["index"], _CACHE["P"])
            last_attempt = max(
                float(_CACHE.get("ts") or 0.0),
                float(_CACHE.get("refresh_requested_at") or 0.0),
            )
            if now - last_attempt >= STALE_REFRESH_SECONDS:
                _CACHE["refresh_requested_at"] = now
                schedule_refresh = True

    if stale_snapshot is not None:
        if schedule_refresh:
            warm_cache_async(delay_seconds=0.05)
        return stale_snapshot

    with _BUILD_LOCK:
        if not force_refresh:
            with _CACHE_LOCK:
                if (
                    _CACHE.get("db_path") == str(DB_PATH)
                    and _CACHE["edge_n"] == edge_n
                    and _CACHE["P"] is not None
                ):
                    return _CACHE["ids"], _CACHE["index"], _CACHE["P"]
        return _build_graph(c)


def invalidate_cache() -> None:
    with _CACHE_LOCK:
        _CACHE["edge_n"] = -1


def warm_cache_async(delay_seconds: float = 2.0) -> bool:
    """Build the relation matrix off the first user's request path."""
    global _WARMING
    with _WARM_LOCK:
        if _WARMING:
            return False
        _WARMING = True

    def run() -> None:
        global _WARMING
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            conn = sqlite3.connect(str(DB_PATH), timeout=15)
            try:
                conn.execute("PRAGMA busy_timeout=8000")
                _load_graph(conn, force_refresh=True)
            finally:
                conn.close()
        finally:
            with _WARM_LOCK:
                _WARMING = False

    threading.Thread(target=run, name="bee-ppr-warm", daemon=True).start()
    return True


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
