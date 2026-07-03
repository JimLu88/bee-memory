"""语义向量层 (v4 记忆大脑 P1) — bge-m3 嵌入 + 暴力余弦检索.

- 嵌入: 本地 Ollama bge-m3 (1024 维, 中英双语强, 免费, 模型在 D 盘).
- 存储: 新表 memories_vec(memory_id, dim, vec BLOB), vec = 归一化 float32 字节.
- 检索: numpy 暴力余弦 (个人级万~十万条毫秒级), 带进程内矩阵缓存 (按行数失效).
- 全部为新增, 不动 memories/edges 等旧表. Ollama 不可用时静默降级 (返回空向量).

为何暴力而非 sqlite-vec: 个人库规模下 numpy 一次矩阵乘即够快, 零原生扩展加载风险,
更好移植. 未来上百万条再换 sqlite-vec / HNSW.
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("BEE_EMBED_MODEL", "bge-m3")
EMBED_DIM = int(os.environ.get("BEE_EMBED_DIM", "1024"))
EMBED_TIMEOUT = float(os.environ.get("BEE_EMBED_TIMEOUT", "30"))
EMBED_STORE_TIMEOUT = float(os.environ.get("BEE_EMBED_STORE_TIMEOUT", "5"))  # 写入路径短超时, 防阻塞

_QCACHE: dict[str, list[float]] = {}  # 查询嵌入小缓存 (省重复 Ollama 往返)


def ensure_vec_schema(c: sqlite3.Connection) -> None:
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories_vec (
            memory_id TEXT PRIMARY KEY,
            dim INTEGER,
            vec BLOB,
            embedded_ts INTEGER
        )""")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=15)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=8000")
    ensure_vec_schema(c)
    return c


# ---- 嵌入 (Ollama bge-m3) ----
def embed_text(text: str, timeout: float | None = None) -> list[float] | None:
    """返回归一化后的向量; Ollama 挂了或空文本返回 None (调用方降级)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=json.dumps({"model": EMBED_MODEL, "prompt": text[:8000]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or EMBED_TIMEOUT) as r:
            emb = json.loads(r.read()).get("embedding")
        if not emb:
            return None
        v = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.tolist()
    except Exception:
        return None


def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """批量嵌入 (Ollama /api/embed 支持 input 数组). 失败逐条降级."""
    texts = [(t or "").strip()[:8000] for t in texts]
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed",
            data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT * 2) as r:
            embs = json.loads(r.read()).get("embeddings") or []
        out: list[list[float] | None] = []
        for e in embs:
            v = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append((v / n).tolist() if n > 0 else None)
        while len(out) < len(texts):
            out.append(None)
        return out
    except Exception:
        return [embed_text(t) for t in texts]  # 回退逐条


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def store_vector(c: sqlite3.Connection, memory_id: str, vec: list[float]) -> None:
    c.execute(
        "INSERT OR REPLACE INTO memories_vec(memory_id, dim, vec, embedded_ts) VALUES (?,?,?,?)",
        (memory_id, len(vec), _pack(vec), int(time.time())),
    )


def embed_and_store(c: sqlite3.Connection, memory_id: str, content: str) -> bool:
    """写入即嵌入 (供 memory.store 调用). 短超时防阻塞. 成功 True."""
    vec = embed_text(content, timeout=EMBED_STORE_TIMEOUT)
    if vec is None:
        return False
    store_vector(c, memory_id, vec)
    return True


def embed_query(text: str) -> list[float] | None:
    """带小缓存的查询嵌入 (同一查询在检索/连接/桥接里会被多次嵌)."""
    key = (text or "").strip()
    if not key:
        return None
    if key in _QCACHE:
        return _QCACHE[key]
    v = embed_text(key)
    if v is not None:
        if len(_QCACHE) > 256:
            _QCACHE.clear()
        _QCACHE[key] = v
    return v


# ---- 检索 (numpy 暴力余弦 + 进程内缓存) ----
_CACHE: dict[str, Any] = {"n": -1, "ids": None, "mat": None, "ts": 0.0}


def _load_matrix(c: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    """载入全部向量为矩阵. 按行数缓存, 变了才重载 (避免每次查询读 50MB)."""
    n = c.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
    if _CACHE["n"] == n and _CACHE["mat"] is not None and (time.time() - _CACHE["ts"] < 300):
        return _CACHE["ids"], _CACHE["mat"]
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for mid, dim, blob in c.execute("SELECT memory_id, dim, vec FROM memories_vec"):
        v = _unpack(blob, dim)
        if v.shape[0] != EMBED_DIM:   # 维度守卫: 混维(换模型/残留/损坏)不进矩阵, 否则 vstack 崩溃拖垮全检索
            continue
        ids.append(mid)
        vecs.append(v)
    mat = np.vstack(vecs) if vecs else np.zeros((0, EMBED_DIM), dtype=np.float32)
    _CACHE.update({"n": n, "ids": ids, "mat": mat, "ts": time.time()})
    return ids, mat


def invalidate_cache() -> None:
    _CACHE["n"] = -1


def bridge(c: sqlite3.Connection, a: str, b: str, k: int = 5,
           min_sim: float = 0.35) -> list[tuple[str, float]]:
    """语义桥: 同时与 A 和 B 都相近的记忆 (回答 'A 和 B 有什么关系' 的鲁棒兜底).

    按 min(sim_A, sim_B) 排 — 要求对两者都相近, 而非只近一端. 低于 min_sim 的丢弃.
    """
    qa, qb = embed_query(a), embed_query(b)
    if qa is None or qb is None:
        return []
    ids, mat = _load_matrix(c)
    if mat.shape[0] == 0:
        return []
    sa = mat @ np.asarray(qa, dtype=np.float32)
    sb = mat @ np.asarray(qb, dtype=np.float32)
    score = np.minimum(sa, sb)
    order = np.argsort(-score)
    out: list[tuple[str, float]] = []
    for i in order:
        if score[i] < min_sim:
            break
        out.append((ids[i], float(score[i])))
        if len(out) >= k:
            break
    return out


def vector_search(c: sqlite3.Connection, query: str, k: int = 20,
                  candidate_ids: set[str] | None = None) -> list[tuple[str, float]]:
    """返回 [(memory_id, cosine)] 前 k. query 嵌不出或库空 → []. 向量已归一化, 点积=余弦.

    candidate_ids: 若给, 只在该集合内排 (persona/kind 预过滤后再语义排序).
    """
    qv = embed_query(query)
    if qv is None:
        return []
    ids, mat = _load_matrix(c)
    if mat.shape[0] == 0:
        return []
    q = np.asarray(qv, dtype=np.float32)
    sims = mat @ q  # (N,) 余弦
    # argpartition 取候选 topN 再精排 (比全排序快, N 大时明显)
    want = k if candidate_ids is None else mat.shape[0]
    if want < mat.shape[0]:
        cand = np.argpartition(-sims, want)[:want]
        order = cand[np.argsort(-sims[cand])]
    else:
        order = np.argsort(-sims)
    out: list[tuple[str, float]] = []
    for i in order:
        mid = ids[i]
        if candidate_ids is not None and mid not in candidate_ids:
            continue
        out.append((mid, float(sims[i])))
        if len(out) >= k:
            break
    return out


# ---- 回填 (后台批量嵌入尚无向量的记忆) ----
def backfill(limit: int = 0, batch: int = 16) -> dict[str, Any]:
    """给尚无向量的记忆补嵌入. limit=0 全量. 幂等, 可反复跑续传."""
    t0 = time.time()
    done = 0
    failed = 0
    with _conn() as c:
        q = ("SELECT m.id, m.content FROM memories m "
             "LEFT JOIN memories_vec v ON m.id=v.memory_id WHERE v.memory_id IS NULL")
        if limit > 0:
            q += f" LIMIT {int(limit)}"
        rows = c.execute(q).fetchall()
        total = len(rows)
        for i in range(0, total, batch):
            chunk = rows[i:i + batch]
            embs = embed_batch([r[1] or "" for r in chunk])
            for (mid, _), vec in zip(chunk, embs):
                if vec is None:
                    failed += 1
                    continue
                store_vector(c, mid, vec)
                done += 1
            c.commit()
    invalidate_cache()
    return {"status": "ok", "embedded": done, "failed": failed,
            "remaining_before": total, "elapsed_s": round(time.time() - t0, 1)}


def stats() -> dict[str, Any]:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        vec = c.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
    return {"memories": total, "embedded": vec, "coverage_pct": (100 * vec // total) if total else 0,
            "model": EMBED_MODEL, "dim": EMBED_DIM}
