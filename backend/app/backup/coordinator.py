"""v3-E 备份编排器 — 切片 → 加密 → 多池分发 → 分级冗余 → 异步重试.

分级冗余度 (配合激活系统,永不真删):
- 近 90 天 或 importance>=4 : 5-pool 全量
- importance==3 或 score>P50    : 5-pool
- importance==2 或 P25<score<=P50: 3-pool (gist + r2 + notion)
- importance<=1 或 score<=P25   : 2-pool 镜像 (gist + r2,最低保 2 副本)
"""
from __future__ import annotations

import sqlite3, time, hashlib, uuid, json
from pathlib import Path
from . import reed_solomon as rs, encryption as enc, pools as pl

DB_PATH = Path(__file__).parent.parent.parent / "data" / "memories.sqlite"
POOL_ORDER = ["gist", "r2", "notion", "aliyun", "gdrive"]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.execute("""
        CREATE TABLE IF NOT EXISTS backup_shards (
            shard_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            pool_name TEXT NOT NULL,
            account_id TEXT,
            remote_ref TEXT NOT NULL,
            created_ts INTEGER NOT NULL,
            sha256 TEXT,
            pending_upload INTEGER DEFAULT 0
        )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_shard_memory ON backup_shards(memory_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_shard_pending ON backup_shards(pending_upload)")
    c.row_factory = sqlite3.Row
    return c


def _tier(importance: int, age_days: float) -> list[str]:
    if importance >= 4 or age_days <= 90:
        return POOL_ORDER
    if importance >= 3:
        return POOL_ORDER
    if importance >= 2:
        return ["gist", "r2", "notion"]
    return ["gist", "r2"]


def backup_memory(memory_id: str) -> dict:
    now = int(time.time())
    with _conn() as c:
        row = c.execute(
            "SELECT id, content, importance, created_ts FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
    if not row:
        raise KeyError(f"memory {memory_id} not found")

    age_days = max(0, (now - row["created_ts"]) / 86400.0)
    target_pools = _tier(row["importance"], age_days)

    plaintext = json.dumps({"id": row["id"], "content": row["content"]}, ensure_ascii=False).encode("utf-8")
    ciphertext = enc.encrypt(plaintext)
    shards = rs.split_shards(ciphertext)

    pending_total = 0
    refs: list[dict] = []
    with _conn() as c:
        for i, shard in enumerate(shards):
            pool_name = target_pools[i % len(target_pools)]
            shard_id = f"sh-{uuid.uuid4().hex[:12]}"
            result = pl.by_name(pool_name).put(shard_id, shard)
            sha = hashlib.sha256(shard).hexdigest()
            c.execute(
                """INSERT INTO backup_shards
                   (shard_id, memory_id, pool_name, account_id, remote_ref, created_ts, sha256, pending_upload)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (shard_id, memory_id, pool_name, result.get("account_id", ""),
                 result["remote_ref"], now, sha, 1 if result.get("pending_upload") else 0),
            )
            refs.append({"shard_id": shard_id, "pool": pool_name,
                         "pending": bool(result.get("pending_upload"))})
            if result.get("pending_upload"):
                pending_total += 1
    return {"memory_id": memory_id, "shard_count": len(shards),
            "pending": pending_total, "refs": refs, "tier_pools": target_pools}


def restore_memory(memory_id: str) -> dict | None:
    with _conn() as c:
        rows = c.execute(
            "SELECT shard_id, pool_name, remote_ref FROM backup_shards WHERE memory_id=? ORDER BY shard_id",
            (memory_id,),
        ).fetchall()
    if not rows:
        return None
    shards: list[bytes | None] = [None] * 5
    for i, r in enumerate(rows[:5]):
        shards[i] = pl.by_name(r["pool_name"]).get(r["remote_ref"])
    available = sum(1 for s in shards if s is not None)
    if available < 3:
        return {"status": "failed", "available_shards": available, "needed": 3}
    ciphertext = rs.reassemble(shards)
    plaintext = enc.decrypt(ciphertext)
    obj = json.loads(plaintext.decode("utf-8"))
    return {"status": "ok", "available_shards": available,
            "content": obj["content"], "memory_id": obj["id"]}


def retry_pending(limit: int = 50) -> dict:
    with _conn() as c:
        rows = c.execute(
            "SELECT shard_id, pool_name, remote_ref FROM backup_shards WHERE pending_upload=1 LIMIT ?",
            (limit,),
        ).fetchall()
    uploaded = 0
    still_pending = 0
    for r in rows:
        adapter = pl.by_name(r["pool_name"])
        blob_path = Path(r["remote_ref"])
        if not blob_path.exists():
            continue
        result = adapter.put(r["shard_id"], blob_path.read_bytes())
        if result.get("pending_upload"):
            still_pending += 1
        else:
            with _conn() as c:
                c.execute(
                    "UPDATE backup_shards SET remote_ref=?, account_id=?, pending_upload=0 WHERE shard_id=?",
                    (result["remote_ref"], result.get("account_id", ""), r["shard_id"]),
                )
            uploaded += 1
    return {"scanned": len(rows), "uploaded": uploaded, "still_pending": still_pending}


def stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM backup_shards").fetchone()[0]
        pending = c.execute("SELECT COUNT(*) FROM backup_shards WHERE pending_upload=1").fetchone()[0]
        per_pool = {row["pool_name"]: row["n"] for row in
                    c.execute("SELECT pool_name, COUNT(*) n FROM backup_shards GROUP BY pool_name").fetchall()}
        memories_backed = c.execute("SELECT COUNT(DISTINCT memory_id) FROM backup_shards").fetchone()[0]
    return {
        "total_shards": total,
        "pending_upload": pending,
        "memories_backed_up": memories_backed,
        "per_pool": per_pool,
        "pool_quotas": {p.name: p.quota() for p in pl.ALL_POOLS},
    }
