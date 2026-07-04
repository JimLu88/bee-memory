"""文件记忆 → 记忆大脑 自动同步 (v5).

用户的真实记忆由 Claude 记忆系统自动写成 .md 文件 (~/.claude/.../memory/*.md);
本模块把这些文件幂等地同步进记忆大脑, 由夜间睡眠循环自动跑 → 用户无需手动 brain_store.

- 只读那些 .md (不改任何 C 盘文件), 写入只碰记忆大脑 DB (D 盘).
- 幂等: 按 meta.name 定位, 内容哈希没变则跳过, 变了则替换 (删旧插新), 新文件则插入.
- type→kind: feedback→procedural, project→episodic, user/reference→semantic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

# 默认指向 Claude Code 的记忆目录 (C 盘, 只读); 可用 env 覆盖或置空以禁用.
FILEMEM_DIR = os.environ.get(
    "BEE_FILEMEM_DIR", r"C:\Users\lzdwy\.claude\projects\C--Users-lzdwy\memory")

KIND_MAP = {"feedback": "procedural", "project": "episodic",
            "user": "semantic", "reference": "semantic"}


def _parse(md: str) -> tuple[str, str, str, str]:
    name = desc = mtype = ""
    body = md
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", md, re.S)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                desc = line.split(":", 1)[1].strip().strip('"')
            elif re.match(r"\s*type:", line):
                mtype = line.split(":", 1)[1].strip()
    return name, desc, mtype, body.strip()


def sync_file_memories() -> dict[str, Any]:
    """把 FILEMEM_DIR 下的 .md 幂等同步进记忆大脑. 目录不存在则跳过 (非本机部署无副作用)."""
    d = Path(FILEMEM_DIR)
    if not FILEMEM_DIR or not d.exists():
        return {"skipped": "no filemem dir", "synced": 0}
    from .memory import StoreRequest, _conn, _hard_delete, store

    added = updated = skipped = 0
    for p in sorted(d.glob("*.md")):
        if p.name == "MEMORY.md":
            continue
        try:
            name, desc, mtype, body = _parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = name or p.stem
        content = (f"{desc}\n\n{body}" if desc else body)[:6000]
        chash = hashlib.md5(content.encode("utf-8")).hexdigest()
        # 找已同步的同名记忆
        with _conn() as c:
            rows = c.execute(
                "SELECT id, meta FROM memories WHERE meta LIKE ?",
                (f'%"name": "{slug}"%',)).fetchall()
        existing = []
        for rid, meta in rows:
            try:
                mj = json.loads(meta or "{}")
            except Exception:
                continue
            if mj.get("source") == "file_memory" and mj.get("name") == slug:
                existing.append((rid, mj))
        if existing:
            if existing[0][1].get("content_hash") == chash:
                skipped += 1
                continue  # 未变, 跳过
            with _conn() as c:   # 变了: 删旧代 (清干净派生)
                for rid, _ in existing:
                    _hard_delete(c, rid)
            updated += 1
        else:
            added += 1
        store(StoreRequest(
            kind=KIND_MAP.get(mtype, "semantic"), content=content, importance=5, mode_id=slug,
            meta={"source": "file_memory", "name": slug, "type": mtype,
                  "title": slug, "content_hash": chash}))
    return {"synced": added + updated, "added": added, "updated": updated,
            "unchanged": skipped, "dir": str(d)}
