"""睡眠循环 (v4 记忆大脑 P3) — 夜间离线固化, 对应 CLS 的新皮层慢学 + Letta sleep-agent.

每晚跑一次 (调度见 mcp/register_schedule.ps1). 步骤 (全部 LLM-free, 可离线):
1. reindex     重建概念图 + 记忆↔记忆共现边 (给扩散激活加油)
2. backfill    给新记忆补 bge-m3 向量
3. dangling    被反复引用的 [[链接]] 提升为正式概念
4. stability   FSRS 式更新存储强度 S (importance + log(recall_count)); 顺带填 difficulty
5. moc         为高连接概念生成结构笔记 (kind='moc', 可召回的"地图")
6. vault       渲染 Obsidian 兼容 vault (概念/MOC 笔记 + [[链接]] + frontmatter) 到 D 盘
7. forget      低激活非保护记忆的遗忘 (默认 dry_run, 只报告不删)

LLM 语义蒸馏 (episodic→semantic 改写) 需要 LLM, 留作可选增强 (BEE_SLEEP_LLM=1 时接 LiteLLM).
"""
from __future__ import annotations

import json as _json
import math
import os
import time
import uuid as _uuid
from pathlib import Path
from typing import Any

from . import associative, semantic

VAULT_DIR = Path(os.environ.get("BEE_VAULT_DIR", r"D:/AI/AI 记忆中心/vault"))
BACKUP_DIR = Path(os.environ.get("BEE_BACKUP_DIR", r"D:/AI/AI 记忆中心/backups"))
BACKUP_KEEP = int(os.environ.get("BEE_BACKUP_KEEP", "7"))
MOC_MIN_DEGREE = int(os.environ.get("BEE_MOC_MIN_DEGREE", "6"))   # 概念连接数 >= 才建 MOC
MOC_MAX = int(os.environ.get("BEE_MOC_MAX", "50"))               # 单次最多建/更新几个 MOC
VAULT_MAX_CONCEPTS = int(os.environ.get("BEE_VAULT_MAX_CONCEPTS", "800"))


def _update_stability(c) -> int:
    """FSRS 式两分量: stability(存储强度, 只增) = f(importance, recall_count);
    difficulty = 1 - importance/5 (越不重要越"难留"). 只填列, 不删数据."""
    rows = c.execute("SELECT id, importance, recall_count FROM memories").fetchall()
    n = 0
    for mid, imp, rc in rows:
        imp = imp or 0
        rc = rc or 0
        S = (imp / 5.0) * 30.0 + math.log1p(rc) * 15.0 + 1.0  # 天, 越重要/越常调越久不忘
        D = round(1.0 - imp / 5.0, 3)
        c.execute("UPDATE memories SET stability=?, difficulty=? WHERE id=?", (round(S, 2), D, mid))
        n += 1
    return n


def _generate_mocs(c) -> int:
    """为高连接判别性概念生成/更新结构笔记 (MOC). LLM-free: 结构化的成员+邻居清单.

    MOC 本身是一条 kind='moc' 记忆 (可被 recall), content = 该概念的地图. 幂等 upsert.
    """
    deg = {}
    for src, cnt in c.execute("SELECT src, COUNT(*) FROM concept_edges GROUP BY src"):
        deg[src] = cnt
    top = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)
    made = 0
    now = int(time.time())
    for cid, d in top:
        if d < MOC_MIN_DEGREE or made >= MOC_MAX:
            continue
        crow = c.execute("SELECT name, is_hub FROM concepts WHERE id=?", (cid,)).fetchone()
        if not crow or crow[1]:  # 跳过 hub
            continue
        name = crow[0]
        nbrs = c.execute(
            "SELECT c.name, e.weight FROM concept_edges e JOIN concepts c ON e.dst=c.id "
            "WHERE e.src=? ORDER BY e.weight DESC LIMIT 12", (cid,)).fetchall()
        mems = c.execute(
            "SELECT m.id, m.content FROM mem_concepts mc JOIN memories m ON mc.memory_id=m.id "
            "WHERE mc.concept_id=? ORDER BY m.importance DESC LIMIT 8", (cid,)).fetchall()
        lines = [f"# 概念地图: {name}", "", "关联概念: " + " ".join(f"[[{n}]]" for n, _ in nbrs), "", "## 相关记忆"]
        for _mid, content in mems:
            first = (content or "").splitlines()
            lines.append(f"- {(first[0] if first else '')[:70]}")
        moc_content = "\n".join(lines)
        existing = c.execute(
            "SELECT id FROM memories WHERE kind='moc' AND meta LIKE ?",
            (f'%"moc_concept": "{name}"%',)).fetchone()
        meta = _json.dumps({"moc_concept": name, "degree": d, "auto": True}, ensure_ascii=False)
        if existing:
            moc_id = existing[0]
            c.execute("UPDATE memories SET content=?, last_recall_ts=? WHERE id=?", (moc_content, now, moc_id))
        else:
            moc_id = "m-" + _uuid.uuid4().hex[:12]
            c.execute(
                "INSERT INTO memories(id,kind,content,mode_id,importance,created_ts,last_recall_ts,meta) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (moc_id, "moc", moc_content, "", 3, now, now, meta))
        try:  # 同周期即索引 (FTS/概念), 否则要等下次 reindex 才可召回
            associative.index_one_memory(c, moc_id, moc_content)
        except Exception:
            pass
        made += 1
    return made


def _render_vault(c) -> dict[str, int]:
    """渲染 Obsidian 兼容 vault (单向: SQLite 仍是唯一真相). 概念笔记 + MOC + 索引."""
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    (VAULT_DIR / "concepts").mkdir(exist_ok=True)
    (VAULT_DIR / "moc").mkdir(exist_ok=True)
    deg = {src: cnt for src, cnt in c.execute("SELECT src, COUNT(*) FROM concept_edges GROUP BY src")}
    concepts = c.execute("SELECT id, name, doc_freq FROM concepts WHERE is_hub=0").fetchall()
    concepts = sorted(concepts, key=lambda r: deg.get(r[0], 0), reverse=True)[:VAULT_MAX_CONCEPTS]
    written = 0
    used_names: set[str] = set()
    for cid, name, df in concepts:
        nbrs = c.execute(
            "SELECT c.name FROM concept_edges e JOIN concepts c ON e.dst=c.id "
            "WHERE e.src=? ORDER BY e.weight DESC LIMIT 15", (cid,)).fetchall()
        mems = c.execute(
            "SELECT m.content FROM mem_concepts mc JOIN memories m ON mc.memory_id=m.id "
            "WHERE mc.concept_id=? ORDER BY m.importance DESC LIMIT 10", (cid,)).fetchall()
        safe = "".join(ch for ch in name if ch not in '\\/:*?"<>|').strip()[:56] or cid
        if safe in used_names:  # 防不同概念清洗后重名互相覆盖
            safe = f"{safe}_{cid[:6]}"
        used_names.add(safe)
        body = ["---", f"doc_freq: {df}", f"degree: {deg.get(cid, 0)}", "type: concept", "---", "",
                f"# {name}", "", "关联: " + " ".join(f"[[{n}]]" for (n,) in nbrs), "", "## 相关记忆"]
        for (content,) in mems:
            first = (content or "").splitlines()
            body.append(f"- {(first[0] if first else '')[:80]}")
        try:
            (VAULT_DIR / "concepts" / f"{safe}.md").write_text("\n".join(body), encoding="utf-8")
            written += 1
        except Exception:
            pass
    moc_written = 0
    for _mid, content in c.execute("SELECT id, content FROM memories WHERE kind='moc'"):
        head = (content or "").splitlines()
        first = head[0].replace("# 概念地图: ", "").strip() if head else _mid
        safe = "".join(ch for ch in first if ch not in '\\/:*?"<>|').strip()[:60] or _mid
        try:
            (VAULT_DIR / "moc" / f"{safe}.md").write_text(content or "", encoding="utf-8")
            moc_written += 1
        except Exception:
            pass
    idx = ["# 记忆大脑 Vault", "", f"渲染于 unix {int(time.time())}", "",
           f"- 概念笔记: {written} (concepts/)", f"- 概念地图 MOC: {moc_written} (moc/)", "",
           "在 Obsidian 里打开本目录即可浏览整个知识网络 (图谱视图看关联)。"]
    (VAULT_DIR / "README.md").write_text("\n".join(idx), encoding="utf-8")
    return {"concept_notes": written, "moc_notes": moc_written}


def _backup() -> dict[str, Any]:
    """VACUUM INTO 时间戳快照到 D 盘, 保留最近 BACKUP_KEEP 份. (107MB 库此前无自动备份.)

    用独立 autocommit 连接 (VACUUM 不能在事务里). WAL 会一并整合进快照.
    """
    import sqlite3
    from .memory import DB_PATH
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"memories_{ts}.sqlite"
    try:
        conn = sqlite3.connect(str(DB_PATH), isolation_level=None, timeout=30)
        try:
            conn.execute("VACUUM INTO ?", (str(dst),))
        finally:
            conn.close()
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}
    # 大小体检: 快照明显小于源库 → 疑似异常, 保留不轮换删, 报 suspicious
    try:
        src_sz = DB_PATH.stat().st_size
        dst_sz = dst.stat().st_size
    except Exception:
        src_sz = dst_sz = 0
    suspicious = src_sz > 1_000_000 and dst_sz < src_sz * 0.5
    files = sorted(BACKUP_DIR.glob("memories_*.sqlite"))
    if not suspicious:
        for old in files[:-BACKUP_KEEP]:
            try:
                old.unlink()
            except Exception:
                pass
    return {"ok": True, "file": dst.name, "size_mb": round(dst_sz / 1e6, 1),
            "kept": min(len(files), BACKUP_KEEP), "suspicious": suspicious}


def run_sleep_cycle(do_forget: bool = False, render_vault: bool = True) -> dict[str, Any]:
    """完整睡眠循环. do_forget=False 时只报告遗忘候选不删 (安全默认)."""
    from .memory import _conn, consolidate, forget, ForgetIn
    t0 = time.time()
    out: dict[str, Any] = {}
    out["consolidate"] = consolidate()            # 1+3 reindex + dangling promote
    out["backfill"] = semantic.backfill()         # 2 补嵌入
    with _conn() as c:
        out["stability_updated"] = _update_stability(c)  # 4
        out["mocs"] = _generate_mocs(c)                  # 5
    if render_vault:
        with _conn() as c:
            out["vault"] = _render_vault(c)              # 6
    out["forget"] = forget(ForgetIn(below_activation=0.05, dry_run=not do_forget, max_delete=100))  # 7
    out["backup"] = _backup()                                                                       # 8 快照
    out["status"] = "ok"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out
