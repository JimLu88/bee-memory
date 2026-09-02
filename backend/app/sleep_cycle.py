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
import sqlite3
import time
import uuid as _uuid
from pathlib import Path
from typing import Any

from . import associative, file_memory_sync, llm, semantic

VAULT_DIR = Path(os.environ.get("BEE_VAULT_DIR", r"D:/AI/AI 记忆中心/vault"))
BACKUP_DIR = Path(os.environ.get("BEE_BACKUP_DIR", r"D:/AI/AI 记忆中心/backups"))
BACKUP_KEEP = int(os.environ.get("BEE_BACKUP_KEEP", "7"))
MOC_MIN_DEGREE = int(os.environ.get("BEE_MOC_MIN_DEGREE", "6"))   # 概念连接数 >= 才建 MOC
MOC_MAX = int(os.environ.get("BEE_MOC_MAX", "50"))               # 单次最多建/更新几个 MOC
VAULT_MAX_CONCEPTS = int(os.environ.get("BEE_VAULT_MAX_CONCEPTS", "800"))


def _update_stability(c) -> int:
    """FSRS 式两分量: stability(存储强度, 只增) = f(importance, recall_count, SM-2复习状态);
    difficulty = 1 - importance/5. 复习过的记忆按 SM-2 的 ef×repetitions 额外增稳 (成功复习=更久不忘).
    只填列, 不删数据. stability 会喂进 memory._activation_score 当遗忘曲线半衰期."""
    rows = c.execute(
        "SELECT m.id, m.importance, m.recall_count, r.ef, r.repetitions "
        "FROM memories m LEFT JOIN review_state r ON m.id=r.memory_id").fetchall()
    n = 0
    for mid, imp, rc, ef, reps in rows:
        imp = imp or 0
        rc = rc or 0
        S = (imp / 5.0) * 30.0 + math.log1p(rc) * 15.0 + 1.0  # 天, 越重要/越常调越久不忘
        if ef and reps:  # 复习闸里的: SM-2 状态贡献 (ef 高 + 复习次数多 → 稳定性显著增长)
            S += (float(ef) - 1.3) * int(reps) * 8.0
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


DISTILL_MAX_CLUSTERS = int(os.environ.get("BEE_DISTILL_MAX", "5"))
DISTILL_SIM = float(os.environ.get("BEE_DISTILL_SIM", "0.55"))


def _norm_bool(v) -> bool:
    """LLM 有时把布尔当字符串返回 ('false'/'否'), 规范化."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是", "y")
    return bool(v)


def _distill_episodics(max_clusters: int = DISTILL_MAX_CLUSTERS, max_src: int = 40) -> dict[str, Any]:
    """P2 经验固化 (CLS 新皮层慢学): 把亲历 episodic/procedural 聚类, LLM 蒸馏成可复用 semantic 笔记,
    建双向溯源边, 源标记 consolidated. 只碰经验类, 绝不碰 book/standard/moc. LLM 不可用则跳过.

    三段分离 (关键: 绝不在持写锁时调 LLM): ①读连接采样+聚类 ②无 DB 连接时逐簇调 LLM ③短写事务逐条落库.
    """
    if not llm.available():
        return {"skipped": "llm 不可用", "distilled": 0}
    from .memory import _conn
    import numpy as np

    # ---- 阶段1: 读 (采样 + 向量 + 聚类), 读完即释放连接 ----
    with _conn() as c:
        rows = c.execute(
            "SELECT id, content, importance FROM memories "
            "WHERE kind IN ('episodic','procedural') AND invalid_at IS NULL "
            "AND (meta IS NULL OR meta NOT LIKE '%\"consolidated\": true%') "
            "ORDER BY created_ts DESC LIMIT ?", (max_src,)).fetchall()
        if not rows:
            return {"distilled": 0, "note": "无待固化经验"}
        ids = [r[0] for r in rows]
        content = {r[0]: (r[1] or "") for r in rows}
        imp = {r[0]: (r[2] or 0) for r in rows}
        ph = ",".join("?" * len(ids))
        vmap: dict[str, Any] = {}
        for mid, dim, blob in c.execute(f"SELECT memory_id,dim,vec FROM memories_vec WHERE memory_id IN ({ph})", ids):
            v = np.frombuffer(blob, dtype=np.float32, count=dim)
            if v.shape[0] == semantic.EMBED_DIM:   # 维度守卫: 混维不参与点积 (防崩溃)
                vmap[mid] = v
    clusters: list[list[str]] = []
    used: set[str] = set()
    for a in ids:
        if a in used or a not in vmap:
            continue
        grp = [a]; used.add(a)
        for b in ids:
            if b in used or b not in vmap:
                continue
            if float(vmap[a] @ vmap[b]) > DISTILL_SIM:
                grp.append(b); used.add(b)
        clusters.append(grp)
    for i in ids:
        if i not in used:
            clusters.append([i]); used.add(i)

    # ---- 阶段2: 无 DB 连接时调 LLM (可能各 120s), 结果攒内存 ----
    pending: list[tuple[list[str], str, str]] = []
    for grp in clusters:
        if len(pending) >= max_clusters:
            break
        if len(grp) < 2 and imp.get(grp[0], 0) < 4:
            continue
        bodies = "\n---\n".join(f"[{i}] {content[i][:600]}" for i in grp)
        data = llm.chat_json(
            "下面是若干条亲历记忆(经历/决策/踩坑)。请提炼成 1 条**可复用的语义知识**"
            "(去掉一次性细节, 留下以后还用得上的规律/口径/教训)。\n"
            '输出 JSON: {"title":"一句话标题", "insight":"120-300字的知识精华", "worth": true/false}\n'
            "worth=false 表示这些内容太琐碎不值得固化。\n记忆:\n" + bodies,
            system="你是记忆固化专家, 只输出合法JSON。")
        if not isinstance(data, dict) or not _norm_bool(data.get("worth")) or not data.get("insight"):
            continue
        title = str(data.get("title") or "")[:80]
        insight = str(data.get("insight"))[:2000]
        pending.append((grp, title, insight))

    # ---- 阶段3: 逐条短写事务落库 (每条独立提交, 不长时持锁) ----
    made = 0
    for grp, title, insight in pending:
        sem_content = f"{title}\n{insight}" if title else insight
        vec = semantic.embed_text(sem_content)  # 嵌在写事务外
        sem_id = "m-" + _uuid.uuid4().hex[:12]
        meta = _json.dumps({"distilled_from": grp, "auto": True, "title": title}, ensure_ascii=False)
        now = int(time.time())
        with _conn() as c:
            c.execute("INSERT INTO memories(id,kind,content,mode_id,importance,created_ts,last_recall_ts,meta) "
                      "VALUES (?,?,?,?,?,?,?,?)", (sem_id, "semantic", sem_content, "", 4, now, now, meta))
            if vec is not None:
                semantic.store_vector(c, sem_id, vec)
            try:
                associative.index_one_memory(c, sem_id, sem_content)
            except Exception:
                pass
            for i in grp:  # 双向溯源边 (kind=provenance, reindex 不会清) + 源标记 consolidated
                c.execute("INSERT OR REPLACE INTO edges(src,dst,weight,kind) VALUES (?,?,3.0,'provenance')", (sem_id, i))
                c.execute("INSERT OR REPLACE INTO edges(src,dst,weight,kind) VALUES (?,?,3.0,'provenance')", (i, sem_id))
                srow = c.execute("SELECT meta FROM memories WHERE id=?", (i,)).fetchone()
                try:
                    m = _json.loads(srow[0] or "{}")
                except Exception:
                    m = {}
                m["consolidated"] = True
                m["consolidated_into"] = sem_id
                c.execute("UPDATE memories SET meta=? WHERE id=?", (_json.dumps(m, ensure_ascii=False), i))
        made += 1
    semantic.invalidate_cache()
    associative.ppr.invalidate_cache()
    return {"distilled": made, "candidates": len(rows)}


TYPED_MAX_PAIRS = int(os.environ.get("BEE_TYPED_MAX", "20"))


def _typed_edges(max_pairs: int = TYPED_MAX_PAIRS) -> dict[str, Any]:
    """P2 类型化边: 对经验类记忆(含蒸馏 semantic)的候选对, LLM 标关系类型+because 理由并存向量.
    候选 = 已有共现边 ∪ embedding kNN(0.4<sim<0.92); 排除蒸馏溯源对(冗余). 三段分离不跨 LLM 持写锁.
    """
    if not llm.available():
        return {"skipped": "llm 不可用", "typed": 0}
    from .memory import _conn
    import struct
    import uuid as _u

    import numpy as np
    # ---- 阶段1: 读 ----
    with _conn() as c:
        rows = c.execute(
            "SELECT id, content, meta FROM memories WHERE kind IN ('episodic','procedural','semantic') "
            "AND invalid_at IS NULL ORDER BY created_ts DESC LIMIT 60").fetchall()
        if len(rows) < 2:
            return {"typed": 0, "note": "经验记忆不足"}
        ids = [r[0] for r in rows]
        content = {r[0]: (r[1] or "") for r in rows}
        # 蒸馏溯源对 (semantic 的 distilled_from 含某源) 不再标类型边 (已有 provenance 边, 冗余)
        prov: set[tuple[str, str]] = set()
        for r in rows:
            try:
                df = _json.loads(r[2] or "{}").get("distilled_from") or []
            except Exception:
                df = []
            for src in df:
                prov.add(tuple(sorted((r[0], src))))
        ph = ",".join("?" * len(ids))
        vmap: dict[str, Any] = {}
        for mid, dim, blob in c.execute(f"SELECT memory_id,dim,vec FROM memories_vec WHERE memory_id IN ({ph})", ids):
            v = np.frombuffer(blob, dtype=np.float32, count=dim)
            if v.shape[0] == semantic.EMBED_DIM:
                vmap[mid] = v
        cand: set[tuple[str, str]] = set()
        for s, d in c.execute(f"SELECT src,dst FROM edges WHERE src IN ({ph})", ids):
            if d in content:
                cand.add(tuple(sorted((s, d))))
        vids = [i for i in ids if i in vmap]
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                sim = float(vmap[vids[i]] @ vmap[vids[j]])
                if 0.4 < sim < 0.92:
                    cand.add(tuple(sorted((vids[i], vids[j]))))
        cand -= prov
        # 已标过的对 (任一方向) 跳过
        todo = []
        for a, b in list(cand)[:max_pairs]:
            if not c.execute("SELECT 1 FROM typed_edges WHERE (src=? AND dst=?) OR (src=? AND dst=?)",
                             (a, b, b, a)).fetchone():
                todo.append((a, b))

    # ---- 阶段2: LLM (无 DB 连接) ----
    results = []
    for a, b in todo:
        data = llm.chat_json(
            f"判断记忆A和B的关系。\nA: {content[a][:400]}\nB: {content[b][:400]}\n"
            '输出 JSON: {"rel":"causes|part_of|contradicts|supports|relates|none",'
            '"direction":"a_to_b|b_to_a|mutual","because":"一句话说明为什么"}\n'
            "rel=none 表示没有实质关系。",
            system="你判断两条知识的关系, 只输出合法JSON。")
        if not isinstance(data, dict):
            continue
        rel = data.get("rel")
        if not isinstance(rel, str) or rel in ("none", "", "relates_none"):
            continue
        because = str(data.get("because") or "")[:300]
        src, dst = (b, a) if data.get("direction") == "b_to_a" else (a, b)
        eb = None
        if because:
            emb = semantic.embed_text(because)
            if emb:
                eb = struct.pack(f"{len(emb)}f", *emb)
        results.append((src, dst, rel[:20], because, eb))

    # ---- 阶段3: 短写事务 ----
    made = 0
    for src, dst, rel, because, eb in results:
        with _conn() as c:
            c.execute("INSERT INTO typed_edges(id,src,dst,rel_type,because,weight,embedding,created_ts) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      ("te-" + _u.uuid4().hex[:10], src, dst, rel, because, 2.0, eb, int(time.time())))
        made += 1
    return {"typed": made, "candidates": len(todo)}


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


STALE_LOCK_SEC = int(os.environ.get("BEE_SLEEP_STALE_SEC", "14400"))  # 4h, 高于最坏单次时长
LOCK_RETRY_DELAYS = (5, 15, 30)


def _acquire_lock() -> tuple[Path, str] | None:
    """单实例锁 (防两次睡眠循环重叠致重复蒸馏/MOC). 拿到返回 (锁路径, 本进程 token), 拿不到返回 None.
    锁内容 = 'ts|uuid'; token 让释放时只删自己那把锁 (别人的活锁不误删). 陈旧 (>STALE) 才接管."""
    from .memory import DB_PATH
    import uuid as _u
    lock = Path(DB_PATH).parent / ".sleep_cycle.lock"
    token = f"{int(time.time())}|{_u.uuid4().hex}"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, token.encode())
        os.close(fd)
        return lock, token
    except FileExistsError:
        try:
            cur = lock.read_text().strip()
            ts = int(cur.split("|", 1)[0]) if cur else 0
            if time.time() - ts < STALE_LOCK_SEC:
                return None  # 有活跃锁, 让出
            lock.unlink()    # 陈旧残留, 接管
            return _acquire_lock()
        except Exception:
            return None


def _release_lock(lock: Path, token: str) -> None:
    """只删属于自己的锁 (内容 == 本进程 token), 绝不误删别人拿到的活锁."""
    try:
        if lock.exists() and lock.read_text().strip() == token:
            lock.unlink()
    except Exception:
        pass


def _consolidate_with_lock_retry(consolidate_fn) -> dict[str, Any]:
    """Retry only transient SQLite writer contention; never replay other failures."""
    attempts = 0
    for delay in (0, *LOCK_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        attempts += 1
        try:
            result = consolidate_fn()
            if isinstance(result, dict):
                result = dict(result)
                result["lock_attempts"] = attempts
            return result
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            transient = "database is locked" in message or "database is busy" in message
            if not transient or attempts >= 1 + len(LOCK_RETRY_DELAYS):
                raise


def _run_step(out: dict[str, Any], name: str, operation) -> Any:
    started = time.monotonic()
    value = operation()
    out[name] = value
    out.setdefault("step_elapsed_s", {})[name] = round(time.monotonic() - started, 3)
    return value


def run_sleep_cycle(do_forget: bool = False, render_vault: bool = True) -> dict[str, Any]:
    """完整睡眠循环. do_forget=False 时只报告遗忘候选不删 (安全默认). 单实例串行."""
    acq = _acquire_lock()
    if acq is None:
        return {"status": "skipped_already_running"}
    lock, _token = acq
    from .memory import _conn, consolidate, forget, ForgetIn
    t0 = time.time()
    out: dict[str, Any] = {}
    current_step = "startup"
    try:
        current_step = "file_sync"
        _run_step(out, current_step, file_memory_sync.sync_file_memories)  # 0. 自动同步文件记忆进大脑
        current_step = "distill"
        _run_step(out, current_step, _distill_episodics)       # 0a episodic→semantic
        current_step = "consolidate"
        _run_step(out, current_step, lambda: _consolidate_with_lock_retry(consolidate))
        current_step = "typed_edges"
        _run_step(out, current_step, _typed_edges)             # 0b 类型化边
        current_step = "backfill"
        _run_step(out, current_step, semantic.backfill)        # 2 补嵌入
        current_step = "stability_and_mocs"
        step_started = time.monotonic()
        with _conn() as c:
            out["stability_updated"] = _update_stability(c)  # 4
            out["mocs"] = _generate_mocs(c)                  # 5
        out.setdefault("step_elapsed_s", {})[current_step] = round(time.monotonic() - step_started, 3)
        if render_vault:
            current_step = "vault"
            step_started = time.monotonic()
            with _conn() as c:
                out["vault"] = _render_vault(c)              # 6
            out.setdefault("step_elapsed_s", {})[current_step] = round(time.monotonic() - step_started, 3)
        current_step = "forget"
        _run_step(out, current_step, lambda: forget(
            ForgetIn(below_activation=0.05, dry_run=not do_forget, max_delete=100)))
        current_step = "backup"
        _run_step(out, current_step, _backup)
        out["status"] = "ok"
        out["elapsed_s"] = round(time.time() - t0, 1)
    except Exception as exc:
        message = str(exc).lower()
        out.update(
            status="failed",
            failed_step=current_step,
            error_type="database_locked" if "database is locked" in message else type(exc).__name__,
            elapsed_s=round(time.time() - t0, 1),
        )
    finally:
        _release_lock(lock, _token)
    return out
