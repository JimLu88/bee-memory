"""v3-F Anki SM-2 Spaced Repetition

SuperMemo 2 算法:间隔 1→3→7→21→60→180 天 (受 EF 与 grade 调节).

API:
- GET  /memory/review/due?limit=20         — 列今天到期项
- POST /memory/review/grade                — 用户评分后推进 SM-2 状态
- POST /memory/review/enroll/{memory_id}   — 把已有记忆纳入复习闸
- GET  /memory/review/stats                — 复习闸总览
"""
from __future__ import annotations

import sqlite3, time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DB_PATH = Path(__file__).parent.parent / "data" / "memories.sqlite"

DAY = 86400


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _sm2(ef: float, interval: int, reps: int, grade: int) -> tuple[float, int, int]:
    """SM-2 公式. grade ∈ [0,5]; <3 → 重置."""
    grade = max(0, min(5, grade))
    if grade < 3:
        return max(1.3, ef - 0.2), 1, 0
    reps += 1
    if reps == 1:
        new_interval = 1
    elif reps == 2:
        new_interval = 3
    else:
        new_interval = max(1, round(interval * ef))
    new_ef = max(1.3, ef + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
    return new_ef, new_interval, reps


@router.get("/review/due")
def due(limit: int = 20) -> dict:
    now = int(time.time())
    with _conn() as c:
        rows = c.execute(
            """SELECT m.id, m.kind, m.content, m.importance,
                      r.ef, r.interval_days, r.repetitions, r.next_review_ts
                 FROM memories m
                 JOIN review_state r ON m.id = r.memory_id
                WHERE r.next_review_ts <= ?
             ORDER BY r.next_review_ts ASC
                LIMIT ?""",
            (now, limit),
        ).fetchall()
    return {"items": [dict(r) for r in rows], "now_ts": now}


class GradeIn(BaseModel):
    memory_id: str
    grade: int  # 0-5; <3 = 不会, 3=勉强, 4=会, 5=完全掌握


@router.post("/review/grade")
def grade(req: GradeIn) -> dict:
    now = int(time.time())
    with _conn() as c:
        row = c.execute(
            "SELECT ef, interval_days, repetitions FROM review_state WHERE memory_id=?",
            (req.memory_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, f"memory {req.memory_id} 未进入复习闸")
        ef, interval, reps = _sm2(row["ef"], row["interval_days"], row["repetitions"], req.grade)
        next_ts = now + interval * DAY
        c.execute(
            """UPDATE review_state
                  SET ef=?, interval_days=?, repetitions=?, next_review_ts=?, last_grade=?
                WHERE memory_id=?""",
            (ef, interval, reps, next_ts, req.grade, req.memory_id),
        )
    return {
        "memory_id": req.memory_id,
        "ef": ef,
        "interval_days": interval,
        "repetitions": reps,
        "next_review_ts": next_ts,
    }


class EnrollIn(BaseModel):
    initial_interval_days: int = 1


@router.post("/review/enroll/{memory_id}")
def enroll(memory_id: str, req: EnrollIn = EnrollIn()) -> dict:
    """把已有记忆纳入复习闸 (重要任务/SOP 等)."""
    now = int(time.time())
    with _conn() as c:
        exists = c.execute("SELECT 1 FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not exists:
            raise HTTPException(404, f"memory {memory_id} 不存在")
        c.execute(
            """INSERT OR REPLACE INTO review_state
                   (memory_id, ef, interval_days, repetitions, next_review_ts, last_grade)
               VALUES (?, 2.5, ?, 0, ?, NULL)""",
            (memory_id, req.initial_interval_days, now + req.initial_interval_days * DAY),
        )
    return {"memory_id": memory_id, "next_review_ts": now + req.initial_interval_days * DAY}


@router.get("/review/stats")
def stats() -> dict:
    now = int(time.time())
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM review_state").fetchone()[0]
        due_now = c.execute("SELECT COUNT(*) FROM review_state WHERE next_review_ts<=?", (now,)).fetchone()[0]
        next_7d = c.execute("SELECT COUNT(*) FROM review_state WHERE next_review_ts<=?", (now + 7 * DAY,)).fetchone()[0]
    return {"total_enrolled": total, "due_now": due_now, "due_within_7d": next_7d}
