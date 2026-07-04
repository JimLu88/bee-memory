"""v5 测试 — 文件记忆自动同步 (幂等: 未变跳过, 变了替换, 新增插入)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, file_memory_sync, memory, ppr, semantic  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    dbp = tmp_path / "m.sqlite"
    for mod in (memory, associative, semantic, ppr):
        monkeypatch.setattr(mod, "DB_PATH", dbp)
    monkeypatch.setattr(semantic, "embed_text", lambda t, timeout=None: None)  # 不连 Ollama
    memdir = tmp_path / "memory"
    memdir.mkdir()
    monkeypatch.setattr(file_memory_sync, "FILEMEM_DIR", str(memdir))
    with memory._conn():
        pass
    return memdir


def _write(memdir, fname, name, mtype, body):
    (memdir / fname).write_text(
        f"---\nname: {name}\ndescription: 摘要{name}\nmetadata:\n  type: {mtype}\n---\n\n{body}",
        encoding="utf-8")


def _count_file_mem():
    with memory._conn() as c:
        return c.execute("SELECT COUNT(*) FROM memories WHERE meta LIKE '%\"source\": \"file_memory\"%'").fetchone()[0]


def test_sync_add_then_idempotent(env):
    _write(env, "project_a.md", "project-a", "project", "甲项目内容")
    _write(env, "feedback_b.md", "fb-b", "feedback", "乙反馈内容")
    (env / "MEMORY.md").write_text("index", encoding="utf-8")  # 索引不导入
    r1 = file_memory_sync.sync_file_memories()
    assert r1["added"] == 2 and _count_file_mem() == 2
    with memory._conn() as c:
        kinds = dict(c.execute("SELECT mode_id, kind FROM memories WHERE meta LIKE '%file_memory%'").fetchall())
    assert kinds["project-a"] == "episodic" and kinds["fb-b"] == "procedural"
    r2 = file_memory_sync.sync_file_memories()
    assert r2["added"] == 0 and r2["unchanged"] == 2 and _count_file_mem() == 2


def test_sync_update_on_change(env):
    _write(env, "project_a.md", "project-a", "project", "甲项目内容 v1")
    file_memory_sync.sync_file_memories()
    _write(env, "project_a.md", "project-a", "project", "甲项目内容 v2 改了")
    r = file_memory_sync.sync_file_memories()
    assert r["updated"] == 1 and _count_file_mem() == 1
    with memory._conn() as c:
        content = c.execute("SELECT content FROM memories WHERE mode_id='project-a'").fetchone()[0]
    assert "v2 改了" in content


def test_sync_skips_when_dir_missing(env, monkeypatch):
    monkeypatch.setattr(file_memory_sync, "FILEMEM_DIR", str(env / "nonexistent"))
    r = file_memory_sync.sync_file_memories()
    assert r.get("synced", 0) == 0 and "skipped" in r
