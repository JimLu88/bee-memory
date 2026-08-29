from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, memory, ppr, semantic  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "memories.sqlite"
    for module in (memory, associative, semantic, ppr):
        monkeypatch.setattr(module, "DB_PATH", path)
    semantic.invalidate_cache()
    ppr.invalidate_cache()
    semantic._QCACHE.clear()
    with memory._conn():
        pass
    return path


def test_batch_store_is_idempotent_and_defers_embeddings(db, monkeypatch):
    monkeypatch.setattr(
        semantic,
        "embed_and_store",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("batch must defer")),
    )
    request = memory.BatchStoreRequest(items=[
        memory.StoreRequest(
            kind="knowledge_book",
            content="第一段书籍正文",
            mode_id="book:test",
            meta={"source_id": "books/a#chunk-1"},
        ),
        memory.StoreRequest(
            kind="knowledge_book",
            content="第二段书籍正文",
            mode_id="book:test",
            meta={"source_id": "books/a#chunk-2"},
        ),
    ])

    first = memory.store_batch(request)
    second = memory.store_batch(request)

    assert first["stored"] == 2
    assert first["deduplicated"] == 0
    assert first["embedding_deferred"] is True
    assert second["stored"] == 0
    assert second["deduplicated"] == 2
    assert second["memory_ids"] == first["memory_ids"]
    with memory._conn() as connection:
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM memory_source_keys").fetchone()[0] == 2


def test_single_store_reuses_batch_source_id(db, monkeypatch):
    monkeypatch.setattr(semantic, "embed_and_store", lambda *args, **kwargs: None)
    item = memory.StoreRequest(
        kind="knowledge_book",
        content="可恢复正文",
        meta={"source_id": "books/b#chunk-1"},
    )

    batched = memory.store_batch(memory.BatchStoreRequest(items=[item]))
    repeated = memory.store(item)

    assert repeated["deduplicated"] is True
    assert repeated["memory_id"] == batched["memory_ids"][0]


def test_raw_book_batch_indexes_fts_but_defers_concept_graph(db):
    item = memory.StoreRequest(
        kind="knowledge_book",
        content="《测试书》[1/1]\n供应链交付和质量验收",
        mode_id="book:test",
        meta={
            "source_id": "books/raw#chunk-1",
            "scope": "book_library",
            "tags": ["book", "full_text"],
        },
    )

    result = memory.store_batch(memory.BatchStoreRequest(items=[item]))

    with memory._conn() as connection:
        memory_id = result["memory_ids"][0]
        assert connection.execute(
            "SELECT COUNT(*) FROM memories_fts WHERE memory_id=?", (memory_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM mem_concepts WHERE memory_id=?", (memory_id,)
        ).fetchone()[0] == 0


def test_new_batch_item_skips_full_fts_delete_scan(db):
    statements = []
    with memory._conn() as connection:
        connection.set_trace_callback(statements.append)
        with connection:
            memory._insert_memory(
                connection,
                memory.StoreRequest(
                    kind="semantic",
                    content="有证据的新记忆直接增量写入索引",
                    meta={"source_id": "night-sync/new-item"},
                ),
                now=1_787_880_000,
                embed_now=False,
            )

    normalized = [statement.upper() for statement in statements]
    assert any("INSERT INTO MEMORIES_FTS" in statement for statement in normalized)
    assert not any("DELETE FROM MEMORIES_FTS" in statement for statement in normalized)


def test_new_connection_does_not_repeat_schema_ddl_while_writer_is_active(db):
    """A reader/next writer may connect while another transaction owns WAL."""
    writer = memory._conn()
    try:
        writer.execute("BEGIN IMMEDIATE")
        other = memory._conn()
        try:
            assert other.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        finally:
            other.close()
    finally:
        writer.rollback()
        writer.close()
