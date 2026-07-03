"""v5 分片冗余 — 复制式 (replication), 取代原来坏掉的纠删码.

原实现 (interleave + zero-fill RS / 单奇偶 XOR) 经复审证实**不满足**"任 3/5 可重建":
XOR 路只能补 1 个数据片且必须留着 parity1; RS 路在丢片时静默返回乱码。
个人级记忆条目很小, 复制的存储开销可忽略, 却换来更强更简单的保证:

  **每片 = 完整密文的一份拷贝; 任何 1 片存活即可完整恢复.**

分发到 N 个独立池 (gist/webdav/notion/gitee/gitlab), 只要有一个池还在就能还原。
"""
from __future__ import annotations

from typing import Sequence

N = 5  # 副本数 (分发到 N 个池)
K = 1  # 最少可恢复副本数 (任 1 份即可)


def split_shards(data: bytes) -> list[bytes]:
    """复制成 N 份完整副本 (每份都是完整密文)."""
    if not data:
        raise ValueError("empty data")
    return [bytes(data) for _ in range(N)]


def reassemble(shards: Sequence[bytes | None]) -> bytes:
    """任取一份存活副本即为完整密文."""
    for s in shards:
        if s:
            return bytes(s)
    raise ValueError(f"need at least {K} shard, got 0 available")
