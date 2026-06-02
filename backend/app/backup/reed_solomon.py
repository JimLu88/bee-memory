"""v3-E Reed-Solomon 切片 (5 片中任 3 片可重建).

优先用 `reedsolo` 包 (纯 Python 无原生依赖);未装时降级到 XOR 校验 +
镜像的 (3,5) 简化方案: 3 数据片 + 2 校验片, 任 3 片可重建。
"""
from __future__ import annotations

from typing import Sequence


N = 5  # 总片数
K = 3  # 最少重建片数 (2 片冗余)


def split_shards(data: bytes) -> list[bytes]:
    """切成 N=5 片;任 K=3 片即可重建."""
    if not data:
        raise ValueError("empty data")
    try:
        return _rs_split(data)
    except ImportError:
        return _xor_split(data)


def reassemble(shards: Sequence[bytes | None]) -> bytes:
    """需提供 N 个槽位 (缺片置 None);可用片 ≥ K 即可重建."""
    available = [s for s in shards if s is not None]
    if len(available) < K:
        raise ValueError(f"need at least {K} shards, got {len(available)}")
    try:
        return _rs_reassemble(shards)
    except ImportError:
        return _xor_reassemble(shards)


# ---- reedsolo 路径 (推荐) ----

def _rs_split(data: bytes) -> list[bytes]:
    from reedsolo import RSCodec  # type: ignore
    rsc = RSCodec(N - K)
    encoded = rsc.encode(data)
    shards: list[bytearray] = [bytearray() for _ in range(N)]
    for i, b in enumerate(encoded):
        shards[i % N].append(b)
    header = len(data).to_bytes(8, "big") + b"RS"
    return [header + i.to_bytes(2, "big") + bytes(s) for i, s in enumerate(shards)]


def _rs_reassemble(shards: Sequence[bytes | None]) -> bytes:
    from reedsolo import RSCodec  # type: ignore
    rsc = RSCodec(N - K)
    pieces: dict[int, bytes] = {}
    original_len = 0
    for s in shards:
        if s is None:
            continue
        if s[8:10] != b"RS":
            raise ValueError("not an RS shard")
        original_len = int.from_bytes(s[:8], "big")
        idx = int.from_bytes(s[10:12], "big")
        pieces[idx] = s[12:]
    max_len = max(len(p) for p in pieces.values())
    encoded = bytearray()
    for col in range(max_len):
        for row in range(N):
            p = pieces.get(row)
            if p is not None and col < len(p):
                encoded.append(p[col])
            else:
                encoded.append(0)
    decoded = rsc.decode(bytes(encoded))[0]
    return bytes(decoded[:original_len])


# ---- 降级 XOR 路径 (无 reedsolo) ----

def _xor_split(data: bytes) -> list[bytes]:
    n = len(data)
    pad = (-n) % K
    padded = data + b"\x00" * pad
    chunk = len(padded) // K
    parts = [padded[i * chunk:(i + 1) * chunk] for i in range(K)]
    parity1 = bytes(a ^ b ^ c for a, b, c in zip(*parts))
    parity2 = bytes((a + b + c) & 0xFF for a, b, c in zip(*parts))
    header = n.to_bytes(8, "big") + b"XR"
    return [
        header + (0).to_bytes(2, "big") + parts[0],
        header + (1).to_bytes(2, "big") + parts[1],
        header + (2).to_bytes(2, "big") + parts[2],
        header + (3).to_bytes(2, "big") + parity1,
        header + (4).to_bytes(2, "big") + parity2,
    ]


def _xor_reassemble(shards: Sequence[bytes | None]) -> bytes:
    pieces: dict[int, bytes] = {}
    original_len = 0
    for s in shards:
        if s is None:
            continue
        if s[8:10] != b"XR":
            raise ValueError("not an XOR shard")
        original_len = int.from_bytes(s[:8], "big")
        idx = int.from_bytes(s[10:12], "big")
        pieces[idx] = s[12:]
    data_parts: list[bytes | None] = [pieces.get(i) for i in range(K)]
    missing = [i for i in range(K) if data_parts[i] is None]
    if not missing:
        return b"".join(data_parts)[:original_len]  # type: ignore[arg-type]
    if len(missing) == 1 and 3 in pieces:
        m = missing[0]
        others = [data_parts[i] for i in range(K) if i != m]
        recovered = bytearray(pieces[3])
        for o in others:
            recovered = bytearray(a ^ b for a, b in zip(recovered, o))  # type: ignore[arg-type]
        data_parts[m] = bytes(recovered)
        return b"".join(data_parts)[:original_len]  # type: ignore[arg-type]
    raise ValueError("XOR fallback can only recover 1 missing data chunk")
