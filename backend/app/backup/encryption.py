"""v3-E 本地密钥 + 对称加密 — 云端只存密文.

优先用 cryptography.Fernet (AES-128-CBC + HMAC);未装则降级到
HMAC-SHA256 keystream + HMAC tag 的零依赖方案。
"""
from __future__ import annotations

import os, hmac, hashlib, secrets
from pathlib import Path

KEY_PATH = Path(__file__).parent.parent.parent / "data" / ".backup_key"


def _ensure_key() -> bytes:
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    KEY_PATH.write_bytes(key)
    try:
        os.chmod(KEY_PATH, 0o600)
    except OSError:
        pass
    return key


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:length])


def encrypt(plaintext: bytes) -> bytes:
    """Returns: prefix || (FERNET token  OR  nonce(16) || ct || hmac_tag(32))."""
    try:
        from cryptography.fernet import Fernet  # type: ignore
        import base64
        key = _ensure_key()
        fkey = base64.urlsafe_b64encode(key)
        return b"FERNET:" + Fernet(fkey).encrypt(plaintext)
    except ImportError:
        key = _ensure_key()
        nonce = secrets.token_bytes(16)
        ks = _keystream(key, nonce, len(plaintext))
        ct = bytes(p ^ k for p, k in zip(plaintext, ks))
        tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
        return b"HMACS:" + nonce + ct + tag


def decrypt(blob: bytes) -> bytes:
    key = _ensure_key()
    if blob.startswith(b"FERNET:"):
        from cryptography.fernet import Fernet  # type: ignore
        import base64
        fkey = base64.urlsafe_b64encode(key)
        return Fernet(fkey).decrypt(blob[7:])
    if not blob.startswith(b"HMACS:"):
        raise ValueError("unknown ciphertext format")
    body = blob[6:]
    nonce, ct, tag = body[:16], body[16:-32], body[-32:]
    expected_tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("backup ciphertext tag mismatch — possibly tampered")
    ks = _keystream(key, nonce, len(ct))
    return bytes(c ^ k for c, k in zip(ct, ks))
