"""Shared bee-memory endpoint and protected credential lookup."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_BASE = "http://192.168.31.21:8004"


def _stored_token() -> str:
    root = Path(os.environ.get("TACHIKOMA_ROOT", r"D:\AI\tachikoma"))
    if not root.is_dir():
        return ""
    inserted = False
    try:
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
        from base.secrets import get_secret

        found, value = get_secret("BEE_BEARER_TOKEN")
        return value if found else ""
    except Exception:
        return ""
    finally:
        if inserted:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass


BASE = os.environ.get("BEE_MEMORY_URL", DEFAULT_BASE).rstrip("/")
TOKEN = os.environ.get("BEE_BEARER_TOKEN", "").strip() or _stored_token() or "dev-token-change-me"


def is_loopback() -> bool:
    return (urlparse(BASE).hostname or "").casefold() in {"127.0.0.1", "localhost", "::1"}
