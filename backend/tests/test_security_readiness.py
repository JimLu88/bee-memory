from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import associative, main, memory, ppr, semantic  # noqa: E402


def test_server_token_and_agent_mapping_cannot_be_self_reported(monkeypatch):
    token = "test-token-not-a-secret"
    token_hash = "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    monkeypatch.setenv("BEE_TOKEN_PRINCIPAL_MAP_JSON", json.dumps({
        token_hash: {"type": "user", "id": "jim", "roles": ["owner"]},
    }))
    monkeypatch.setenv("BEE_AGENT_PRINCIPAL_MAP_JSON", json.dumps({
        "tachikoma-squad": {
            "principal_id": "tachikoma-squad",
            "roles": ["memory-reader"],
            "token_hashes": [token_hash],
        },
    }))

    user = main._principal_for_token(token)
    agent = main._principal_for_token(token, "tachikoma-squad")
    assert (user["principal_type"], user["principal_id"]) == ("user", "jim")
    assert (agent["principal_type"], agent["principal_id"]) == ("agent", "tachikoma-squad")
    with pytest.raises(HTTPException) as error:
        main._principal_for_token(token, "request-body-invented-agent")
    assert error.value.status_code == 403


def test_readyz_exposes_cold_warming_and_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "readiness.sqlite")
    monkeypatch.setitem(semantic._CACHE, "mat", None)
    monkeypatch.setitem(ppr._CACHE, "P", None)
    monkeypatch.setattr(associative, "_HOT_FTS_READY", False)
    monkeypatch.setattr(associative, "_HOT_FTS_WARMING", False)
    monkeypatch.setattr(semantic, "_WARMING", False)
    monkeypatch.setattr(ppr, "_WARMING", False)
    assert main.readyz()["status"] == "degraded"

    monkeypatch.setattr(semantic, "_WARMING", True)
    assert main.readyz()["status"] == "warming"

    monkeypatch.setitem(semantic._CACHE, "mat", object())
    monkeypatch.setitem(ppr._CACHE, "P", object())
    monkeypatch.setattr(associative, "_HOT_FTS_READY", True)
    monkeypatch.setattr(semantic, "_WARMING", False)
    ready = main.readyz()
    assert ready["status"] == "ready"
    assert ready["components"]["lexical_cache"] == "ready"
