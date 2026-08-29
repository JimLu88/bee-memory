"""bee-memory - 心脏
三层记忆 + 知识图谱 + 宪法 + 遗忘曲线
端口: 8004

H-SEMAS 蜂群微服务架构(七剑客之一)。
对外: REST + Bearer Token + plugin-manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


SERVICE_NAME = "bee-memory"
SERVICE_PORT = 8004
BEARER_TOKEN = os.environ.get("BEE_BEARER_TOKEN", "dev-token-change-me")

app = FastAPI(title=SERVICE_NAME, version="0.2.0")
bearer = HTTPBearer(auto_error=False)

# v4 安全: 默认 token 大声告警 (本地默认可用, 上公网前务必改 BEE_BEARER_TOKEN)
if BEARER_TOKEN == "dev-token-change-me":
    import logging as _lg
    _lg.getLogger("bee.security").warning(
        "[bee-memory] 正在使用默认 Bearer token; 仅限本地回环. 对外暴露前请设 BEE_BEARER_TOKEN.")

# v4 安全(可选): 严格 Host 白名单, 防 DNS-rebinding. 默认关 (不影响 LAN/NAS 访问).
if os.environ.get("BEE_STRICT_HOST") == "1":
    _ALLOWED = {"127.0.0.1", "localhost", "::1"} | set(
        h.strip() for h in os.environ.get("BEE_ALLOWED_HOSTS", "").split(",") if h.strip())

    @app.middleware("http")
    async def _host_guard(request, call_next):
        from starlette.responses import JSONResponse
        host = (request.headers.get("host") or "").split(":")[0]
        if host and host not in _ALLOWED:
            return JSONResponse({"detail": "host not allowed"}, status_code=421)
        return await call_next(request)

# v3-I OpenTelemetry + /metrics (silent fallback if deps missing)
import sys as _sys  # noqa: E402
_sys.path.insert(0, "D:/AI/observability")
try:
    from bee_otel import init_otel  # type: ignore
    init_otel(SERVICE_NAME, app)
except Exception:
    pass

_log_router_ok = False
try:
    from bee_logs import setup_service_logging, log_router  # type: ignore
    from pathlib import Path as _BeePath
    setup_service_logging(SERVICE_NAME,
                          _BeePath(__file__).parent.parent / "data" / "logs")
    _log_router_ok = True
except Exception:
    pass


def _principal_for_token(token: str, agent_id: str = "") -> dict[str, Any]:
    """Resolve identity only from server-owned token/agent mappings."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    token_key = f"sha256:{digest}"
    principal: dict[str, Any] = {
        "principal_type": os.environ.get("BEE_DEFAULT_PRINCIPAL_TYPE", "user").strip() or "user",
        "principal_id": os.environ.get("BEE_DEFAULT_PRINCIPAL_ID", "jim").strip() or "jim",
        "roles": [value.strip() for value in os.environ.get("BEE_DEFAULT_PRINCIPAL_ROLES", "").split(",") if value.strip()],
        "identity_source": "default_token",
        "token_hash": token_key,
    }
    try:
        configured = json.loads(os.environ.get("BEE_TOKEN_PRINCIPAL_MAP_JSON", "{}") or "{}")
    except Exception:
        configured = {}
    candidate = configured.get(token_key) if isinstance(configured, dict) else None
    if isinstance(candidate, dict):
        principal.update({
            "principal_type": str(candidate.get("principal_type") or candidate.get("type") or "user").strip()[:40],
            "principal_id": str(candidate.get("principal_id") or candidate.get("id") or "").strip()[:160],
            "roles": [str(value).strip()[:160] for value in candidate.get("roles", []) if str(value).strip()][:20],
            "identity_source": "token_map",
        })
    if not principal["principal_id"]:
        raise HTTPException(status_code=500, detail="server principal mapping is incomplete")

    requested_agent = str(agent_id or "").strip()[:160]
    if requested_agent:
        try:
            agents = json.loads(os.environ.get("BEE_AGENT_PRINCIPAL_MAP_JSON", "{}") or "{}")
        except Exception:
            agents = {}
        agent = agents.get(requested_agent) if isinstance(agents, dict) else None
        allowed_hashes = set(str(value) for value in (agent or {}).get("token_hashes", [])) if isinstance(agent, dict) else set()
        if not isinstance(agent, dict) or token_key not in allowed_hashes:
            raise HTTPException(status_code=403, detail="agent identity is not mapped for this token")
        principal.update({
            "principal_type": "agent",
            "principal_id": str(agent.get("principal_id") or requested_agent).strip()[:160],
            "roles": [str(value).strip()[:160] for value in agent.get("roles", []) if str(value).strip()][:20],
            "identity_source": "agent_map",
        })
    return principal


def auth(request: Request,
         credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any]:
    if credentials is None or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
    principal = _principal_for_token(
        credentials.credentials,
        request.headers.get("x-bee-agent-id", ""),
    )
    request.state.bee_principal = principal
    return principal


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME, "port": str(SERVICE_PORT)}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    """Expose cold-start state so callers can warm or degrade deliberately."""
    from . import associative, memory, ppr, semantic

    lexical_ready = bool(getattr(associative, "_HOT_FTS_READY", False))
    vector_ready = semantic._CACHE.get("mat") is not None
    graph_ready = ppr._CACHE.get("P") is not None
    warming = bool(
        getattr(associative, "_HOT_FTS_WARMING", False)
        or getattr(semantic, "_WARMING", False)
        or getattr(ppr, "_WARMING", False)
    )
    database_ready = False
    database_state = "locked"
    try:
        with memory._conn() as conn:
            conn.execute("SELECT 1 FROM memories LIMIT 1").fetchone()
        database_ready = True
        database_state = "ready"
    except sqlite3.OperationalError as exc:
        database_state = "locked" if "locked" in str(exc).casefold() else "error"
    state = (
        "ready" if database_ready and lexical_ready and vector_ready and graph_ready
        else ("warming" if database_ready and warming else "degraded")
    )
    return {
        "status": state,
        "service": SERVICE_NAME,
        "components": {
            "lexical_cache": "ready" if lexical_ready else ("warming" if warming else "cold"),
            "vector_cache": "ready" if vector_ready else ("warming" if warming else "cold"),
            "graph_cache": "ready" if graph_ready else ("warming" if warming else "cold"),
            "database": database_state,
        },
    }


@app.get("/manifest")
def manifest() -> dict:
    """plugin-spec.md compliant manifest (v5-G)."""
    return {
        "name": SERVICE_NAME,
        "version": "0.1.0",
        "purpose": "三层记忆 + 知识图谱 + 宪法 + 遗忘曲线",
        "endpoints": [],  # filled per-service below in subroutes
        "auth": {"type": "bearer"},
    }


# ----- subroutes loaded from app. -----
from .memory import router as service_router  # noqa: E402
from .spaced_repetition import router as sr_router  # noqa: E402
from .backup import router as backup_router  # noqa: E402
from .associative import router as assoc_router  # noqa: E402  v4 关联层
from .cognitive import router as cognitive_router  # noqa: E402  v9 海马体/分层召回
from .ui import router as ui_router  # noqa: E402  v4 小白 UI (无鉴权页面)

app.include_router(service_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(sr_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(backup_router, prefix="/memory/backup", dependencies=[Depends(auth)])
app.include_router(assoc_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(cognitive_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(ui_router)  # /ui 与 / (页面不鉴权, JS 调 API 时带 token)


@app.on_event("startup")
def _warm_vector_cache_after_startup() -> None:
    """Keep first-user recall from paying NAS vector/graph load costs."""
    from . import associative, ppr, semantic

    # jieba loads its dictionary lazily (~0.9s on this NAS). Pay that cost
    # during service startup, not on the first person's Q1 recall.
    associative._fts_query("塔奇克马 memory warmup")
    associative.warm_hot_fts()
    semantic.warm_cache_async(delay_seconds=1.0)
    ppr.warm_cache_async(delay_seconds=2.0)

if _log_router_ok:
    app.include_router(log_router, dependencies=[Depends(auth)])
