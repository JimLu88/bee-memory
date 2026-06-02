"""bee-memory - 心脏
三层记忆 + 知识图谱 + 宪法 + 遗忘曲线
端口: 8004

H-SEMAS 蜂群微服务架构(七剑客之一)。
对外: REST + Bearer Token + plugin-manifest.json
"""
from __future__ import annotations

import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


SERVICE_NAME = "bee-memory"
SERVICE_PORT = 8004
BEARER_TOKEN = os.environ.get("BEE_BEARER_TOKEN", "dev-token-change-me")

app = FastAPI(title=SERVICE_NAME, version="0.1.0")
bearer = HTTPBearer(auto_error=False)

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


def auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    if credentials is None or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
    return credentials.credentials


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME, "port": str(SERVICE_PORT)}


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

app.include_router(service_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(sr_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(backup_router, prefix="/memory/backup", dependencies=[Depends(auth)])

if _log_router_ok:
    app.include_router(log_router, dependencies=[Depends(auth)])