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
from .associative import router as assoc_router  # noqa: E402  v4 关联层
from .ui import router as ui_router  # noqa: E402  v4 小白 UI (无鉴权页面)

app.include_router(service_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(sr_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(backup_router, prefix="/memory/backup", dependencies=[Depends(auth)])
app.include_router(assoc_router, prefix="/memory", dependencies=[Depends(auth)])
app.include_router(ui_router)  # /ui 与 / (页面不鉴权, JS 调 API 时带 token)

if _log_router_ok:
    app.include_router(log_router, dependencies=[Depends(auth)])