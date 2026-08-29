"""Identity-only endpoint; never imports the memory database or MCP auto-starter."""
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

BACKEND_ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Bee Memory Tachikoma Connection", docs_url=None, redoc_url=None)


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.method == "GET" and request.url.path in {"/healthz", "/tachikoma/v1/contract"}:
        return await call_next(request)
    return JSONResponse(status_code=403, content={"code": "PRODUCTION_EXECUTION_DISABLED", "service": "bee-memory", "mode": "connection_only"})


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok", "service": "bee-memory", "software_id": "ai-memory",
        "version": "0.2.0", "contract_version": "v1", "ready": True,
        "dependencies_ready": False, "mode": "connection_only",
        "production_execution_enabled": False,
        "business_entrypoint_present": (BACKEND_ROOT / "app" / "main.py").is_file(),
        "memory_runtime_started": False,
        "dependency_reason": "memory database, graph caches and MCP runtime are intentionally not opened",
    }


@app.get("/tachikoma/v1/contract")
def contract() -> dict:
    return {"schema_version": "tachikoma-connection/v1", "software_id": "ai-memory", "service": "bee-memory", "production_execution_enabled": False, "upgrade_execution_enabled": False, "business_actions": []}
