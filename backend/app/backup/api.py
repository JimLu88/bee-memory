"""v3-E 备份 HTTP 端点 — POST /run/{memory_id}, GET /restore/{memory_id}, POST /retry, GET /stats, GET/POST /config."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from . import coordinator as coord
from .pools import POOL_CONFIG_PATH

router = APIRouter()


@router.post("/run/{memory_id}")
def run(memory_id: str) -> dict:
    try:
        return coord.backup_memory(memory_id)
    except KeyError:
        raise HTTPException(404, f"memory {memory_id} 不存在")


@router.get("/restore/{memory_id}")
def restore(memory_id: str) -> dict:
    result = coord.restore_memory(memory_id)
    if result is None:
        raise HTTPException(404, "no shards on record for this memory")
    return result


@router.post("/retry")
def retry(limit: int = 50) -> dict:
    return coord.retry_pending(limit)


@router.get("/stats")
def stats() -> dict:
    return coord.stats()


# v6-O 前端写 5 池 Key 入口
class PoolConfigBody(BaseModel):
    GITHUB_GIST_TOKENS: str = ""
    # 坚果云 WebDAV
    WEBDAV_URL: str = ""
    WEBDAV_USER: str = ""
    WEBDAV_PASS: str = ""
    # Notion
    NOTION_TOKEN: str = ""
    NOTION_DATABASE_ID: str = ""
    # Gitee 码云私有仓库 (替代七牛, 长期免费无 30 天限制)
    GITEE_TOKEN: str = ""
    GITEE_OWNER: str = ""
    GITEE_REPO: str = ""
    GITEE_BRANCH: str = ""
    # Google Drive (服务账号 JSON 整张 / 或短期 OAuth token)
    GOOGLE_DRIVE_SA_JSON: str = ""
    GOOGLE_DRIVE_TOKEN: str = ""
    GOOGLE_DRIVE_FOLDER: str = ""


@router.get("/config")
def get_config() -> dict[str, Any]:
    if not POOL_CONFIG_PATH.is_file():
        return {"configured": False, "fields": {}}
    try:
        data = json.loads(POOL_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"configured": False, "fields": {}}
    return {
        "configured": True,
        "fields": {k: ("***" + v[-4:] if isinstance(v, str) and len(v) >= 4 else "")
                   for k, v in data.items()},
        "path": str(POOL_CONFIG_PATH),
    }


@router.post("/config")
def set_config(body: PoolConfigBody) -> dict[str, Any]:
    """前端 BackupConfigPanel 写 5 池 Key. 留空 = 该字段不改 (保留旧值)."""
    existing: dict[str, str] = {}
    if POOL_CONFIG_PATH.is_file():
        try:
            existing = json.loads(POOL_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    submitted = body.model_dump()
    for k, v in submitted.items():
        if v:
            existing[k] = v
    POOL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_CONFIG_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "saved": True, "fields_updated": [k for k, v in submitted.items() if v],
        "note": "已保存. 需重启 bee-memory 让 pool adapters 重读 (托盘 → bee-memory → 重启).",
        "path": str(POOL_CONFIG_PATH),
    }
