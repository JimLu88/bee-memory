"""小白友好 UI (v4 记忆大脑) — 零构建单页仪表盘, 由 FastAPI 直接服务.

GET /ui  → 一个自包含 HTML (内联 CSS/JS). 页面本身不鉴权 (方便浏览器打开),
API 调用由页面 JS 带 Bearer token (默认 dev-token-change-me, 可在 ⚙️ 里改).
打开: http://127.0.0.1:8004/ui
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()
_HTML = Path(__file__).parent / "brain_ui.html"


@router.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    try:
        return HTMLResponse(_HTML.read_text(encoding="utf-8"))
    except Exception as e:
        return HTMLResponse(f"<h3>UI 文件缺失: {e}</h3>", status_code=500)


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")
