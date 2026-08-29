"""SessionStart 钩子 (v4 记忆大脑 P2) — 会话开场自动注入记忆概览.

快而失败开放 (fail-open): 服务在跑就取紧凑 digest 打到 stdout (被 Claude 当上下文注入);
服务没起就 detached 拉起 (不等待) 并打一句提示, 绝不拖慢会话启动.

settings.json:
  "hooks": {"SessionStart": [{"hooks": [{"type":"command",
    "command":"py -3.11 \"D:/AI/AI 记忆中心/mcp/session_digest.py\""}]}]}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from memory_client_config import BASE, TOKEN

BACKEND_DIR = Path(os.environ.get("BEE_BACKEND_DIR", r"D:/AI/AI 记忆中心/backend"))


def _get(path: str, timeout: float) -> dict | None:
    try:
        req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _spawn_service() -> None:
    try:
        creation = 0x00000008 | 0x08000000 if os.name == "nt" else 0
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", "8004", "--log-level", "warning"],
            cwd=str(BACKEND_DIR), creationflags=creation,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _spawn_sync() -> None:
    """后台触发文件记忆同步 (fire-and-forget, 不阻塞会话启动). 让新写的记忆文件自动进大脑."""
    try:
        creation = 0x00000008 | 0x08000000 if os.name == "nt" else 0
        code = ("import urllib.request as u;"
                f"u.urlopen(u.Request('{BASE}/memory/sync-file-memories',"
                f"headers={{'Authorization':'Bearer {TOKEN}'}},method='POST'),timeout=120)")
        subprocess.Popen([sys.executable, "-c", code], creationflags=creation,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main() -> None:
    project = Path(os.getcwd()).name  # 钩子在项目 cwd 运行, 目录名当项目名
    if _get("/healthz", 1.5) is None:
        _spawn_service()  # 不等待, 下次工具调用时就绪
        print("🧠 记忆大脑 (claude-brain) 正在后台启动，可用 brain_recall / brain_digest 调用。")
        return
    _spawn_sync()  # 服务在线: 后台同步文件记忆 (自动, 不阻塞)
    dg = _get("/memory/digest?project=" + urllib.parse.quote(project) + "&limit=6", 3.5)
    if not dg:
        print("🧠 记忆大脑在线。可用 brain_recall(查询) / brain_connect(A,B) / brain_store(内容)。")
        return
    c = dg.get("counts", {})
    lines = [f"🧠 记忆大脑在线: {c.get('memories', 0)} 记忆 / {c.get('edges', 0)} 关联边 / "
             f"{c.get('due_review', 0)} 待复习。新任务前可 brain_recall(查询) 调历史；"
             f"拍板/踩坑用 brain_store 记住。"]
    for it in (dg.get("top") or [])[:5]:
        t = (it.get("title") or "")[:36]
        if t:
            lines.append(f"  · {t}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
