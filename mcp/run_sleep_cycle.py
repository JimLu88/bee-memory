"""夜间睡眠循环 runner (v4 记忆大脑 P3) — 由 Windows 计划任务每晚调用.

确保 bee-memory 在跑 (没起就拉起), 然后 POST /memory/sleep-cycle (走服务单进程, 避免并发写).
注册计划任务见 register_schedule.ps1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("BEE_MEMORY_URL", "http://127.0.0.1:8004")
TOKEN = os.environ.get("BEE_BEARER_TOKEN", "dev-token-change-me")
BACKEND_DIR = Path(os.environ.get("BEE_BACKEND_DIR", r"D:/AI/AI 记忆中心/backend"))
DO_FORGET = os.environ.get("BEE_SLEEP_FORGET", "0")  # 1=真删低激活记忆; 默认只报告


def _health() -> bool:
    try:
        urllib.request.urlopen(
            urllib.request.Request(BASE + "/healthz", headers={"Authorization": f"Bearer {TOKEN}"}),
            timeout=2)
        return True
    except Exception:
        return False


def _ensure() -> bool:
    if _health():
        return True
    creation = 0x00000008 | 0x08000000 if os.name == "nt" else 0
    try:
        subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                          "--port", "8004", "--log-level", "warning"],
                         cwd=str(BACKEND_DIR), creationflags=creation,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    for _ in range(30):
        time.sleep(1)
        if _health():
            return True
    return False


def main() -> int:
    if not _ensure():
        sys.stderr.write("[sleep-cycle] bee-memory 起不来, 跳过\n")
        return 1
    req = urllib.request.Request(
        BASE + f"/memory/sleep-cycle?do_forget={DO_FORGET}&render_vault=1",
        headers={"Authorization": f"Bearer {TOKEN}"}, method="POST")
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        sys.stdout.write(json.dumps(r, ensure_ascii=False)[:800] + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"[sleep-cycle] 失败: {e!r}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
