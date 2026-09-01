"""夜间睡眠循环 runner (v4 记忆大脑 P3) — 由 Windows 计划任务每晚调用.

bee-memory 的权威实例在 NAS。本 runner 只等待 NAS 恢复并调用一次睡眠接口，绝不在
本机启动第二个记忆后端。每次运行都会写一份脱敏 JSON 回执，便于区分“没有运行”、
“NAS 暂时不可达”和“远端循环失败”。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from memory_client_config import BASE, TOKEN

DO_FORGET = os.environ.get("BEE_SLEEP_FORGET", "0")  # 1=真删低激活记忆; 默认只报告
RECEIPT_DIR = Path(os.environ.get(
    "BEE_SLEEP_RECEIPT_DIR", r"D:/AI/AI 记忆中心/logs/sleep-cycle"))
HEALTH_DELAYS = (0, 5, 15, 30)


def _health() -> bool:
    try:
        urllib.request.urlopen(
            urllib.request.Request(BASE + "/healthz", headers={"Authorization": f"Bearer {TOKEN}"}),
            timeout=2)
        return True
    except Exception:
        return False


def _wait_for_nas() -> tuple[bool, int]:
    """等待短暂网络抖动；只探测 NAS，不启动本机服务。"""
    for attempt, delay in enumerate(HEALTH_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        if _health():
            return True, attempt
    return False, len(HEALTH_DELAYS)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_endpoint() -> str:
    parsed = urlparse(BASE)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "invalid"


def _write_receipt(receipt: dict, receipt_dir: Path) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    path = receipt_dir / f"sleep-cycle-{stamp}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _finish(receipt: dict, receipt_dir: Path, code: int) -> int:
    receipt["finished_at"] = _now()
    path = _write_receipt(receipt, receipt_dir)
    stream = sys.stdout if code == 0 else sys.stderr
    stream.write(json.dumps({
        "status": receipt["status"],
        "receipt": str(path),
        "health_attempts": receipt.get("health_attempts", 0),
    }, ensure_ascii=False) + "\n")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只检查 NAS 健康，不运行睡眠循环")
    parser.add_argument("--receipt-dir", type=Path, default=RECEIPT_DIR)
    args = parser.parse_args(argv)

    receipt = {
        "schema_version": "bee-memory.sleep-cycle.runner-receipt.v1",
        "started_at": _now(),
        "mode": "health_check" if args.check else "sleep_cycle",
        "endpoint": _safe_endpoint(),
        "do_forget": False,
    }
    if DO_FORGET != "0":
        receipt.update(status="invalid_configuration", error_type="do_forget_not_authorized")
        return _finish(receipt, args.receipt_dir, 1)

    ready, attempts = _wait_for_nas()
    receipt["health_attempts"] = attempts
    if not ready:
        receipt.update(status="nas_unavailable", error_type="health_check_failed")
        return _finish(receipt, args.receipt_dir, 1)
    if args.check:
        receipt["status"] = "healthy"
        return _finish(receipt, args.receipt_dir, 0)

    req = urllib.request.Request(
        BASE + f"/memory/sleep-cycle?do_forget={DO_FORGET}&render_vault=1",
        headers={"Authorization": f"Bearer {TOKEN}"}, method="POST")
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        receipt["status"] = "ok" if r.get("status") in {"ok", "skipped_already_running"} else "remote_failed"
        receipt["remote_status"] = r.get("status", "missing")
        if "elapsed_s" in r:
            receipt["remote_elapsed_s"] = r["elapsed_s"]
        return _finish(receipt, args.receipt_dir, 0 if receipt["status"] == "ok" else 1)
    except Exception as e:
        receipt.update(
            status="request_failed",
            error_type=type(e).__name__,
            error_message=str(e)[:240],
        )
        return _finish(receipt, args.receipt_dir, 1)


if __name__ == "__main__":
    raise SystemExit(main())
