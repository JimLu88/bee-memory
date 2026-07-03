"""本地 LLM 助手 (v5 记忆大脑 Phase 2) — 睡眠循环里的蒸馏/建边用.

走本地 Ollama /api/chat, 默认 qwen2.5:7b-instruct (非思考型, JSON 稳, D 盘, 免费).
- 只在夜间睡眠循环调用 (BEE_SLEEP_LLM=1 时), 不进在线检索热路径.
- 严格 JSON 模式 + 容错解析; Ollama 挂了返回 None, 调用方全部降级 (不阻断循环).
- 绝不用思考型模型默认档 (qwen3.5/qwen3-vl 会思考停不下来致 content 空).
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("BEE_SLEEP_MODEL", "qwen2.5:7b-instruct")
LLM_TIMEOUT = float(os.environ.get("BEE_LLM_TIMEOUT", "120"))
SLEEP_LLM_ENABLED = os.environ.get("BEE_SLEEP_LLM", "1") == "1"


def available() -> bool:
    """LLM 是否可用 (开关开 且 Ollama 在)."""
    if not SLEEP_LLM_ENABLED:
        return False
    try:
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/tags"), timeout=3)
        return True
    except Exception:
        return False


def _parse_json_loose(text: str) -> Any | None:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        for lo, hi in (("{", "}"), ("[", "]")):
            s, e = t.find(lo), t.rfind(hi)
            if s >= 0 and e > s:
                try:
                    return json.loads(t[s:e + 1])
                except Exception:
                    pass
    return None


def chat_json(prompt: str, system: str = "You output ONLY valid JSON, no prose.") -> Any | None:
    """一次 LLM 调用, 强制 JSON. 失败返回 None (调用方降级)."""
    if not SLEEP_LLM_ENABLED:
        return None
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "think": False,           # 防思考型模型空 content
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            resp = json.loads(r.read())
        content = (resp.get("message") or {}).get("content") or ""
        return _parse_json_loose(content)
    except Exception:
        return None
