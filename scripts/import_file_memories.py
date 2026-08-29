"""一次性: 把用户的文件记忆 (~/.claude/.../memory/*.md) 导入记忆大脑.

只读 C 盘那些 .md, 写入走 :8004 /memory/store (库在 D 盘). type→kind:
feedback→procedural, project→episodic, user/reference→semantic. 首次导入, 别重复跑.
"""
from __future__ import annotations
import json, os, re, sys, urllib.request
from pathlib import Path

MEM_DIR = Path(os.environ.get("BEE_FILEMEM_DIR",
              r"C:\Users\lzdwy\.claude\projects\C--Users-lzdwy\memory"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))
from memory_client_config import BASE, TOKEN as TOK
H = {"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"}

KIND_MAP = {"feedback": "procedural", "project": "episodic",
            "user": "semantic", "reference": "semantic"}


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers=H, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _parse(md: str):
    name = desc = mtype = ""
    body = md
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", md, re.S)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                desc = line.split(":", 1)[1].strip().strip('"')
            elif re.match(r"\s*type:", line):
                mtype = line.split(":", 1)[1].strip()
    return name, desc, mtype, body.strip()


def main():
    files = sorted(p for p in MEM_DIR.glob("*.md") if p.name != "MEMORY.md")
    done = 0
    for p in files:
        try:
            name, desc, mtype, body = _parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        slug = name or p.stem
        kind = KIND_MAP.get(mtype, "semantic")
        content = (f"{desc}\n\n{body}" if desc else body)[:6000]
        meta = {"source": "file_memory", "name": slug, "type": mtype, "title": slug}
        try:
            _post("/memory/store", {"kind": kind, "content": content,
                                    "importance": 5, "mode_id": slug, "meta": meta})
            done += 1
        except Exception as e:
            print("  fail", p.name, repr(e)[:120])
    print(f"imported {done}/{len(files)} file memories")
    try:
        idx = _post("/memory/reindex-concepts", {"rebuild_edges": True})
        print("reindex:", {k: idx.get(k) for k in ("scanned", "concepts", "cooccur_edges")})
    except Exception as e:
        print("reindex fail", repr(e)[:120])


if __name__ == "__main__":
    main()
