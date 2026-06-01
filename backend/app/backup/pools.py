"""v3-E 5 池适配器 — GitHub Gist / 坚果云 WebDAV / Notion / Gitee 码云 / GitLab.

每池支持多账号轮换 (env 列表配置)。未配 API key 时降级本地缓存 + 标 pending,
后台 retry 队列稍后上传 — 用户体验:写入立即返回,云端慢慢同步.

v6-O: pool_config.json (前端 BackupConfigPanel 写) 优先于 env, 实现"前端配 Key"闭环.
v8-CN: 去掉 Cloudflare R2 (国内连不上) 和假的阿里云盘池 (从没真上传),
       换成国内直连的 坚果云 WebDAV; Google Drive 支持服务账号 JSON.
v9-CN: 七牛云 Kodo 只有 30 天免费 → 换成 Gitee 码云私有仓库 (长期免费、国内直连、无期限).
"""
from __future__ import annotations

import os, json, base64, hashlib, hmac, time
from pathlib import Path
from typing import Protocol
from urllib import request as urlreq, error as urlerr

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "backup_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

POOL_CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "pool_config.json"


def _cfg(key: str, default: str = "") -> str:
    """优先读 pool_config.json (前端写的); 没有则回退 env."""
    try:
        if POOL_CONFIG_PATH.is_file():
            data = json.loads(POOL_CONFIG_PATH.read_text(encoding="utf-8"))
            v = data.get(key)
            if v:
                return str(v)
    except Exception:
        pass
    return os.environ.get(key, default)


class PoolAdapter(Protocol):
    name: str
    def put(self, shard_id: str, blob: bytes) -> dict: ...
    def get(self, remote_ref: str) -> bytes | None: ...
    def quota(self) -> dict: ...


def _local_put(pool: str, shard_id: str, blob: bytes) -> dict:
    d = CACHE_DIR / pool
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{shard_id}.bin"
    p.write_bytes(blob)
    return {"remote_ref": str(p), "account_id": "local", "pending_upload": True}


def _local_get(remote_ref: str) -> bytes | None:
    p = Path(remote_ref)
    return p.read_bytes() if p.exists() else None


class GistPool:
    name = "gist"

    def __init__(self) -> None:
        self.tokens = [t.strip() for t in _cfg("GITHUB_GIST_TOKENS", "").split(",") if t.strip()]
        self._idx = 0

    def _next_token(self) -> str | None:
        if not self.tokens:
            return None
        t = self.tokens[self._idx % len(self.tokens)]
        self._idx += 1
        return t

    def put(self, shard_id: str, blob: bytes) -> dict:
        token = self._next_token()
        if not token:
            return _local_put(self.name, shard_id, blob)
        payload = json.dumps({
            "description": f"bee-memory shard {shard_id}",
            "public": False,
            "files": {f"{shard_id}.b64": {"content": base64.b64encode(blob).decode("ascii")}},
        }).encode()
        req = urlreq.Request("https://api.github.com/gists", data=payload,
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}, method="POST")
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
                return {"remote_ref": data["id"],
                        "account_id": hashlib.sha1(token.encode()).hexdigest()[:8],
                        "pending_upload": False}
        except (urlerr.URLError, KeyError):
            return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        if "\\" in remote_ref or "/" in remote_ref:
            return _local_get(remote_ref)
        token = self._next_token() or ""
        req = urlreq.Request(f"https://api.github.com/gists/{remote_ref}",
            headers={"Authorization": f"token {token}"} if token else {})
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
                first = next(iter(data["files"].values()))
                return base64.b64decode(first["content"])
        except (urlerr.URLError, KeyError, StopIteration):
            return None

    def quota(self) -> dict:
        return {"accounts": len(self.tokens), "configured": bool(self.tokens)}


class WebDAVPool:
    """坚果云 (或任意 WebDAV: Nextcloud/TeraCLOUD 等). 国内直连, 配置最简单.

    WEBDAV_URL 形如 https://dav.jianguoyun.com/dav/bee-memory/ (必须以 / 结尾,
    指向一个已存在或可自动创建的目录). WEBDAV_USER=账号邮箱, WEBDAV_PASS=应用密码
    (坚果云: 网页端 → 账户信息 → 安全选项 → 添加应用密码).
    """
    name = "webdav"

    def __init__(self) -> None:
        url = _cfg("WEBDAV_URL", "").strip()
        if url and not url.endswith("/"):
            url += "/"
        self.url = url
        self.user = _cfg("WEBDAV_USER", "")
        self.password = _cfg("WEBDAV_PASS", "")
        self._mkcol_done = False

    def _auth_header(self) -> str:
        raw = f"{self.user}:{self.password}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _ensure_dir(self) -> None:
        # 幂等: 对目录发一次 MKCOL; 已存在返回 405, 忽略.
        if self._mkcol_done or not self.url:
            return
        self._mkcol_done = True
        try:
            req = urlreq.Request(self.url, headers={"Authorization": self._auth_header()}, method="MKCOL")
            urlreq.urlopen(req, timeout=15)
        except Exception:
            pass  # 已存在 / 不支持都无所谓, 后面 PUT 会暴露真问题

    def put(self, shard_id: str, blob: bytes) -> dict:
        if not (self.url and self.user and self.password):
            return _local_put(self.name, shard_id, blob)
        self._ensure_dir()
        key = f"{shard_id}.bin"
        req = urlreq.Request(
            self.url + key, data=blob,
            headers={"Authorization": self._auth_header(),
                     "Content-Type": "application/octet-stream"},
            method="PUT",
        )
        try:
            with urlreq.urlopen(req, timeout=30) as r:
                if r.status in (200, 201, 204):
                    return {"remote_ref": key,
                            "account_id": hashlib.sha1(self.user.encode()).hexdigest()[:8],
                            "pending_upload": False}
        except Exception:
            pass
        return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        if "\\" in remote_ref or remote_ref.count("/") > 1 or ":" in remote_ref:
            local = _local_get(remote_ref)
            if local is not None:
                return local
        if not (self.url and self.user and self.password):
            return None
        req = urlreq.Request(self.url + remote_ref,
                             headers={"Authorization": self._auth_header()})
        try:
            with urlreq.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            return None

    def quota(self) -> dict:
        return {"configured": bool(self.url and self.user and self.password)}


class NotionPool:
    name = "notion"

    def __init__(self) -> None:
        self.token = _cfg("NOTION_TOKEN", "")
        self.db = _cfg("NOTION_DATABASE_ID", "")

    def put(self, shard_id: str, blob: bytes) -> dict:
        if not (self.token and self.db):
            return _local_put(self.name, shard_id, blob)
        b64 = base64.b64encode(blob).decode("ascii")
        chunks = [b64[i:i + 1900] for i in range(0, len(b64), 1900)]
        blocks = [{"object": "block", "type": "code", "code": {
            "language": "plain text",
            "rich_text": [{"type": "text", "text": {"content": ch}}],
        }} for ch in chunks]
        payload = json.dumps({
            "parent": {"database_id": self.db},
            "properties": {"Name": {"title": [{"text": {"content": shard_id}}]}},
            "children": blocks,
        }).encode()
        req = urlreq.Request("https://api.notion.com/v1/pages", data=payload,
            headers={"Authorization": f"Bearer {self.token}", "Notion-Version": "2022-06-28",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                page_id = json.loads(r.read())["id"]
                return {"remote_ref": page_id, "account_id": self.token[-6:], "pending_upload": False}
        except (urlerr.URLError, KeyError):
            return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        if "/" in remote_ref or "\\" in remote_ref:
            return _local_get(remote_ref)
        if not self.token:
            return None
        req = urlreq.Request(f"https://api.notion.com/v1/blocks/{remote_ref}/children?page_size=100",
            headers={"Authorization": f"Bearer {self.token}", "Notion-Version": "2022-06-28"})
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
                b64 = "".join(rt.get("text", {}).get("content", "")
                              for block in data.get("results", [])
                              for rt in block.get("code", {}).get("rich_text", []))
                return base64.b64decode(b64) if b64 else None
        except (urlerr.URLError, KeyError):
            return None

    def quota(self) -> dict:
        return {"configured": bool(self.token and self.db)}


def _qiniu_b64(raw: bytes) -> str:
    """七牛 URL-safe base64 (保留 = 填充)."""
    return base64.urlsafe_b64encode(raw).decode("ascii")


class GiteePool:
    """Gitee 码云 私有仓库. 国内直连、长期免费、无 30 天限制 (替代七牛).

    原理: 和 GitHub Gist 池同套路, 但用码云私有仓库的"文件内容 API"存加密分片.
    每个分片存成仓库里一个文件 shards/<shard_id>.b64 (内容 = base64(分片密文)).

    配置 (前端 BackupConfigPanel 填, 或 env):
      GITEE_TOKEN  : 私人令牌 (码云 → 设置 → 私人令牌, 勾 projects 权限)
      GITEE_OWNER  : 你的码云用户名 (个人空间地址里的那个, 形如 zhangsan)
      GITEE_REPO   : 一个【私有】仓库名 (先在码云手动建一个空私有仓库, 如 bee-backup)
      GITEE_BRANCH : 分支, 默认 master
    """
    name = "gitee"
    _API = "https://gitee.com/api/v5"

    def __init__(self) -> None:
        self.token = _cfg("GITEE_TOKEN", "").strip()
        self.owner = _cfg("GITEE_OWNER", "").strip()
        self.repo = _cfg("GITEE_REPO", "").strip()
        self.branch = (_cfg("GITEE_BRANCH", "master") or "master").strip()

    def _ok(self) -> bool:
        return bool(self.token and self.owner and self.repo)

    def _path(self, shard_id: str) -> str:
        return f"shards/{shard_id}.b64"

    def _contents_url(self, path: str) -> str:
        return f"{self._API}/repos/{self.owner}/{self.repo}/contents/{path}"

    def _get_sha(self, path: str) -> str | None:
        """文件已存在则返回其 sha (更新需要), 不存在返回 None."""
        url = f"{self._contents_url(path)}?access_token={self.token}&ref={self.branch}"
        try:
            with urlreq.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
                return data.get("sha") if isinstance(data, dict) else None
        except Exception:
            return None

    def put(self, shard_id: str, blob: bytes) -> dict:
        if not self._ok():
            return _local_put(self.name, shard_id, blob)
        path = self._path(shard_id)
        content_b64 = base64.b64encode(blob).decode("ascii")
        sha = self._get_sha(path)  # 有则走更新 (PUT), 无则新建 (POST)
        payload: dict = {
            "access_token": self.token,
            "content": content_b64,
            "message": f"bee-memory shard {shard_id}",
            "branch": self.branch,
        }
        method = "POST"
        if sha:
            payload["sha"] = sha
            method = "PUT"
        req = urlreq.Request(
            self._contents_url(path),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json;charset=UTF-8"},
            method=method,
        )
        try:
            with urlreq.urlopen(req, timeout=30) as r:
                if r.status in (200, 201):
                    return {"remote_ref": path,
                            "account_id": hashlib.sha1(self.token.encode()).hexdigest()[:8],
                            "pending_upload": False}
        except Exception:
            pass
        return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        # 本地缓存 ref: Windows 绝对路径 (含 "\\" 或盘符 ":")
        if "\\" in remote_ref or ":" in remote_ref:
            local = _local_get(remote_ref)
            if local is not None:
                return local
        if not self._ok():
            return None
        url = f"{self._contents_url(remote_ref)}?access_token={self.token}&ref={self.branch}"
        try:
            with urlreq.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
                inner = data.get("content") if isinstance(data, dict) else None
                if not inner:
                    return None
                # 码云 PUT 时 content=base64(分片), 码云解码后把"分片"存为文件;
                # GET 返回 content=base64(文件)=base64(分片) → 解一次即得原始分片.
                return base64.b64decode(inner)
        except Exception:
            return None

    def quota(self) -> dict:
        return {"configured": self._ok(),
                "note": "" if self._ok() else "需配 GITEE_TOKEN/OWNER/REPO (私有仓库)"}


class GitLabPool:
    """GitLab 私有仓库 (替代 Google Drive — 个人 Google 账号 + 服务账号传不了).

    和码云/Gist 同套路: 用 GitLab Repository Files API 把加密分片存成仓库文件
    shards/<id>.b64. gitlab.com 免费无限私有库; 也支持自建 GitLab (改 GITLAB_HOST).

    配置 (前端 BackupConfigPanel 填, 或 env):
      GITLAB_TOKEN   : Personal Access Token
                       (gitlab.com → 右上头像 → Preferences → Access Tokens,
                        勾 api 或 write_repository 权限)
      GITLAB_PROJECT : 项目 ID(数字, 仓库主页 Settings→General 顶部可见) 或 "用户名/仓库名"
      GITLAB_BRANCH  : 分支, 默认 main
      GITLAB_HOST    : 默认 https://gitlab.com (自建实例才改)
    """
    name = "gitlab"

    def __init__(self) -> None:
        self.token = _cfg("GITLAB_TOKEN", "").strip()
        self.project = _cfg("GITLAB_PROJECT", "").strip()
        self.branch = (_cfg("GITLAB_BRANCH", "main") or "main").strip()
        host = _cfg("GITLAB_HOST", "https://gitlab.com").strip().rstrip("/")
        self.host = host or "https://gitlab.com"

    def _ok(self) -> bool:
        return bool(self.token and self.project)

    def _file_url(self, path: str, *, raw: bool = False) -> str:
        from urllib.parse import quote
        proj = quote(self.project, safe="")
        fp = quote(path, safe="")  # 含 / 一起编码成 %2F
        tail = "/raw" if raw else ""
        return f"{self.host}/api/v4/projects/{proj}/repository/files/{fp}{tail}"

    def put(self, shard_id: str, blob: bytes) -> dict:
        if not self._ok():
            return _local_put(self.name, shard_id, blob)
        path = f"shards/{shard_id}.b64"
        payload = json.dumps({
            "branch": self.branch,
            "content": base64.b64encode(blob).decode("ascii"),
            "encoding": "base64",
            "commit_message": f"bee-memory shard {shard_id}",
        }).encode()
        hdr = {"PRIVATE-TOKEN": self.token, "Content-Type": "application/json"}
        # 先试创建(POST); 文件已存在(400)则改更新(PUT)
        for method in ("POST", "PUT"):
            req = urlreq.Request(self._file_url(path), data=payload, headers=hdr, method=method)
            try:
                with urlreq.urlopen(req, timeout=30) as r:
                    if r.status in (200, 201):
                        return {"remote_ref": path,
                                "account_id": hashlib.sha1(self.token.encode()).hexdigest()[:8],
                                "pending_upload": False}
            except urlerr.HTTPError as e:
                if e.code == 400 and method == "POST":
                    continue  # 已存在 → 走 PUT 更新
                break
            except Exception:
                break
        return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        # 本地缓存 ref: Windows 绝对路径 (含 "\\" 或盘符 ":")
        if "\\" in remote_ref or ":" in remote_ref:
            local = _local_get(remote_ref)
            if local is not None:
                return local
        if not self._ok():
            return None
        url = self._file_url(remote_ref, raw=True) + f"?ref={self.branch}"
        req = urlreq.Request(url, headers={"PRIVATE-TOKEN": self.token})
        try:
            with urlreq.urlopen(req, timeout=30) as r:
                return r.read()  # /raw 直接返回原始分片字节
        except Exception:
            return None

    def quota(self) -> dict:
        return {"configured": self._ok(),
                "note": "" if self._ok() else "需配 GITLAB_TOKEN + GITLAB_PROJECT(ID 或 user/repo)"}


ALL_POOLS: list[PoolAdapter] = [GistPool(), WebDAVPool(), NotionPool(), GiteePool(), GitLabPool()]


def by_name(name: str) -> PoolAdapter:
    for p in ALL_POOLS:
        if p.name == name:
            return p
    raise KeyError(f"unknown pool: {name}")
