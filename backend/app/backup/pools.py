"""v3-E 5 池适配器 — GitHub Gist / 坚果云 WebDAV / Notion / 七牛云 Kodo / Google Drive.

每池支持多账号轮换 (env 列表配置)。未配 API key 时降级本地缓存 + 标 pending,
后台 retry 队列稍后上传 — 用户体验:写入立即返回,云端慢慢同步.

v6-O: pool_config.json (前端 BackupConfigPanel 写) 优先于 env, 实现"前端配 Key"闭环.
v8-CN: 去掉 Cloudflare R2 (国内连不上) 和假的阿里云盘池 (从没真上传),
       换成国内直连的 坚果云 WebDAV + 七牛云 Kodo; Google Drive 支持服务账号 JSON.
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


class QiniuPool:
    """七牛云 Kodo 对象存储. 国内直连, 标准存储 10GB 免费.

    需要: QINIU_AK / QINIU_SK (个人中心 → 密钥管理), QINIU_BUCKET (存储空间名),
    QINIU_DOMAIN (空间绑定的下载域名, 形如 http://xxx.bkt.clouddn.com 或自有域名 —
    私有空间下载必须靠它), QINIU_REGION (区域上传域名后缀, 默认 z0=华东).
    上传 token 用 HMAC-SHA1 签名 (stdlib, 无需第三方库).
    """
    name = "qiniu"

    _REGION_HOST = {
        "z0": "https://up-z0.qiniup.com", "z1": "https://up-z1.qiniup.com",
        "z2": "https://up-z2.qiniup.com", "na0": "https://up-na0.qiniup.com",
        "as0": "https://up-as0.qiniup.com",
    }

    def __init__(self) -> None:
        self.ak = _cfg("QINIU_AK", "")
        self.sk = _cfg("QINIU_SK", "")
        self.bucket = _cfg("QINIU_BUCKET", "")
        dom = _cfg("QINIU_DOMAIN", "").strip().rstrip("/")
        if dom and not dom.startswith("http"):
            dom = "http://" + dom
        self.domain = dom
        self.region = (_cfg("QINIU_REGION", "z0") or "z0").strip()

    def _ok(self) -> bool:
        return bool(self.ak and self.sk and self.bucket)

    def _upload_token(self, key: str) -> str:
        deadline = int(time.time()) + 3600
        policy = json.dumps({"scope": f"{self.bucket}:{key}", "deadline": deadline},
                            ensure_ascii=False).encode("utf-8")
        encoded = _qiniu_b64(policy)
        sign = hmac.new(self.sk.encode(), encoded.encode(), hashlib.sha1).digest()
        return f"{self.ak}:{_qiniu_b64(sign)}:{encoded}"

    def put(self, shard_id: str, blob: bytes) -> dict:
        if not self._ok():
            return _local_put(self.name, shard_id, blob)
        key = f"shards/{shard_id}.bin"
        token = self._upload_token(key)
        boundary = "----beeqiniu" + hashlib.md5(key.encode()).hexdigest()[:12]
        parts: list[bytes] = []
        for field, val in (("key", key), ("token", token)):
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{val}\r\n".encode())
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{shard_id}.bin\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n".encode()
            + blob + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        host = self._REGION_HOST.get(self.region, self._REGION_HOST["z0"])
        req = urlreq.Request(host, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        try:
            with urlreq.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
                if data.get("key"):
                    return {"remote_ref": data["key"], "account_id": self.ak[:8], "pending_upload": False}
        except Exception:
            pass
        return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        if "\\" in remote_ref or ":" in remote_ref or remote_ref.endswith(".bin") and "/" in remote_ref and not remote_ref.startswith("shards/"):
            local = _local_get(remote_ref)
            if local is not None:
                return local
        if not (self._ok() and self.domain):
            return None
        deadline = int(time.time()) + 3600
        base = f"{self.domain}/{remote_ref}?e={deadline}"
        sign = hmac.new(self.sk.encode(), base.encode(), hashlib.sha1).digest()
        url = f"{base}&token={self.ak}:{_qiniu_b64(sign)}"
        try:
            with urlreq.urlopen(url, timeout=30) as r:
                return r.read()
        except Exception:
            return None

    def quota(self) -> dict:
        return {"configured": self._ok(),
                "note": "" if self.domain else "未配 QINIU_DOMAIN: 能上传但恢复需绑定下载域名"}


class GDrivePool:
    """Google Drive. 国内需梯子. 支持两种凭证:
    1) GOOGLE_DRIVE_SA_JSON = 服务账号 JSON 整张 (推荐, 贴一次即可, 代码自动签 JWT 换 token).
       ⚠ 服务账号本身没有 My Drive 配额, 必须把 GOOGLE_DRIVE_FOLDER 指向一个
         "共享云端硬盘(Shared Drive)" 内的文件夹, 或一个已共享给该服务账号邮箱
         (client_email) 的文件夹, 否则上传会报 storageQuotaExceeded.
       需要 pip install cryptography (RS256 签名); 未装则该模式不可用.
    2) GOOGLE_DRIVE_TOKEN = 直接给一个短期 OAuth access token (1 小时过期, 不推荐).
    """
    name = "gdrive"

    def __init__(self) -> None:
        self.raw_token = _cfg("GOOGLE_DRIVE_TOKEN", "")
        self.folder = _cfg("GOOGLE_DRIVE_FOLDER", "")
        self.sa_json = _cfg("GOOGLE_DRIVE_SA_JSON", "")
        self._cached_token = ""
        self._token_exp = 0

    def _sa_access_token(self) -> str:
        """用服务账号私钥签 JWT, 换 1 小时 access token (带缓存)."""
        now = int(time.time())
        if self._cached_token and now < self._token_exp - 60:
            return self._cached_token
        try:
            sa = json.loads(self.sa_json)
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except Exception:
            return ""  # JSON 坏 或 没装 cryptography
        token_uri = sa.get("token_uri") or "https://oauth2.googleapis.com/token"
        header = _qiniu_b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip("=")
        claim = _qiniu_b64(json.dumps({
            "iss": sa.get("client_email", ""),
            "scope": "https://www.googleapis.com/auth/drive",
            "aud": token_uri, "iat": now, "exp": now + 3600,
        }).encode()).rstrip("=")
        signing_input = f"{header}.{claim}".encode()
        try:
            key = serialization.load_pem_private_key(sa["private_key"].encode(), password=None)
            sig = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        except Exception:
            return ""
        assertion = f"{header}.{claim}.{_qiniu_b64(sig).rstrip('=')}"
        body = ("grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer"
                f"&assertion={assertion}").encode()
        req = urlreq.Request(token_uri, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                tok = json.loads(r.read())
            self._cached_token = tok.get("access_token", "")
            self._token_exp = now + int(tok.get("expires_in", 3600))
            return self._cached_token
        except Exception:
            return ""

    def _token(self) -> str:
        if self.sa_json:
            return self._sa_access_token()
        return self.raw_token

    def put(self, shard_id: str, blob: bytes) -> dict:
        token = self._token()
        if not token:
            return _local_put(self.name, shard_id, blob)
        metadata = json.dumps({"name": f"{shard_id}.bin",
                               "parents": [self.folder] if self.folder else []}).encode()
        boundary = "bee_boundary"
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode()
            + metadata
            + f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
            + blob
            + f"\r\n--{boundary}--".encode()
        )
        req = urlreq.Request(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&supportsAllDrives=true",
            data=body,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": f"multipart/related; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                fid = json.loads(r.read())["id"]
                return {"remote_ref": fid, "account_id": "gdrive", "pending_upload": False}
        except (urlerr.URLError, KeyError):
            return _local_put(self.name, shard_id, blob)

    def get(self, remote_ref: str) -> bytes | None:
        if "/" in remote_ref or "\\" in remote_ref:
            return _local_get(remote_ref)
        token = self._token()
        if not token:
            return None
        req = urlreq.Request(
            f"https://www.googleapis.com/drive/v3/files/{remote_ref}?alt=media&supportsAllDrives=true",
            headers={"Authorization": f"Bearer {token}"})
        try:
            with urlreq.urlopen(req, timeout=15) as r:
                return r.read()
        except urlerr.URLError:
            return None

    def quota(self) -> dict:
        return {"configured": bool(self.sa_json or self.raw_token),
                "note": "服务账号需把目标文件夹共享给 client_email" if self.sa_json else ""}


ALL_POOLS: list[PoolAdapter] = [GistPool(), WebDAVPool(), NotionPool(), QiniuPool(), GDrivePool()]


def by_name(name: str) -> PoolAdapter:
    for p in ALL_POOLS:
        if p.name == name:
            return p
    raise KeyError(f"unknown pool: {name}")
