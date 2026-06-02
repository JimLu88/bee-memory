# bee-memory (心脏) · 集成说明

> 三层记忆 (工作/长期/语义) + 6 因子激活检索 + SM-2 Anki 复习 + 5 池备份。
> 服务端口 **8004** · Bearer 鉴权 · `/docs` OpenAPI

---

## 一、启动

```powershell
cd "D:\AI\AI 记忆中心\backend"
py -3.11 -m pip install -r requirements.txt
py -3.11 -m uvicorn app.main:app --host 127.0.0.1 --port 8004
```

健康检查：`(Invoke-WebRequest http://127.0.0.1:8004/healthz).Content`

---

## 二、端点 (核心)

| 端点 | 方法 | 说明 |
|---|---|---|
| `/memory/store` | POST | `{persona_id,kind,content,meta?,importance?,novelty?,predictive_value?}` |
| `/memory/recall` | POST | `{persona_id,query,k?,strategy?}` 6 因子激活检索 |
| `/memory/review/due` | GET | SM-2 到期复习卡 |
| `/memory/review/grade` | POST | `{card_id,grade(0-5)}` 给评分 |
| `/memory/backup/stats` | GET | 5 池备份状态 |
| `/memory/backup/retry` | POST | 重试失败上传 |
| `/logs/recent` `/logs/stats` | GET | 日志 |

---

## 三、环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `BEE_BEARER_TOKEN` | dev-token-change-me | 鉴权 |
| `BEE_MEMORY_BACKUP_GITHUB_TOKEN` | (空) | GitHub 私库备份 |
| `BEE_MEMORY_BACKUP_TENCENT_*` | (空) | 腾讯 COS |
| `BEE_MEMORY_BACKUP_ALIYUN_*` | (空) | 阿里 OSS |
| `BEE_MEMORY_BACKUP_S3_*` | (空) | 通用 S3 |
| `BEE_MEMORY_BACKUP_LOCAL_DIR` | `D:/AI/_temp/memory_pool_local` | 本地池目录 |

---

## 四、调用示例

```python
import httpx
H = {"Authorization": "Bearer dev-token-change-me"}
BASE = "http://127.0.0.1:8004"

httpx.post(f"{BASE}/memory/store", json={
    "persona_id": "doc_chen",
    "kind": "knowledge_book",
    "content": "PDA 是急性心梗的常见类型...",
    "importance": 0.8, "novelty": 0.4,
}, headers=H, timeout=20)

r = httpx.post(f"{BASE}/memory/recall", json={
    "persona_id": "doc_chen",
    "query": "心梗如何快速识别",
    "k": 8, "strategy": "activation",
}, headers=H, timeout=20).json()
```

---

## 五、日志 & 数据

- 日志：`backend/data/logs/bee-memory.log`
- SQLite：`backend/data/memory.sqlite`
- 5 池备份目录：见 env 配置

---

## 六、故障

| 症状 | 修法 |
|---|---|
| 备份池失败 | 看 `/memory/backup/stats`；补对应 token |
| recall 返空 | persona_id 没存过；或激活阈值过高 |
| 复习卡总是 0 | 还没存过 kind=knowledge_*；或 due 都未到 |
