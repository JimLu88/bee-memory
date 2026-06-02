# bee-memory (记忆中心) — 群晖/Docker 部署用. 端口 8004.
# 提供记忆/复习/5 池加密备份等能力. 依赖很轻 (fastapi/uvicorn/pydantic).
# 注意: /app/data 由 docker-compose 挂载卷持久化 (pool_config.json 等配置不随重建丢失).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
