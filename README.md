# bee-memory (心脏)

三层记忆 + 知识图谱 + 宪法 + 遗忘曲线

## Run

\\\ash
cd backend
py -3.11 -m pip install -r requirements.txt
py -3.11 -m uvicorn app.main:app --reload --port 8004
\\\

## Auth

All routes except /healthz and /manifest require `Authorization: Bearer <BEE_BEARER_TOKEN>`.

## Status

Scaffold only. Real implementation tracks: plan v2 阶段 2-5 + v3 增量包.