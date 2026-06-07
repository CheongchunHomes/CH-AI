from contextlib import asynccontextmanager
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat
from dotenv import load_dotenv
from services.announcement_service import init_embeddings, ANNOUNCEMENT_CACHE_FILE, POLICY_CACHE_FILE
from services.policy_service import init_policy_embeddings

load_dotenv()

async def schedule_reindex():
    # 1시간(3600초)마다 캐시 삭제 후 재임베딩
    while True:
        await asyncio.sleep(3600)
        print("[캐시 삭제 후 재임베딩]")
        if os.path.exists(ANNOUNCEMENT_CACHE_FILE):
            os.remove(ANNOUNCEMENT_CACHE_FILE)
        if os.path.exists(POLICY_CACHE_FILE):
            os.remove(POLICY_CACHE_FILE)
        await init_embeddings()
        await init_policy_embeddings()
        print("[재임베딩 완료]")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 임베딩 초기화 (캐시 있으면 캐시 사용)
    await init_embeddings()
    await init_policy_embeddings()
    # asyncio.create_task = 백그라운드에서 schedule_reindex 실행
    task = asyncio.create_task(schedule_reindex())
    yield
    # 서버 종료 시 백그라운드 태스크 정리
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat.router)
