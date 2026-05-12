from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat
from dotenv import load_dotenv
from services.announcement_service import init_embeddings

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서비스 시작 시 임베딩 초기화
    await init_embeddings()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat.router)