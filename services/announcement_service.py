import httpx
import os
import numpy as np
from openai import  AsyncOpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 메모리 캐시
_announcement_items: list[dict] = []
_announcement_vectors: np.ndarray | None = None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return np.dot(b, a) / (norm_b * norm_a + 1e-10)

def item_to_text(a: dict) -> str:
    return " ".join(filter(None, [
        a.get('title', ''),
        a.get('region', ''),
        a.get('address', ''),
        a.get('status', ''),
        a.get('targetType',''),
        a.get('recuitmentType', ''),
        a.get('supplyInstitution', ''),
    ]))

async def get_embedding(text: str) -> list[float]:
    res = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding

async def fetch_all_announcements() -> list[dict]:
    all_items = []
    page=0
    async with httpx.AsyncClient() as http:
        while True:
            res = await http.get(
                f"{API_BASE_URL}/announcements",
                params={"size": 100, "page": page},
                timeout=10.0
            )
            if res.status_code != 200:
                break
            data = res.json()
            items = data.get("content", [])
            all_items.extend(items)
            if data.get("last", True):
                break
            page += 1
    return all_items

async def init_embeddings():
    global _announcement_items, _announcement_vectors

    print("[임베딩 초기화 시작]")
    # DB에서 공고 전부 가져옴
    items = await fetch_all_announcements()
    print(f"[공고 총 건수] {len(items)}건")

    if not items:
        return

    texts = [item_to_text(a) for a in items]

    # 배치로 임베딩 (최대 100개씩)
    all_embeddings = []
    batch_size = 100
    for i in range (0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # OpenAI에 텍스트로 보내서 백터로 변환 (100개씩 배치)
        res = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in res.data])
        print(f"[임베딩 진행] {min(i + batch_size, len(texts))}/{len(texts)}")

    # 원본 공고 데이터
    _announcement_items = items
    # 벡터 배열
    _announcement_vectors = np.array(all_embeddings, dtype=np.float32)
    print("[임베딩 초기화 완료]")

async def fetch_announcement_context(user_message: str = "") -> str | None:
    global _announcement_items, _announcement_vectors

    if _announcement_items is None or len(_announcement_items) == 0:
        print("[임베딩 캐시 없음]")
        return None

    if not user_message.strip():
        # 질문 없을 시 최신 10개
        items = _announcement_items[:10]
    else:
        # 질문 벡터화
        query_vector = np.array(await get_embedding(user_message), dtype=np.float32)

        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, _announcement_vectors)

        # 유사도 높은 Top 10 노출
        top_indices = np.argsort(similarities)[::-1][:20]
        seen_titles = set()
        items = []
        for i in top_indices:
            title = _announcement_items[i].get('title', '')
            if title not in seen_titles:
                seen_titles.add(title)
                items.append(_announcement_items[i])
                if len(items) >= 10:
                    break

        print(f"[Top 10 유사 공고]")
        for i in top_indices[:3]:
            print(f" - {_announcement_items[i].get('title')}  ({similarities[i]:.3f})")


    lines = []
    for a in items:
        lines.append("\n".join([
            f"- {a.get('title', '')}",
            f" 지역: {a.get('region', '전국')} / 주소: {a.get('address','')}",
            f" 상태: {a.get('status', '')} / 신청기간: {a.get('applyStartDate', '')}~{a.get('applyEndDate', '')}",
        ]))

    return "\n\n".join(lines)
