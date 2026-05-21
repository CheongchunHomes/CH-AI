import pickle
import httpx
import os
import numpy as np
from openai import AsyncOpenAI

# 캐시 파일 경로 상수
# 임베딩 결과를 파일로 저장 > 서버 재시작시 재임베딩 불필요
ANNOUNCEMENT_CACHE_FILE = "cache/announcement_embeddings.pkl"
POLICY_CACHE_FILE = "cache/policy_embeddings.pkl"

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 서버가 살아있는 동안 메모리에 유지
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
        a.get('targetType', ''),
        a.get('recuitmentType', ''),
        a.get('supplyInstitution', ''),
    ]))

async def get_embedding(text: str) -> list[float]:
    # 텍스트를 벡터(숫자 배열)로 변환
    res = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding

async def fetch_all_announcements() -> list[dict]:
    # BE /announcements API를 페이지 단위로 전부 가져옴
    all_items = []
    page = 0
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

    # 캐시 파일이 이미 있으면 파일에서 불러옴 > 임베딩 API 호출 생략
    if os.path.exists(ANNOUNCEMENT_CACHE_FILE):
        print("[공고 임베딩 캐시 로드]")
        with open(ANNOUNCEMENT_CACHE_FILE, "rb") as f:
            # pickle.load = 저장된 Python 객체를 파일에서 읽어옴
            cache = pickle.load(f)
            _announcement_items = cache["items"]
            _announcement_vectors = cache["vectors"]
        print(f"[공고 캐시 로드 완료] {len(_announcement_items)}건")
        return  # 여기서 함수 종료 > 아래 임베딩 코드 실행 안 함

    # 캐시 없을 때만 새로 임베딩
    print("[공고 임베딩 초기화 시작]")
    items = await fetch_all_announcements()
    print(f"[공고 총 건수] {len(items)}건")

    if not items:
        return

    texts = [item_to_text(a) for a in items]

    all_embeddings = []

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in res.data])
        print(f"[임베딩 진행] {min(i + batch_size, len(texts))}/{len(texts)}")

    _announcement_items = items
    _announcement_vectors = np.array(all_embeddings, dtype=np.float32)
    print("[임베딩 초기화 완료]")

    # 임베딩 결과를 파일로 저장
    os.makedirs("cache", exist_ok=True)
    with open(ANNOUNCEMENT_CACHE_FILE, "wb") as f:
        pickle.dump({"items": _announcement_items, "vectors": _announcement_vectors}, f)
    print("[공고 임베딩 캐시 저장 완료]")


async def fetch_announcement_context(user_message: str = "") -> str | None:
    global _announcement_items, _announcement_vectors

    if not _announcement_items:
        print("[임베딩 캐시 없음]")
        return None

    if not user_message.strip():
        # 질문 없으면 최신 10개 반환
        items = _announcement_items[:10]
    else:
        # 질문을 벡터로 변환해서 저장된 공고 벡터들과 유사도 비교
        query_vector = np.array(await get_embedding(user_message), dtype=np.float32)
        similarities = cosine_similarity(query_vector, _announcement_vectors)

        # 유사도 높은 순서로 정렬 후 상위 20개 인덱스 추출
        # np.argsort = 오름차순 정렬 인덱스 반환
        # [::-1] = 뒤집어서 내림차순으로 만듦
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
            f" 지역: {a.get('region', '전국')} / 주소: {a.get('address', '')}",
            f" 상태: {a.get('status', '')} / 신청기간: {a.get('applyStartDate', '')}~{a.get('applyEndDate', '')}",
        ]))

    return "\n\n".join(lines)