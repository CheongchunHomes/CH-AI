import pickle
import httpx
import os
import numpy as np
from openai import AsyncOpenAI

from services.announcement_service import POLICY_CACHE_FILE

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_policy_items: list[dict] = []
_policy_vectors: np.ndarray | None = None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return np.dot(b, a) / (norm_b * norm_a + 1e-10)

def item_to_text(p: dict) -> str:
    return " ".join(filter(None, [
        p.get('title', ''),
        p.get('region', ''),
        p.get('mainCategory', ''),
        p.get('subCategory', ''),
        p.get('summary', ''),
        p.get('targetDesc', ''),
        p.get('supportType', ''),
        p.get('supervisingInstitution', ''),
    ]))

async def fetch_all_policies() -> list[dict]:
    all_items = []
    page = 0
    async with httpx.AsyncClient() as http:
        while True:
            res = await http.get(
                f"{API_BASE_URL}/policies",
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

async def init_policy_embeddings():
    global _policy_items, _policy_vectors

    # 캐시 있으면 불러오기
    if os.path.exists(POLICY_CACHE_FILE):
        print("[지원제도 임베딩 캐시 로드]")
        with open(POLICY_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
            _policy_items = cache["items"]
            _policy_vectors = cache["vectors"]
        print(f"[지원제도 캐시 로드 완료] {len(_policy_items)}건")
        return

    # 캐시 없을 때만 새로 임베딩
    print("[지원제도 임베딩 초기화 시작]")
    items = await fetch_all_policies()
    print(f"[지원제도 총 건수] {len(items)}건")

    if not items:
        return

    texts = [item_to_text(p) for p in items]

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in res.data])
        print(f"[지원제도 임베딩 진행] {min(i + batch_size, len(texts))}/{len(texts)}")

    _policy_items = items
    _policy_vectors = np.array(all_embeddings, dtype=np.float32)
    print("[지원제도 임베딩 초기화 완료]")

    os.makedirs("cache", exist_ok=True)
    with open(POLICY_CACHE_FILE, "wb") as f:
        pickle.dump({"items": _policy_items, "vectors": _policy_vectors}, f)
    print("[지원제도 임베딩 캐시 저장 완료]")

async def fetch_policy_context(user_message: str = "") -> str | None:
    global _policy_items, _policy_vectors

    if not _policy_items:
        return None

    if not user_message.strip():
        items = _policy_items[:5]
    else:
        # 메모리에 있는 벡터(_policy_vectors)와 비교 > DB 재호출 없음
        query_vector = np.array(
            (await client.embeddings.create(
                model="text-embedding-3-small",
                input=user_message
            )).data[0].embedding,
            dtype=np.float32
        )

        similarities = cosine_similarity(query_vector, _policy_vectors)
        top_indices = np.argsort(similarities)[::-1][:10]

        seen_titles = set()
        items = []
        for i in top_indices:
            title = _policy_items[i].get('title', '')
            if title not in seen_titles:
                seen_titles.add(title)
                items.append(_policy_items[i])
                if len(items) >= 5:
                    break

        print(f"[Top 5 유사 지원제도]")
        for i in top_indices[:3]:
            print(f" - {_policy_items[i].get('title')} ({similarities[i]:.3f})")

    lines = []
    for p in items:
        lines.append("\n".join([
            f"- {p.get('title', '')}",
            f" 지역: {p.get('region', '전국')} / 분류: {p.get('mainCategory', '')} > {p.get('subCategory', '')}",
            f" 대상: {p.get('targetDesc', '')}",
            f" 요약: {p.get('summary', '')}",
            f" 신청기간: {p.get('applyPeriod', '')} / 상태: {p.get('status', '')}",
        ]))

    return "\n\n".join(lines)