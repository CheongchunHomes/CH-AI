from fastapi import APIRouter
from pydantic import BaseModel
from services.openai_service import create_chat_reply, create_suggestions
from services.announcement_service import fetch_announcement_context
from services.policy_service import fetch_policy_context

router = APIRouter()

# TODO: 추후 /api/users/profile에서 실제 테이블 및 데이터로 교체
DUMMY_USER_CONTEXT = """
- 나이: 28세
- 연소득: 3,000만원
- 자산: 5,000만원
- 혼인여부: 미혼
- 무주택여부: 무주택
- 청약통장 가입기간: 24개월
- 거주지역: 경기도 수원시
- 희망지역: 경기도
"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel) :
    messages: list[ChatMessage]
    userContext: str | None = None
    pageContext: str | None = None

@router.post("/api/chat")
async def chat(request: ChatRequest):
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m.role in ("user", "assistant") and m.content.strip()
    ]

    if not messages:
        return {"error": "메시지를 하나 이상 보내주세요."}, 400

    # 사용자 마지막 메시지 추출
    user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # 공고 + 지원제도 컨텍스트 각각 검색 후 합치기
    context_prompt = await fetch_announcement_context(user_message)
    policy_context = await fetch_policy_context(user_message)
    combined_context = "\n\n".join(filter(None, [context_prompt, policy_context]))

    if request.pageContext:
        combined_context = request.pageContext + "\n\n" + combined_context

    try:
        # userContext가 FE에서 넘어오면 사용, 없으면 더미 데이터 사용
        user_context = request.userContext or DUMMY_USER_CONTEXT
        reply = await create_chat_reply(messages, combined_context, user_context)
        suggestions = await create_suggestions(messages, reply)

        # 답변 내용에 따라 관련 링크 자동 추가
        # reply user_message에 관련 키워드 있을 시 링크 포함
        links =[]
        if any(k in user_message + reply for k in ["공고", "행복주택", "전세임대", "매입임대","임대주택","청약"]):
            links.append({"label": "공고 전체보기", "url": "/site/announcements"})
        if any(k in user_message + reply for k in ["지원제도", "지원금", "보조금", "월세지원","주거급여","정책"]):
            links.append({"label": "지원제도 전체보기","url": "/site/policies"})

        return {"reply": reply, "suggestions": suggestions, "links": links}
    except Exception as e:
        return {"error": str(e)}
