from fastapi import APIRouter
from pydantic import BaseModel
from services.openai_service import create_chat_reply, create_suggestions
from services.announcement_service import fetch_announcement_context

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

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    userContext: str | None = None

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
        (m["content"]for m in reversed(messages) if m["role"] == "user"), ""
    )
    context_prompt = await fetch_announcement_context(user_message)

    try:
        user_context = request.userContext or DUMMY_USER_CONTEXT
        reply = await create_chat_reply(messages, context_prompt, user_context)
        suggestions = await create_suggestions(messages, reply)
        return {"reply": reply, "suggestions": suggestions}
    except Exception as e:
        return {"error": str(e)}