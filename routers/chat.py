from fastapi import APIRouter
from pydantic import BaseModel
from services.openai_service import create_chat_reply, create_suggestions
from services.announcement_service import fetch_announcement_context

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

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
        reply = await create_chat_reply(messages, context_prompt)
        suggestions = await create_suggestions(messages, reply)
        return {"reply": reply, "suggestions": suggestions}
    except Exception as e:
        return {"error": str(e)}