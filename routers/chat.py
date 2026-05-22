from fastapi import APIRouter
from pydantic import BaseModel
from services.openai_service import create_chat_reply, create_suggestions
from services.announcement_service import fetch_announcement_context
from services.policy_service import fetch_policy_context

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
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

    user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    context_prompt = await fetch_announcement_context(user_message)
    policy_context = await fetch_policy_context(user_message)
    combined_context = "\n\n".join(filter(None, [context_prompt, policy_context]))

    if request.pageContext:
        combined_context = request.pageContext + "\n\n" + combined_context

    try:
        user_context = request.userContext or ""
        reply = await create_chat_reply(messages, combined_context, user_context)
        suggestions = await create_suggestions(messages, reply)

        links = []
        if any(k in user_message + reply for k in ["공고", "행복주택", "전세임대", "매입임대", "임대주택", "청약"]):
            links.append({"label": "공고 전체보기", "url": "/site/announcements"})
        if any(k in user_message + reply for k in ["지원제도", "지원금", "보조금", "월세지원", "주거급여", "정책"]):
            links.append({"label": "지원제도 전체보기", "url": "/site/policies"})

        return {"reply": reply, "suggestions": suggestions, "links": links}
    except Exception as e:
        return {"error": str(e)}