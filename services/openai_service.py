import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from config.prompt import SYSTEM_PROMPT

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def create_chat_reply(messages: list[dict], context_prompt: str | None = None) -> str:
    enriched_prompt = (
        f"{SYSTEM_PROMPT}\n\n[현재 공고]\n{context_prompt}"
        if context_prompt
        else SYSTEM_PROMPT
    )

    response = await client.responses.create(
        model="gpt-4.1-mini",
        instructions=enriched_prompt,
        input=messages,
        store=False,
    )

    return response.output_text

async  def create_suggestions(messages: list[dict], last_reply: str) -> list[str]:
    try:
        response = await client.responses.create(
            model="gpt-4.1-mini",
            instructions=(
                "You are a youth housing consultation AI. "
                "Based on the conversation that just took place, generate 3 questions the user is likely to ask next. "
                "Each question should be short, specific, and returned on a separate line without numbering. "
                "You must return the response only in JSON array format. Example: [\"Question1\", \"Question2\", \"Question3\"] "
                "Do not include any other text under any circumstances. "
                "Respond only in Korean."
            ),
            input=messages + [{"role": "assistant", "content": last_reply}],
            store=False,
        )
        import json
        text = response.output_text.strip()
        return json.loads(text)
    except Exception as e:
        print(f"[suggestions 생성실패] {e}]")
        return[]