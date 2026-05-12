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