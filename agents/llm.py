import os

from langchain_openai import ChatOpenAI


def make_llm(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.getenv("GEMMA_MODEL", "gemma4:12b"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=2048,
    )
