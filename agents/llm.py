import os

from crewai import LLM


def make_llm(model: str | None = None) -> LLM:
    model_id = model or os.getenv("GEMMA_MODEL", "gemma4:12b")
    if "/" not in model_id:
        model_id = f"openai/{model_id}"
    return LLM(
        model=model_id,
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
    )
