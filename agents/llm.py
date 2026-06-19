import os

from crewai import LLM


def internal_openai_base() -> str:
    port = os.getenv("PORT", "8080")
    return os.getenv("CREWAI_OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1/internal").rstrip("/")


def make_llm(model: str | None = None) -> LLM:
    model_id = model or os.getenv("GEMMA_MODEL", "gemma4:12b")
    if "/" not in model_id:
        model_id = f"openai/{model_id}"
    return LLM(
        model=model_id,
        base_url=internal_openai_base(),
        api_key=os.getenv("OPENAI_API_KEY") or "runpod",
        temperature=0.7,
        max_tokens=int(os.getenv("CREWAI_MAX_TOKENS", "2048")),
    )
