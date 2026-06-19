"""RunPod OpenAI compat — same endpoints/tags as chat.geniuzs.com, CrewAI-safe responses."""

from __future__ import annotations

import os
from typing import Any


def upstream_model_id(model: str) -> str:
    """Strip openai/ prefix; keep Ollama tags unchanged (gemma4:12b, qwen3:8b, gemma4:31b)."""
    if not model:
        return os.getenv("GEMMA_MODEL", "gemma4:12b")
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def resolve_runpod_base(model: str) -> str:
    """Route model tags to the same RunPod endpoints as Genius chat (no endpoint changes)."""
    tag = upstream_model_id(model)
    if tag == "qwen3:8b":
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID_FAST") or os.getenv("RUNPOD_ENDPOINT_ID", "")
    elif tag == "gemma4:31b":
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID_EXTENDED") or os.getenv("RUNPOD_ENDPOINT_ID", "")
    else:
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")
    if not endpoint_id:
        raise RuntimeError("RUNPOD_ENDPOINT_ID missing from agents-crewai env")
    return f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"


def ensure_max_tokens(body: dict[str, Any], minimum: int = 1024) -> dict[str, Any]:
    payload = dict(body)
    max_tokens = payload.get("max_tokens")
    if not max_tokens or max_tokens < minimum:
        payload["max_tokens"] = minimum
    return payload


def normalize_completion(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ollama reasoning models (gemma4, qwen3) may return empty content with a reasoning field.
    Chat streams tokens into the UI; CrewAI needs a non-empty message.content.
    """
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        if content:
            continue
        reasoning = (message.get("reasoning") or "").strip()
        if reasoning:
            message = {**message, "content": reasoning}
            choice["message"] = message
    return data
