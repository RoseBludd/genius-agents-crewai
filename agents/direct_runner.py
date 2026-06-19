"""Direct RunPod completions for GTM agents — same endpoints/tags as chat.geniuzs.com."""

from __future__ import annotations

import os
from typing import Any

import httpx

from agents.registry import EXPECTED, SYSTEM_PROMPTS
from agents.runpod_compat import (
    ensure_max_tokens,
    normalize_completion,
    resolve_runpod_base,
    runpod_api_key,
    upstream_model_id,
)


def run_agent_direct(agent: str, task: str, model: str | None = None, context: str | None = None) -> str:
    model_tag = upstream_model_id(model or os.getenv("GEMMA_MODEL", "gemma4:12b"))
    system = SYSTEM_PROMPTS.get(agent, SYSTEM_PROMPTS["strategist"])
    expected = EXPECTED.get(agent, "Clear, actionable response.")
    user_parts = [task]
    if context:
        user_parts.append(f"\n\nContext from prior steps or data sources:\n{context}")
    user_parts.append(f"\n\nExpected output: {expected}")
    user = "".join(user_parts)

    payload: dict[str, Any] = ensure_max_tokens(
        {
            "model": model_tag,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": int(os.getenv("CREWAI_MAX_TOKENS", "2048")),
        }
    )

    api_key = runpod_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or RUNPOD_API not configured")

    base = resolve_runpod_base(model_tag)
    with httpx.Client(timeout=300.0) as client:
        r = client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod error ({r.status_code}): {r.text[:300]}")
        data = normalize_completion(r.json())

    message = (data.get("choices") or [{}])[0].get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        raise RuntimeError("RunPod returned empty content after normalization")
    return text
