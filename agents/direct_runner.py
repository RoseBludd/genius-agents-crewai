"""Direct RunPod completions for GTM agents — same endpoints/tags as chat.geniuzs.com."""

from __future__ import annotations

import os
from typing import Any

import httpx

from agents.definitions import GENIUS_CONTEXT
from agents.runpod_compat import (
    ensure_max_tokens,
    normalize_completion,
    resolve_runpod_base,
    runpod_api_key,
    upstream_model_id,
)

SYSTEM_PROMPTS = {
    "strategist": (
        f"{GENIUS_CONTEXT} You are the GTM Strategist: plan campaigns, prioritize channels, "
        "sequence the calendar, and give concrete next actions."
    ),
    "content": (
        f"{GENIUS_CONTEXT} You are the GTM Content Writer: Reddit, HN, PH, email, and social copy "
        "in the right voice for each channel."
    ),
    "monitor": (
        f"{GENIUS_CONTEXT} You are the Monitor Agent: interpret PostHog analytics, flag funnel drop-offs, "
        "pricing page hot leads (3+ visits), and recommend outreach actions."
    ),
    "outreach": (
        f"{GENIUS_CONTEXT} You are the Outreach Agent: personalize cold and triggered emails for Emailzs. "
        'Return JSON: {"subject": "...", "messageText": "..."} with {{name}} and {{company}} merge tags.'
    ),
}

EXPECTED = {
    "strategist": "Structured GTM strategy with prioritized channels and next actions.",
    "content": "Ready-to-use marketing copy for the requested channel.",
    "monitor": "Concise monitor report with signals, drop-offs, hot leads, and recommended actions.",
    "outreach": 'JSON with "subject" and "messageText" keys.',
}


def run_agent_direct(agent: str, task: str, model: str | None = None) -> str:
    model_tag = upstream_model_id(model or os.getenv("GEMMA_MODEL", "gemma4:12b"))
    system = SYSTEM_PROMPTS.get(agent, SYSTEM_PROMPTS["strategist"])
    expected = EXPECTED.get(agent, "Clear, actionable response.")
    user = f"{task}\n\nExpected output: {expected}"

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
