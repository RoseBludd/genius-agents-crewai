"""Genius GTM Agents — CrewAI Strategist + Content on Railway, CADIS via RunPod."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agents.crew_runner import run_content_task, run_strategist_task

app = FastAPI(
    title="Genius GTM Agents",
    description="CrewAI Strategist + Content agents with OpenAI-compatible proxy to RunPod CADIS.",
    version="1.0.0",
)

API_KEY = os.getenv("LITELLM_MASTER_KEY", "GeniusAgents2026!")
OPENAI_BASE = os.getenv("OPENAI_API_BASE_URL", "").rstrip("/")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma4:12b")

AGENT_CATALOG = {
    "strategist": {
        "name": "Strategist Agent",
        "role": "Campaign Director",
        "model": "CADIS (RunPod)",
        "description": "Plans GTM calendar, messaging angles, and channel prioritization.",
    },
    "content": {
        "name": "Content Agent",
        "role": "Copy & Posts",
        "model": "CADIS (RunPod)",
        "description": "Writes Reddit, HN, PH, email, and social copy per channel voice.",
    },
}


def require_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> None:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if not token or token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class AgentRunRequest(BaseModel):
    task: str = Field(..., description="Natural language task for the agent")
    model: Optional[str] = Field(None, description="RunPod model id (default gemma4:12b)")


class AgentRunByTypeRequest(AgentRunRequest):
    agent: Literal["strategist", "content"] = Field(..., description="Agent to run")


@app.get("/health")
@app.get("/health/liveliness")
async def health():
    return {"status": "ok", "service": "agents-crewai", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/agents", dependencies=[Depends(require_key)])
async def list_agents():
    return {"agents": AGENT_CATALOG}


@app.post("/agents/run", dependencies=[Depends(require_key)])
async def run_agent(req: AgentRunByTypeRequest):
    return _execute_agent(req.agent, req.task, req.model)


@app.post("/agents/strategist", dependencies=[Depends(require_key)])
async def run_strategist(req: AgentRunRequest):
    return _execute_agent("strategist", req.task, req.model)


@app.post("/agents/content", dependencies=[Depends(require_key)])
async def run_content(req: AgentRunRequest):
    return _execute_agent("content", req.task, req.model)


def _execute_agent(agent: str, task: str, model: Optional[str]) -> dict[str, Any]:
    model_id = model or DEFAULT_MODEL
    try:
        if agent == "strategist":
            response = run_strategist_task(task, model_id)
        else:
            response = run_content_task(task, model_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc
    return {
        "status": "completed",
        "agent": agent,
        "response": response,
        "model": model_id,
    }


# --- OpenAI-compatible proxy (backward compat for LiteLLM clients) ---


@app.get("/v1/models", dependencies=[Depends(require_key)])
async def list_models():
    return {
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model", "owned_by": "runpod"}],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
async def chat_completions(body: dict[str, Any]):
    if not OPENAI_BASE or not OPENAI_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_BASE_URL / OPENAI_API_KEY not configured")
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            f"{OPENAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json=body,
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:500])
        return r.json()
