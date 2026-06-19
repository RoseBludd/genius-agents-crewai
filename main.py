"""Genius GTM Agents — 7 specialized agents + multi-agent pipelines on Railway."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TELEMETRY", "false")

from agents.registry import AGENT_CATALOG, PIPELINE_CATALOG  # noqa: E402

app = FastAPI(
    title="Genius GTM Agents",
    description="Seven GTM agents with direct RunPod + CrewAI pipelines, OpenAI-compatible proxy.",
    version="2.0.0",
)

API_KEY = os.getenv("LITELLM_MASTER_KEY", "GeniusAgents2026!")
OPENAI_BASE = os.getenv("OPENAI_API_BASE_URL", "").rstrip("/")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma4:12b")
INTERNAL_BASE = os.getenv(
    "CREWAI_OPENAI_BASE_URL",
    f"http://127.0.0.1:{os.getenv('PORT', '8080')}/v1/internal",
).rstrip("/")

AgentId = Literal[
    "strategist", "content", "visual", "monitor", "outreach", "seo", "publisher",
]
PipelineId = Literal[
    "campaign", "launch", "intelligence", "seo_content", "publish_ready", "full_gtm",
]
ExecutionMode = Literal["direct", "crew"]


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
    context: Optional[str] = Field(None, description="Optional context (PostHog data, prior agent output)")
    mode: Optional[ExecutionMode] = Field(None, description="direct (fast) or crew (CrewAI orchestration)")


class AgentRunByTypeRequest(AgentRunRequest):
    agent: AgentId = Field(..., description="Agent to run")


class PipelineRunRequest(BaseModel):
    pipeline: PipelineId = Field(..., description="Predefined multi-agent pipeline")
    task: str = Field(..., description="Campaign or content goal")
    model: Optional[str] = Field(None, description="RunPod model id")
    mode: Optional[ExecutionMode] = Field(None, description="direct or crew")
    context: Optional[str] = Field(None, description="Injected data (e.g. PostHog JSON for intelligence)")
    generate_image: bool = Field(False, description="Generate FLUX image after visual step (launch/full_gtm)")
    image_width: int = Field(1024, ge=256, le=2048)
    image_height: int = Field(1024, ge=256, le=2048)


class VisualGenerateRequest(BaseModel):
    prompt: str = Field(..., description="FLUX image prompt")
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    task: Optional[str] = Field(None, description="Optional brief for Visual Agent before generation")


class CrewRunRequest(BaseModel):
    agents: list[AgentId] = Field(..., min_length=1, max_length=7)
    task: str = Field(...)
    model: Optional[str] = None


@app.get("/health")
@app.get("/health/liveliness")
async def health():
    return {
        "status": "ok",
        "service": "agents-crewai",
        "agents": len(AGENT_CATALOG),
        "pipelines": len(PIPELINE_CATALOG),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/agents", dependencies=[Depends(require_key)])
async def list_agents():
    return {"agents": AGENT_CATALOG, "pipelines": PIPELINE_CATALOG}


@app.post("/agents/run", dependencies=[Depends(require_key)])
def run_agent_by_type(req: AgentRunByTypeRequest):
    return _execute_agent(req.agent, req.task, req.model, req.context, req.mode)


@app.post("/agents/pipeline/run", dependencies=[Depends(require_key)])
def run_pipeline_endpoint(req: PipelineRunRequest):
    from agents.orchestrator import run_pipeline

    try:
        result = run_pipeline(
            req.pipeline,
            req.task,
            req.model,
            req.mode,
            req.context,
            req.generate_image,
            req.image_width,
            req.image_height,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {exc}") from exc
    return result


@app.post("/agents/crew/run", dependencies=[Depends(require_key)])
def run_crew_endpoint(req: CrewRunRequest):
    from agents.crew_runner import run_multi_agent_crew

    try:
        response = run_multi_agent_crew(req.agents, req.task, req.model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Crew failed: {exc}") from exc
    return {
        "status": "completed",
        "agents": req.agents,
        "response": response,
        "model": req.model or DEFAULT_MODEL,
        "via": "crewai_crew",
    }


@app.post("/agents/visual/generate", dependencies=[Depends(require_key)])
def visual_generate(req: VisualGenerateRequest):
    """Visual Agent brief (optional) + RunPod FLUX image generation."""
    from agents.direct_runner import run_agent_direct
    from agents.image_gen import generate_flux_image

    brief = None
    if req.task:
        try:
            brief = run_agent_direct("visual", req.task, None, None)
        except Exception as exc:
            brief = f"(brief failed: {exc})"

    prompt = req.prompt
    if brief and not req.prompt:
        prompt = brief[:500]

    try:
        image = generate_flux_image(prompt, req.width, req.height)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}") from exc

    return {
        "status": "completed",
        "agent": "visual",
        "brief": brief,
        "prompt": prompt,
        "image": image,
    }


@app.post("/agents/{agent_id}", dependencies=[Depends(require_key)])
def run_agent_endpoint(agent_id: AgentId, req: AgentRunRequest):
    return _execute_agent(agent_id, req.task, req.model, req.context, req.mode)


def _execute_agent(
    agent: str,
    task: str,
    model: Optional[str],
    context: Optional[str] = None,
    mode: Optional[ExecutionMode] = None,
) -> dict[str, Any]:
    model_id = model or DEFAULT_MODEL
    exec_mode = (mode or os.getenv("CREWAI_EXECUTION_MODE", "direct")).lower()

    try:
        if exec_mode == "crew":
            from agents.crew_runner import run_agent_crew

            response = run_agent_crew(agent, task, model_id, context)
            via = "crewai"
        else:
            from agents.direct_runner import run_agent_direct

            response = run_agent_direct(agent, task, model_id, context)
            via = "runpod_direct"
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc

    return {
        "status": "completed",
        "agent": agent,
        "response": response,
        "model": model_id,
        "via": via,
    }


# --- OpenAI-compatible proxy (backward compat for LiteLLM clients) ---


@app.get("/v1/models", dependencies=[Depends(require_key)])
async def list_models():
    return {
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model", "owned_by": "runpod"}],
    }


@app.post("/v1/internal/chat/completions")
async def internal_chat_completions(body: dict[str, Any]):
    """CrewAI → RunPod proxy with the same model routing as chat.geniuzs.com."""
    from agents.runpod_compat import (
        ensure_max_tokens,
        normalize_completion,
        resolve_runpod_base,
        runpod_api_key,
        upstream_model_id,
    )

    api_key = runpod_api_key() or OPENAI_KEY
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY or RUNPOD_API not configured")

    model = upstream_model_id(body.get("model", DEFAULT_MODEL))
    payload = ensure_max_tokens(body)
    payload["model"] = model
    base = resolve_runpod_base(model)

    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:500])
        return normalize_completion(r.json())


@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
async def chat_completions(body: dict[str, Any]):
    return await internal_chat_completions(body)
