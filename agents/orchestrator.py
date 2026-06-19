"""Multi-agent pipelines — direct chain (default) or CrewAI crew mode."""

from __future__ import annotations

import os
from typing import Any, Literal

from agents.crew_runner import run_agent_crew
from agents.direct_runner import run_agent_direct
from agents.image_gen import generate_flux_image
from agents.registry import PIPELINE_CATALOG, PipelineId

ExecutionMode = Literal["direct", "crew"]


def _execution_mode(requested: ExecutionMode | None) -> ExecutionMode:
    if requested:
        return requested
    env = os.getenv("CREWAI_EXECUTION_MODE", "direct").lower()
    return "crew" if env == "crew" else "direct"


def _run_step(
    agent: str,
    task: str,
    model: str | None,
    context: str | None,
    mode: ExecutionMode,
) -> str:
    if mode == "crew":
        return run_agent_crew(agent, task, model, context)
    return run_agent_direct(agent, task, model, context)


def _pipeline_task(pipeline: PipelineId, agent: str, original_task: str, step_index: int) -> str:
    if step_index == 0:
        return original_task
    labels = {
        "strategist": "Using the campaign strategy above,",
        "content": "Using the strategy and context above, write the requested marketing copy.",
        "visual": "Using the content and strategy above, design the visual asset brief.",
        "monitor": "Using all prior GTM outputs above, summarize what to watch in analytics.",
        "outreach": "Using the monitor insights and lead signals above, draft outreach.",
        "seo": "Using the task and any prior context, produce the SEO research.",
        "publisher": "Using the polished content above, produce the publish plan JSON.",
    }
    prefix = labels.get(agent, "Continue the GTM workflow:")
    return f"{prefix}\n\nOriginal request: {original_task}"


def run_pipeline(
    pipeline: PipelineId,
    task: str,
    model: str | None = None,
    mode: ExecutionMode | None = None,
    context: str | None = None,
    generate_image: bool = False,
    image_width: int = 1024,
    image_height: int = 1024,
) -> dict[str, Any]:
    spec = PIPELINE_CATALOG.get(pipeline)
    if not spec:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    exec_mode = _execution_mode(mode)
    agents: list[str] = spec["agents"]
    steps: list[dict[str, Any]] = []
    accumulated = context or ""

    for i, agent_id in enumerate(agents):
        step_task = _pipeline_task(pipeline, agent_id, task, i)
        response = _run_step(agent_id, step_task, model, accumulated or None, exec_mode)
        steps.append({"agent": agent_id, "response": response})
        accumulated = (
            f"{accumulated}\n\n--- {agent_id} ---\n{response}".strip()
            if accumulated
            else response
        )

    result: dict[str, Any] = {
        "status": "completed",
        "pipeline": pipeline,
        "mode": exec_mode,
        "steps": steps,
        "summary": steps[-1]["response"] if steps else "",
        "model": model or os.getenv("GEMMA_MODEL", "gemma4:12b"),
    }

    if generate_image and "visual" in agents:
        visual_text = next((s["response"] for s in steps if s["agent"] == "visual"), "")
        prompt = _extract_flux_prompt(visual_text) or task
        try:
            result["image"] = generate_flux_image(prompt, image_width, image_height)
        except Exception as exc:
            result["image_error"] = str(exc)

    return result


def _extract_flux_prompt(visual_response: str) -> str | None:
    for line in visual_response.splitlines():
        lower = line.lower()
        if "flux prompt" in lower or "prompt:" in lower:
            _, _, rest = line.partition(":")
            if rest.strip():
                return rest.strip()
    if len(visual_response) > 40:
        return visual_response[:500]
    return None
