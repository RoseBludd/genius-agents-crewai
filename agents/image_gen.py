"""RunPod ComfyUI FLUX image generation for the Visual Agent."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx


def _workflow_kind() -> str:
    img = os.getenv("RUNPOD_COMFYUI_IMAGE", "")
    if "dev-fp8" in img or "flux1-dev" in img:
        return "flux-dev-fp8"
    if "schnell" in img:
        return "flux-schnell"
    return os.getenv("RUNPOD_IMAGE_WORKFLOW", "flux-schnell")


def _build_schnell_workflow(prompt: str, width: int, height: int) -> dict[str, Any]:
    url = (
        "https://raw.githubusercontent.com/runpod-workers/worker-comfyui/main/"
        "test_resources/workflows/workflow_flux1_schnell.json"
    )
    with httpx.Client(timeout=60.0) as client:
        payload = client.get(url).json()
    workflow = payload["input"]["workflow"]
    workflow["6"]["inputs"]["text"] = prompt
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height
    return {"input": {"workflow": workflow}}


def _build_dev_workflow(prompt: str, width: int, height: int) -> dict[str, Any]:
    url = (
        "https://raw.githubusercontent.com/runpod-workers/worker-comfyui/main/"
        "test_resources/workflows/flux_dev_checkpoint_example.json"
    )
    with httpx.Client(timeout=60.0) as client:
        workflow = client.get(url).json()
    workflow["6"]["inputs"]["text"] = prompt
    workflow["27"]["inputs"]["width"] = width
    workflow["27"]["inputs"]["height"] = height
    return {"input": {"workflow": workflow}}


def generate_flux_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    timeout_sec: int = 1200,
) -> dict[str, Any]:
    """Submit FLUX job to RunPod ComfyUI endpoint; return job metadata + base64 images."""
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID_IMAGE", "")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RUNPOD_API") or ""
    if not endpoint_id:
        raise RuntimeError("RUNPOD_ENDPOINT_ID_IMAGE not configured")
    if not api_key:
        raise RuntimeError("RUNPOD_API or OPENAI_API_KEY not configured")

    kind = _workflow_kind()
    if kind == "flux-dev-fp8":
        body = _build_dev_workflow(prompt, width, height)
    else:
        body = _build_schnell_workflow(prompt, width, height)

    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=120.0) as client:
        job = client.post(f"{base}/run", headers=headers, json=body).json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod image job failed to start: {job}")

        deadline = time.time() + timeout_sec
        status = job
        while time.time() < deadline:
            status = client.get(f"{base}/status/{job_id}", headers=headers).json()
            state = status.get("status", "")
            if state == "COMPLETED":
                output = status.get("output") or {}
                images = output.get("images") or []
                if not images:
                    raise RuntimeError("FLUX job completed with no images")
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "workflow": kind,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "images": images,
                    "execution_time_ms": status.get("executionTime"),
                }
            if state == "FAILED":
                raise RuntimeError(f"FLUX job failed: {status}")
            time.sleep(15)

    raise RuntimeError(f"FLUX job timed out after {timeout_sec}s (last status: {status.get('status')})")
