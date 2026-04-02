"""
FastAPI server for OmniSupport-Sim.
Exposes /reset, /step, /state endpoints per OpenEnv spec.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from server.omnisupport_environment import OmniSupportEnvironment

app = FastAPI(
    title="OmniSupport-Sim",
    description="A High-Fidelity OpenEnv for Multi-Tool Support Agents",
    version="1.0.0",
)

env = OmniSupportEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "order_check"

class StepRequest(BaseModel):
    action_type: str
    query: Optional[str] = None
    topic: Optional[str] = None
    cmd: Optional[str] = None
    params: Optional[dict] = None
    text: Optional[str] = None


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    req = request or ResetRequest()
    try:
        result = env.reset(task_id=req.task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """Execute a single agent action."""
    try:
        action = request.model_dump(exclude_none=True)
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_state():
    """Get the full environment state for grading/debugging."""
    return env.state()


@app.get("/health")
async def health():
    return {"status": "healthy", "env_id": "omnisupport-sim-v1"}

# ── Serve frontend if available ──
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    @app.get("/web")
    async def web_interface():
        return FileResponse(os.path.join(frontend_path, "code.html"))

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()