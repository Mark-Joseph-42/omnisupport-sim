from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import os

from server.omnisupport_environment import OmniSupportEnvironment

app = FastAPI(
    title="OmniSupport-Sim",
    description="A High-Fidelity OpenEnv for Multi-Tool Support Agents",
    version="1.0.0",
)

# ── CORS Middleware for Hugging Face Mirrors ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = OmniSupportEnvironment()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler so unhandled errors return structured JSON."""
    return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


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

@app.get("/")
async def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()