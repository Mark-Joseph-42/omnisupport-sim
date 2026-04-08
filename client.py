import sys
import os
import httpx
from openenv.core.env_client import StepResult

try:
    from omnisupport_sim.models import OmniSupportAction, OmniSupportObservation
except ImportError:
    # If running from within the package or if the package is not found, try local import
    try:
        from models import OmniSupportAction, OmniSupportObservation
    except ImportError:
        # Final fallback: Add current and parent dir to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models import OmniSupportAction, OmniSupportObservation

class OmniSupportEnv:
    """Async environment client for OmniSupport-Sim, using HTTP."""
    action_type = OmniSupportAction
    observation_type = OmniSupportObservation

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def reset(self, task_id: str = "order_check") -> StepResult:
        """Call the /reset endpoint and track state."""
        response = await self.client.post("/reset", json={"task_id": task_id})
        response.raise_for_status()
        data = response.json()
        return self._parse_result(data)

    async def step(self, action: OmniSupportAction) -> StepResult:
        """Call the /step endpoint with a typed action."""
        payload = action.model_dump(exclude_none=True)
        response = await self.client.post("/step", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._parse_result(data)

    async def close(self):
        """Cleanup connection."""
        await self.client.aclose()

    def _parse_result(self, data: dict) -> StepResult:
        """Parse result into a StepResult."""
        obs_data = data.get("observation", {})
        obs = OmniSupportObservation.model_validate(obs_data)
        # Note: StepResult in openenv-core 0.1.0 doesn't take an info field
        return StepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False)
        )

    # For from_docker_image compatibility — return an instance pointed to the local Docker port
    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs):
        """Minimal mock for local development when Docker isn't available."""
        # For this hackathon, we assume the server is running on 8000 when Docker fails.
        return cls(base_url="http://localhost:8000")
