"""
Baseline inference script for OmniSupport-Sim.
Compliant with OpenEnv stdout logging protocols.
"""
import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
import sys
import os

# ── Dynamic Path Injection (fix ModuleNotFoundError) ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Also add parent dir if we are inside omnisupport_sim
parent_dir = os.path.dirname(current_dir)
if os.path.basename(current_dir) == "omnisupport_sim":
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from client import OmniSupportEnv

try:
    from omnisupport_sim.models import OmniSupportObservation
except ImportError:
    try:
        from models import OmniSupportObservation
    except ImportError:
        # Fallback if pathing still issues
        pass

# ── Configuration ──
API_BASE_URL = os.getenv("API_BASE_URL") or "http://localhost:1234/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "qwen3.5-4b-python-coder"
IMAGE_NAME = os.getenv("IMAGE_NAME") or "omnisupport-sim:latest"
API_KEY = os.getenv("HF_TOKEN") or "dummy-key"
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000") # used if not using docker
BENCHMARK = os.getenv("OMNISUPPORT_BENCHMARK", "omnisupport_sim")
TASK_NAME = os.getenv("OMNISUPPORT_TASK", "order_check")

TIMEOUT_MINUTES = 19  # Must complete under 20 min
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Tier-2 Support Specialist AI agent for Team NerK. You solve customer support tickets by using tools.
    
    VERIFY-THEN-ACT PROTOCOL:
    1. ALWAYS call VerifyPolicy BEFORE issuing any refund or changing shipping.
    2. For return-related tickets: You MUST verify carrier delivery status via SearchDB with the tracking_id.
    3. Ensure you have ALL required information (order_id, eligibility, delivery status) before calling ExecuteAction.
    
    Available tools:
    1. SearchDB(query): Search the order/customer database. IMPORTANT: 'query' must be a STRING (e.g. "5510" or "cust_882"). DO NOT use dictionaries.
    2. VerifyPolicy(topic): Check company policy rules. Topics: refund_eligibility, escalation_protocol, return_verification, shipping_change.
    3. ExecuteAction(cmd, params): Execute an action. Commands: issue_refund. 'params' is a dictionary containing 'order_id'.
    4. FinalResponse(text): Close the ticket with a response to the customer.
    
    CRITICAL RULES:
    - Never hallucinate tracking IDs.
    - Respond with EXACTLY ONE tool call per turn as JSON.
    - JSON keys must be exactly: "action_type" and one of ["query", "topic", "cmd", "text"].
    
    Example: {"action_type": "search_db", "query": "TRK-9928-XZ"}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# Consolidated into get_agent_action

async def get_agent_action(obs: OmniSupportObservation, history: List[dict]) -> str:
    """Query the LLM and extract a structured action."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add history for context
    for entry in history:
        messages.append({"role": "user", "content": f"Output from last tool: {json.dumps(entry.get('last_tool_output', {}))}"})
    
    # Add current observation
    messages.append({"role": "user", "content": f"Current Observation: {obs.model_dump_json()}"})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
        
        # ── Extraction Logic ──
        cleaned_content = content
        if "```" in cleaned_content:
            # Multi-block check: take the first JSON-like block
            blocks = cleaned_content.split("```")
            for block in blocks:
                if "{" in block and "}" in block:
                    cleaned_content = block
                    if cleaned_content.startswith("json"):
                        cleaned_content = cleaned_content[4:]
                    break
        
        cleaned_content = cleaned_content.strip()

        try:
            action = json.loads(cleaned_content)
            
            # ── Normalization (Harding against model quirks) ──
            if isinstance(action, dict):
                # 1. Normalize case (FinalResponse -> final_response)
                if "action_type" in action:
                    action["action_type"] = action["action_type"].lower()
                
                # 2. Extract string from query if model incorrectly sent a dict
                if action.get("action_type") == "search_db" and isinstance(action.get("query"), dict):
                    q = action["query"]
                    # If model sent {"order_id": "5510"} or similar
                    action["query"] = str(next(iter(q.values()))) if q else ""
            
            return json.dumps(action)
        except json.JSONDecodeError as je:
            # LOG RAW OUTPUT ON FAILURE (to stderr for visibility in console)
            print(f"\n[DEBUG] LLM returned non-JSON content. Raw output follows:\n{'-'*40}\n{content}\n{'-'*40}", file=sys.stderr)
            raise je

    except Exception as e:
        # Fallback closure if JSON parse or LLM fails
        default_action = {"action_type": "final_response", "text": f"Unable to process at this time. Error: {str(e)}"}
        return json.dumps(default_action)


async def check_connectivity():
    """Verify that both the LLM and Env server are reachable."""
    print("Checking connectivity...", file=sys.stderr)
    
    # 1. Check Env
    try:
        from client import OmniSupportEnv
        test_env = OmniSupportEnv(base_url=ENV_URL)
        await test_env.reset()
        print(f"  [OK] Environment server at {ENV_URL} is reachable.", file=sys.stderr)
        await test_env.close()
    except Exception as e:
        print(f"  [ERROR] Environment server at {ENV_URL} is NOT reachable: {e}", file=sys.stderr)

    # 2. Check LLM
    try:
        client.models.list()
        print(f"  [OK] LLM server at {API_BASE_URL} is reachable.", file=sys.stderr)
    except Exception as e:
        print(f"  [ERROR] LLM server at {API_BASE_URL} is NOT reachable: {e}", file=sys.stderr)


async def main() -> None:
    global ENV_URL
    # Allow command line argument for ENV_URL
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        ENV_URL = sys.argv[1]
        print(f"Using ENV_URL from argument: {ENV_URL}", file=sys.stderr)

    # Use From Docker method if explicitly requested, else hit local env
    if os.getenv("USE_DOCKER"):
        import openenv.core
        env = await OmniSupportEnv.from_docker_image(IMAGE_NAME, env={"OMNISUPPORT_TASK": TASK_NAME})
    else:
        # Default to local server (useful for local development/testing)
        env = OmniSupportEnv(base_url=ENV_URL)

    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    await check_connectivity()
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        
        last_obs = result.observation.model_dump()
        last_error = None
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_json_str = await get_agent_action(result.observation, history)

            try:
                # Use TypeAdapter for standard Pydantic V2 Union validation
                from pydantic import TypeAdapter
                adapter = TypeAdapter(OmniSupportEnv.action_type)
                action_obj = adapter.validate_json(action_json_str)
                
                result = await env.step(action_obj)
                
                obs = result.observation.model_dump()
                reward = result.reward or 0.0
                done = result.done
                error = None
            except Exception as e:
                obs = last_obs
                reward = 0.0
                done = False
                error = str(e).replace("\n", " ") # Keep error on single line

            rewards.append(reward)
            steps_taken = step
            last_obs = obs
            last_error = error

            # Clean action logs for 1-liner STDOUT constraint
            clean_action_str = action_json_str.replace("\n", "").replace("\r", "")
            log_step(step=step, action=clean_action_str, reward=reward, done=done, error=error)

            if done:
                # In your previous logic, you used the total_reward / score logic, simplified here:
                score = sum(rewards)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        # final fallback score set if broke out early
        if score == 0.0 and len(rewards) > 0:
            score = sum(rewards)
            success = score >= SUCCESS_SCORE_THRESHOLD
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())