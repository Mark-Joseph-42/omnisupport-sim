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
    1. SearchDB(query): Search the order/customer database. Use this to find order details and CARRIER tracking status.
    2. VerifyPolicy(topic): Check company policy rules. Topics: refund_eligibility, escalation_protocol, return_verification, shipping_change.
    3. ExecuteAction(cmd, params): Execute an action. Commands: issue_refund, change_shipping. Params MUST include order_id.
    4. FinalResponse(text): Close the ticket with a response to the customer.
    
    CRITICAL RULES:
    - Never hallucinate tracking IDs or order statuses.
    - If a refund condition is not met (e.g., item > 14 days old), use FinalResponse to explain why to the customer.
    - Respond with EXACTLY ONE tool call per turn as JSON.
    
    Example: {"action_type": "verify_policy", "topic": "refund_eligibility"}
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


def llm_decide(observation: dict, error_msg: Optional[str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if error_msg:
        messages.append({
            "role": "user",
            "content": f"Current ticket observation:\n{json.dumps(observation, indent=2)}\n\nThe last action resulted in an error:\n{error_msg}\n\nWhat is your next action? Respond with JSON only."
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Current ticket observation:\n{json.dumps(observation, indent=2)}\n\nWhat is your next action? Respond with JSON only."
        })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        # Test parse
        action = json.loads(content)
        return json.dumps(action) # ensure single line format for stdout string
    except Exception as e:
        # Fallback closure if JSON parse or LLM fails
        default_action = {"action_type": "final_response", "text": f"Unable to process at this time. Error: {str(e)}"}
        return json.dumps(default_action)


async def main() -> None:
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

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        
        last_obs = result.observation.model_dump()
        last_error = None
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_json_str = llm_decide(last_obs, last_error)

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