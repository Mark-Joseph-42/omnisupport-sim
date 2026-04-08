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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("url_arg", nargs="?", default=None)
parser.add_argument("--task", default=os.getenv("OMNISUPPORT_TASK", "fraud_mitigation"))
args, unknown = parser.parse_known_args()

ENV_URL = args.url_arg or os.getenv("ENV_URL", "http://localhost:8000")
TASK_NAME = args.task
BENCHMARK = os.getenv("OMNISUPPORT_BENCHMARK", "omnisupport_sim")

TIMEOUT_MINUTES = 19  # Must complete under 20 min
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Tier-2 Support AI. You solve tickets using JSON tool calls.
    
    TOOLS:
    1. search_db: query (string) - Search orders or tracking IDs.
    2. verify_policy: topic (string) - Topics: refund_eligibility, return_verification.
    3. execute_action: cmd ("issue_refund"), params (dict with order_id).
    4. final_response: text (string) - Professional summary to customer.
    
    STRICT SOP:
    1. Search Order ID.
    2. If a tracking_id is found, you MUST search that tracking_id to check Carrier status.
    3. MANDATORY: You MUST use 'verify_policy' for every ticket before taking any action. For Task 3 (Conflict), use 'return_verification'.
    4. Provide a CONCISE and professional FinalResponse when done.
    
    Example: {"action_type": "search_db", "query": "4829"}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    
    # Contextual tag for clarity
    tag = ""
    if reward > 0.5: tag = " [BONUS]"
    elif reward < 0: tag = " [PENALTY]"
    
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f}{tag} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    status = "SUCCESS" if success else "FAILED (SOP Violation)"
    print(f"[END] status={status} steps={steps} final_grade={score:.1f} rewards={rewards_str}", flush=True)


# Consolidated into get_agent_action

async def get_agent_action(obs: OmniSupportObservation, history: List[dict]) -> str:
    """Query the LLM and extract a structured action."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # history contains pairs of {"action": ..., "tool_result": ...}
    for entry in history:
        messages.append({"role": "assistant", "content": entry.get("action", "")})
        messages.append({"role": "user", "content": f"Tool Result: {entry.get('tool_result', '')}"})
    
    # Add a progress nudge
    if len(history) > 0:
        messages.append({"role": "user", "content": f"You are in the middle of a ticket. Analyze the results above and proceed with the SOP."})

    # Add current observation (SLIMMED for local LLM performance)
    slim_obs = {
        "ticket_id": obs.ticket_id,
        "internal_notes": obs.internal_notes,
        "last_tool_output": obs.last_tool_output
    }
    messages.append({"role": "user", "content": f"Current Observation: {json.dumps(slim_obs)}"})

    # ── Retry Loop for robustness ──
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=1024, # Increased for final response room
                stream=False,
            )
            content = (response.choices[0].message.content or "").strip()
            
            if not content:
                # If model is being shy, nudge it
                messages.append({"role": "user", "content": "CONTINUE. Provide your next action in JSON format based on the tools above."})
                continue

            # ── Extraction & Repair Logic ──
            cleaned_content = content
            
            # Simple Auto-Repair for truncated JSON
            if cleaned_content.count('"') % 2 != 0:
                cleaned_content += '"'
            if cleaned_content.count('{') > cleaned_content.count('}'):
                cleaned_content += '}'
            
            if "```" in cleaned_content:
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
                
                # ── Normalization ──
                if isinstance(action, dict):
                    if "action_type" in action:
                        action["action_type"] = action["action_type"].lower()
                    if action.get("action_type") == "search_db" and isinstance(action.get("query"), dict):
                        q = action["query"]
                        action["query"] = str(next(iter(q.values()))) if q else ""
                
                return json.dumps(action)
            except json.JSONDecodeError as je:
                if attempt < max_retries - 1:
                    messages.append({"role": "user", "content": "Your response was empty or malformed. Reply with a valid JSON tool call."})
                    continue
                # DIAGNOSTIC DUMP
                print(f"\n[DIAGNOSTIC DUMP] Complete message history for failed turn:\n{json.dumps(messages, indent=2)}", file=sys.stderr)
                print(f"\n[DEBUG] Raw LLM Output:\n{content}", file=sys.stderr)
                raise je

        except Exception as e:
            if attempt < max_retries - 1:
                last_error = e
                continue
            default_action = {"action_type": "final_response", "text": f"Unable to process at this time. Error: {str(e or last_error)}"}
            return json.dumps(default_action)
    
    # FINAL DIAGNOSTIC DUMP before giving up
    print(f"\n[DIAGNOSTIC DUMP] Failure after {max_retries} attempts. Messages:\n{json.dumps(messages, indent=2)}", file=sys.stderr)
    return json.dumps({"action_type": "final_response", "text": "Model failed to respond after multiple attempts."})


async def check_connectivity():
    """Verify that both the LLM and Env server are reachable."""
    print("Checking connectivity...", file=sys.stderr)
    
    # 1. Check Env
    try:
        from client import OmniSupportEnv
        test_env = OmniSupportEnv(base_url=ENV_URL)
        await test_env.reset(task_id=TASK_NAME)
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
        result = await env.reset(task_id=TASK_NAME)
        
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
            
            # ── Update History (Only essential results to save tokens) ──
            history.append({
                "action": action_json_str,
                "tool_result": json.dumps(obs.get("last_tool_output", {})) if isinstance(obs, dict) else str(obs)
            })

            last_obs = obs
            last_error = error

            # Clean action logs for 1-liner STDOUT constraint
            clean_action_str = action_json_str.replace("\n", "").replace("\r", "")
            log_step(step=step, action=clean_action_str, reward=reward, done=done, error=error)

            if done:
                # Use the official grader score from the server state
                score = obs.get("grader_score", 0.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        # final score is the grader score from the last observation
        if score == 0.0 and last_obs:
            score = last_obs.get("grader_score", 0.0)
            success = score >= SUCCESS_SCORE_THRESHOLD
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())