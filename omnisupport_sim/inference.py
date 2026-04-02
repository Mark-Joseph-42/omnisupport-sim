"""
Baseline inference script for OmniSupport-Sim.
Uses OpenAI-compatible API (LM Studio locally, cloud at runtime).
Mandatory [START]/[STEP]/[END] logging protocol.
"""
import os
import json
import time
import signal
import requests
from openai import OpenAI

# ── Configuration ──
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-4b-python-coder")
ENV_URL = os.getenv("ENV_URL", "https://markjoseph2003-metahacky.hf.space")
TIMEOUT_MINUTES = 19  # Must complete under 20 min

client = OpenAI(base_url=API_BASE_URL, api_key=os.getenv("LM_API_KEY", "lm-studio"))

SYSTEM_PROMPT = """You are a Tier-2 Support Specialist AI agent. You solve customer support tickets by using tools.

Available tools:
1. SearchDB(query): Search the order/customer database. Use this to find order details.
2. VerifyPolicy(topic): Check company policy rules. Topics: refund_eligibility, escalation_protocol, return_verification, shipping_change, fraud_investigation
3. ExecuteAction(cmd, params): Execute an action. Commands: issue_refund, change_shipping. Params must include order_id.
4. FinalResponse(text): Close the ticket with a response to the customer.

CRITICAL RULES (SOP):
- ALWAYS call VerifyPolicy BEFORE issuing any refund
- For returns: ALWAYS verify carrier delivery status via SearchDB with tracking_id BEFORE refunding
- Never hallucinate information — only use data from tool outputs
- Respond with EXACTLY ONE tool call per turn as JSON

Respond with a JSON object like:
{"action_type": "search_db", "query": "cust_882"}
or {"action_type": "verify_policy", "topic": "refund_eligibility"}
or {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": 4829}}
or {"action_type": "final_response", "text": "Your refund has been processed."}
"""


def llm_decide(observation: dict) -> dict:
    """Ask the LLM to decide the next action based on the observation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current ticket observation:\n{json.dumps(observation, indent=2)}\n\nWhat is your next action? Respond with JSON only."}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        action = json.loads(content)
        return action
    except Exception as e:
        # Fallback: close ticket if LLM fails
        return {"action_type": "final_response", "text": f"Unable to process at this time. Error: {str(e)}"}


def env_reset(task_id: str) -> dict:
    """Call environment reset endpoint."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """Call environment step endpoint."""
    resp = requests.post(f"{ENV_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    """Get environment state."""
    resp = requests.get(f"{ENV_URL}/state")
    resp.raise_for_status()
    return resp.json()


def run_task(task_id: str) -> float:
    """Run a single task with [START]/[STEP]/[END] logging."""
    print(f"[START] task_id={task_id}")

    result = env_reset(task_id)
    max_steps = 10  # Safety limit

    for step_num in range(max_steps):
        observation = result.get("observation", {})
        action = llm_decide(observation)

        log_entry = {
            "step": step_num + 1,
            "action": action,
            "observation_summary": {
                "ticket_id": observation.get("ticket_id"),
                "has_tool_output": observation.get("last_tool_output") is not None,
            }
        }
        print(f"[STEP] {json.dumps(log_entry)}")

        result = env_step(action)

        if result.get("done", False):
            break

    # Get final state and score
    final_state = env_state()
    score = result.get("info", {}).get("total_reward", 0.0)

    print(f"[END] task_id={task_id} score={score}")
    return score


def main():
    """Run all 3 tasks within timeout."""
    start_time = time.time()
    task_ids = ["order_check", "refund_logic", "fraud_mitigation"]
    scores = {}

    for task_id in task_ids:
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_MINUTES * 60:
            print(f"[ERROR] Timeout exceeded ({TIMEOUT_MINUTES} min). Skipping remaining tasks.")
            break

        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}")
            scores[task_id] = 0.0

    # Summary
    total_time = time.time() - start_time
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETE")
    print(f"Tasks: {json.dumps(scores, indent=2)}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()