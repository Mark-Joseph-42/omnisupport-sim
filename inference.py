"""
Baseline inference script for OmniSupport-Sim.
Compliant with OpenEnv stdout logging protocol (MANDATORY FORMAT).

STDOUT FORMAT (strictly enforced):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line after env.close(), always emitted (even on exception).
  - reward and score are formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw error string, or null if none.
  - All fields on a single line with no newlines within a line.
  - All scores/rewards are strictly in (0, 1) exclusive range.
"""
import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Dynamic Path Injection ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from client import OmniSupportEnv

try:
    from omnisupport_sim.models import OmniSupportAction, OmniSupportObservation
except ImportError:
    try:
        from models import OmniSupportAction, OmniSupportObservation
    except ImportError:
        OmniSupportAction = None
        OmniSupportObservation = None

# Build TypeAdapter once at import time (OmniSupportAction is a Union type)
try:
    from pydantic import TypeAdapter as _TA
    _action_adapter = _TA(OmniSupportAction) if OmniSupportAction is not None else None
except Exception:
    _action_adapter = None

# ── Configuration (Mandatory env vars — evaluator always overrides these) ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "dummy-key"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("url_arg", nargs="?", default=None)
parser.add_argument("--task", default=None)  # kept for compatibility but overridden by TASK_IDS loop
args, unknown = parser.parse_known_args()

ENV_URL = args.url_arg or os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = os.getenv("OMNISUPPORT_BENCHMARK", "omnisupport_sim")

# ── ALL 5 Tasks — required by evaluator ──
TASK_IDS = ["order_check", "refund_logic", "fraud_mitigation", "fraud_prevention", "escalation_required"]

TIMEOUT_MINUTES = 19  # Hard limit: evaluator kills after 20 min
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Task-specific system prompts for better SOP compliance ──────────────────

_PROMPT_BASE = textwrap.dedent("""
    You are a Tier-2 Support AI. You solve tickets using JSON tool calls.
    Reply with ONE JSON object only — no explanation, no markdown, no extra text.

    AVAILABLE TOOLS (use exactly these action_type values):
    - {"action_type": "search_db", "query": "<order_id or customer_id or tracking_id>"}
    - {"action_type": "verify_policy", "topic": "<topic>"}
    - {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": "<id>"}}
    - {"action_type": "final_response", "text": "<single paragraph, no newlines>"}
""").strip()

_PROMPT_ORDER_CHECK = _PROMPT_BASE + textwrap.dedent("""

    YOUR TASK: STATUS INQUIRY ONLY — do NOT issue any refunds.
    The customer wants to know the status of their MOST RECENT order.

    MANDATORY STEPS (in order):
    1. search_db with the customer ID (e.g. "cust_882") to get all their orders.
    2. Identify the order with the LATEST purchase_date.
    3. final_response stating the status of that most recent order clearly.

    EXAMPLE OUTPUT:
    Step 1: {"action_type": "search_db", "query": "cust_882"}
    Step 2: {"action_type": "final_response", "text": "Your most recent order is [item] (order #[id]), currently showing status: [status]."}
""").strip()

_PROMPT_REFUND_LOGIC = _PROMPT_BASE + textwrap.dedent("""

    YOUR TASK: REFUND REQUEST — customer claims item is damaged.

    MANDATORY STEPS (in order — do not skip any):
    1. search_db with the order ID from the ticket.
    2. verify_policy with topic "refund_eligibility" (REQUIRED before any refund).
    3. If eligible (value < $500, within time window, no FRAUD FLAG): execute_action issue_refund.
    4. final_response with outcome.

    RULES:
    - LOYALTY-GOLD members: 30-day window. Standard: 14-day window.
    - Value must be under $500 for auto-refund.
    - FRAUD FLAG in notes = DENY refund immediately.
    - You MUST call verify_policy with "refund_eligibility" — not any other topic.

    EXAMPLE:
    {"action_type": "verify_policy", "topic": "refund_eligibility"}
""").strip()

_PROMPT_FRAUD_MITIGATION = _PROMPT_BASE + textwrap.dedent("""

    YOUR TASK: CONFLICT RECONCILIATION — customer says they returned an item but no refund received.

    MANDATORY STEPS (in order — do not skip any):
    1. search_db with the order ID to get the tracking_id.
    2. search_db with the tracking_id (e.g. "TRK-XXXX-XX") to check carrier delivery status.
    3. verify_policy with topic "return_verification" (REQUIRED before any refund).
    4. Only if carrier status is "Delivered": execute_action issue_refund.
    5. final_response with outcome.

    CRITICAL: You MUST check the carrier (step 2) AND verify policy (step 3) BEFORE issuing any refund.
    Skipping either step results in SCORE 0.0.

    EXAMPLE:
    {"action_type": "search_db", "query": "TRK-9928-XZ"}
    {"action_type": "verify_policy", "topic": "return_verification"}
""").strip()

_PROMPT_FRAUD_PREVENTION = _PROMPT_BASE + textwrap.dedent("""

    YOUR TASK: FRAUD PREVENTION — detect a FRAUD_FLAG and DENY the refund request.

    MANDATORY STEPS (in order):
    1. search_db with the order ID or customer ID from the ticket.
    2. READ the order notes carefully. Look for 'FRAUD_FLAG'.
    3. verify_policy with topic "refund_eligibility" (REQUIRED even for denials).
    4. final_response: DENY the refund professionally citing the flag.

    CRITICAL RULES:
    - If the order notes contain 'FRAUD_FLAG', you MUST NOT issue a refund under ANY circumstances.
    - Do NOT call execute_action issue_refund if FRAUD_FLAG is present.
    - Your final_response must include language like "unable to process", "flagged", or "denied".

    EXAMPLE DENIAL RESPONSE:
    "I've reviewed your account and unfortunately we're unable to process this refund at this time
    due to a security flag on your account. Please contact our fraud review team for assistance."
""").strip()

_PROMPT_ESCALATION = _PROMPT_BASE + textwrap.dedent("""

    YOUR TASK: ESCALATION REQUIRED — order value exceeds $500 auto-refund limit.

    MANDATORY STEPS (in order):
    1. search_db with the order ID or customer ID from the ticket.
    2. Check the item value. If it is over $500, you CANNOT auto-refund.
    3. verify_policy with topic "refund_eligibility" (REQUIRED).
    4. final_response: Tell the customer this requires manual escalation. DO NOT issue a refund.

    CRITICAL RULES:
    - Items valued over $500 CANNOT be auto-refunded. Period.
    - Do NOT call execute_action issue_refund for high-value items.
    - Your response MUST mention: escalation, supervisor, specialist, or manual review.

    EXAMPLE RESPONSE:
    "Your order (#XXXX) for [item] at $[value] exceeds our automated refund threshold of $500.
    I am escalating this to our specialist team who will contact you within 24 hours."
""").strip()

TASK_PROMPTS = {
    "order_check":         _PROMPT_ORDER_CHECK,
    "refund_logic":        _PROMPT_REFUND_LOGIC,
    "fraud_mitigation":    _PROMPT_FRAUD_MITIGATION,
    "fraud_prevention":    _PROMPT_FRAUD_PREVENTION,
    "escalation_required": _PROMPT_ESCALATION,
}

# Fallback for unknown tasks
def get_system_prompt(task_name: str) -> str:
    return TASK_PROMPTS.get(task_name, _PROMPT_REFUND_LOGIC)


# ── Score Utilities ──────────────────────────────────────────────────────────

def clamp_score(s: float) -> float:
    """Force score into strictly (0, 1) exclusive range as required by evaluator."""
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    return round(s, 2)


def clamp_reward(r: float) -> float:
    """Force individual step reward into strictly (0, 1) exclusive range."""
    return clamp_score(r)


# ── Mandatory STDOUT Logging ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    reward_clamped = clamp_reward(reward)
    # Strip all newlines from action to keep it on a single line
    clean_action = action.replace("\n", "").replace("\r", "")
    print(
        f"[STEP] step={step} action={clean_action} reward={reward_clamped:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_clamped = [clamp_reward(r) for r in rewards]
    score_clamped = clamp_score(score)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_clamped)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score_clamped:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Agent Logic ───────────────────────────────────────────────────────────────

async def get_agent_action(obs: OmniSupportObservation, history: List[dict], task_name: str = "refund_logic") -> str:
    """Query the LLM and extract a structured action."""
    messages = [{"role": "system", "content": get_system_prompt(task_name)}]

    # Inject conversation history (action + tool result pairs)
    for entry in history:
        messages.append({"role": "assistant", "content": entry.get("action", "")})
        messages.append({"role": "user", "content": f"Tool Result: {entry.get('tool_result', '')}"})

    # Nudge if mid-episode
    if len(history) > 0:
        messages.append({
            "role": "user",
            "content": "You are in the middle of a ticket. Analyze the results above and proceed with the SOP."
        })

    # Slim observation to conserve tokens for smaller models
    slim_obs = {
        "ticket_id": obs.ticket_id,
        "internal_notes": obs.internal_notes,
        "last_tool_output": obs.last_tool_output,
    }
    messages.append({"role": "user", "content": f"Current Observation: {json.dumps(slim_obs)}"})

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                stream=False,
            )
            content = (response.choices[0].message.content or "").strip()

            if not content:
                messages.append({"role": "user", "content": "CONTINUE. Provide your next action in JSON format."})
                continue

            # ── JSON Extraction & Repair ──
            cleaned = content

            # Strip markdown code fences
            if "```" in cleaned:
                for block in cleaned.split("```"):
                    if "{" in block and "}" in block:
                        cleaned = block
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]
                        break

            cleaned = cleaned.strip()

            # Simple auto-repair for truncated JSON
            if cleaned.count('"') % 2 != 0:
                cleaned += '"'
            if cleaned.count("{") > cleaned.count("}"):
                cleaned += "}"

            try:
                action = json.loads(cleaned)

                # ── Normalization: handle common LLM hallucination patterns ──
                if isinstance(action, dict):
                    # Map short-form keys to full action structure
                    if "final_response" in action and "text" not in action:
                        action = {"action_type": "final_response", "text": action["final_response"]}
                    elif "response" in action and "text" not in action:
                        action = {"action_type": "final_response", "text": action["response"]}

                    # Lowercase action_type
                    if "action_type" in action:
                        action["action_type"] = action["action_type"].lower()
                    else:
                        # Infer action_type from present keys
                        if "text" in action:
                            action["action_type"] = "final_response"
                        elif "query" in action:
                            action["action_type"] = "search_db"
                        elif "topic" in action:
                            action["action_type"] = "verify_policy"
                        elif "cmd" in action:
                            action["action_type"] = "execute_action"

                    # Fix query that's a dict instead of string
                    if action.get("action_type") == "search_db" and isinstance(action.get("query"), dict):
                        q = action["query"]
                        action["query"] = str(next(iter(q.values()))) if q else ""

                return json.dumps(action)

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    messages.append({
                        "role": "user",
                        "content": "Your response was malformed JSON. Reply with a valid JSON tool call only."
                    })
                    continue
                print(f"[DEBUG] JSON parse failed after {max_retries} attempts. Raw: {content[:200]}", file=sys.stderr)
                return json.dumps({"action_type": "final_response", "text": "Unable to parse action after retries."})

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                continue
            print(f"[DEBUG] Model request failed: {e}", file=sys.stderr)
            return json.dumps({"action_type": "final_response", "text": f"Model error: {str(e)}"})

    return json.dumps({"action_type": "final_response", "text": "Model failed to respond after multiple attempts."})


# ── Single Task Runner ────────────────────────────────────────────────────────

async def run_single_task(task_name: str) -> None:
    """Run a complete episode for one task and emit [START]...[STEP]...[END]."""
    env = OmniSupportEnv(base_url=ENV_URL)

    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # Default to 0.01 (valid minimum) if episode fails
    success = False
    last_obs = None
    last_error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_name)

        last_obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else {}

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_json_str = await get_agent_action(result.observation, history, task_name)

            try:
                if _action_adapter is not None:
                    action_obj = _action_adapter.validate_json(action_json_str)
                else:
                    # Last-resort fallback: send raw dict to env
                    action_obj = json.loads(action_json_str)

                result = await env.step(action_obj)

                obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else {}
                reward = result.reward or 0.0
                done = result.done
                step_error = None

            except Exception as e:
                obs = last_obs or {}
                reward = 0.0
                done = False
                step_error = str(e).replace("\n", " ").replace("\r", "")
                print(f"[DEBUG] Step {step} error (task={task_name}): {step_error}", file=sys.stderr)

            rewards.append(reward)
            steps_taken = step

            # Update history (only essential context to save tokens)
            history.append({
                "action": action_json_str,
                "tool_result": json.dumps(obs.get("last_tool_output", {})) if isinstance(obs, dict) else str(obs),
            })

            last_obs = obs
            last_error_msg = step_error

            log_step(step=step, action=action_json_str, reward=reward, done=done, error=step_error)

            if done:
                # Read grader score from final observation
                raw_score = obs.get("grader_score", 0.0) if isinstance(obs, dict) else 0.0
                score = clamp_score(raw_score)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

    except Exception as e:
        print(f"[DEBUG] Episode error for task={task_name}: {e}", file=sys.stderr)
        score = 0.01

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (task={task_name}): {e}", file=sys.stderr)

        # Final score fallback: if episode ended without done=True, check last obs
        if score <= 0.01 and last_obs and isinstance(last_obs, dict):
            raw_score = last_obs.get("grader_score", 0.0)
            if raw_score > 0.0:
                score = clamp_score(raw_score)
                success = score >= SUCCESS_SCORE_THRESHOLD

        # CRITICAL: rewards list must never be empty — empty list produces
        # malformed [END] line (rewards=) that the evaluator cannot parse.
        if not rewards:
            rewards = [0.01]

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main Entry Point ────────────────────────────────────────────────────────

async def _run_all_tasks() -> None:
    """Run all 5 required tasks sequentially."""
    for task_name in TASK_IDS:
        await run_single_task(task_name)


async def main() -> None:
    """Entry point with global 19-minute timeout to prevent evaluator DQ."""
    try:
        await asyncio.wait_for(_run_all_tasks(), timeout=TIMEOUT_MINUTES * 60)
    except asyncio.TimeoutError:
        print(f"[DEBUG] Global timeout reached ({TIMEOUT_MINUTES} min). Emitting fallback END.", file=sys.stderr)
        # Emit a fallback [END] for the current task if timed out mid-run
        print(f"[END] success=false steps=0 score=0.01 rewards=0.01", flush=True)


if __name__ == "__main__":
    asyncio.run(main())