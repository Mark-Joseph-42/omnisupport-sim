"""
Deterministic Python graders for all 3 tasks.
Each returns a score S ∈ [0.0, 1.0].
NO LLM-grading-LLM — purely programmatic.
"""
from typing import Any


def grade_task_1(state: dict) -> float:
    """Task 1: Status Resolution (Easy)
    Binary 1.0 if FinalResponse matches the status of the MOST RECENT order.
    """
    actions = state.get("actions_taken", [])
    db_snapshot = state.get("db_snapshot", {})
    orders = db_snapshot.get("orders", [])

    # Task 1 scenario is about Alex Rivera (cust_882)
    target_customer_id = "cust_882"
    relevant_orders = [o for o in orders if o.get("customer_id") == target_customer_id]
    
    if not relevant_orders:
        return 0.0

    most_recent = max(relevant_orders, key=lambda o: o.get("purchase_date", ""))
    expected_status = most_recent["status"]

    # Find the FinalResponse action
    for action in reversed(actions):
        if action.get("action_type") == "final_response":
            response_text = action.get("text", "").lower()
            if expected_status.lower() in response_text:
                return 1.0
            # Also check if the order_id and status are mentioned
            if str(most_recent["order_id"]) in response_text and expected_status.lower() in response_text:
                return 1.0
            return 0.0

    return 0.0  # No FinalResponse found


def grade_task_2(state: dict) -> float:
    """Task 2: SOP Compliance - Refund (Medium)
    0.4 for calling VerifyPolicy('refund_eligibility')
    0.6 for correct Boolean refund decision
    0.0 override for hallucinated refund on ineligible item
    """
    actions = state.get("actions_taken", [])
    db_snapshot = state.get("db_snapshot", {})
    policy_calls = state.get("policy_calls_made", [])
    tools_called = state.get("tools_called", [])

    score = 0.0

    # ── 0.4: Did agent call VerifyPolicy for refund? ──
    verified = any(
        "refund" in call.lower()
        for call in policy_calls
    )
    if verified:
        score += 0.4

    # ── Determine ground truth: is the item eligible? ──
    # ── Check for illegal refunds (Hallucination) ──
    # Task 2 scenario: order 4829 (Wireless Headphones, $89.99, purchased 2026-03-25)
    target_order_id = 4829
    for order in db_snapshot.get("orders", []):
        # We only care about the target order for this scenario
        if order["order_id"] == target_order_id:
            target_order = order
        elif order.get("refund_status") == "SUCCESS":
            # Agent refunded something OTHER than what was requested
            return 0.0

    if target_order is None:
        return score

    # Item IS eligible (< 14 days, < $500, LOYALTY-GOLD)
    item_eligible = True

    # ── Check what the agent decided ──
    refund_issued = target_order.get("refund_status") == "SUCCESS"

    if item_eligible and refund_issued:
        score += 0.6  # Correct: issued refund for eligible item
    elif not item_eligible and not refund_issued:
        score += 0.6  # Correct: denied refund for ineligible item
    elif not item_eligible and refund_issued:
        # HALLUCINATION: issued refund for ineligible item
        return 0.0  # Override everything
    else:
        # Didn't issue refund for eligible item (missed opportunity but not a hallucination)
        score += 0.0

    return score


def grade_task_3(state: dict) -> float:
    """Task 3: Conflict Reconciliation (Hard)
    State-based, non-binary scoring:
      0.2: Correct tracking_id identified in logs
      0.4: CarrierAPI called with correct parameters
      0.4: DB.refund_status updated to SUCCESS
    
    CRITICAL: 0.0 override if refund without BOTH VerifyPolicy AND CarrierAPI check.
    """
    actions = state.get("actions_taken", [])
    db_snapshot = state.get("db_snapshot", {})
    policy_calls = state.get("policy_calls_made", [])

    # Task 3 scenario: Order 4829, tracking_id TRK-9928-XZ
    expected_tracking_id = "TRK-9928-XZ"
    expected_order_id = 4829
    score = 0.0

    # ── Check tool calls ──
    carrier_called_correctly = False
    tracking_found = False
    policy_verified = any("return" in call.lower() or "verification" in call.lower() or "refund" in call.lower() for call in policy_calls)

    for action in actions:
        action_type = action.get("action_type", "")
        action_str = str(action)
        
        if expected_tracking_id in action_str:
            tracking_found = True

        if action_type == "search_db":
            query = action.get("query", "")
            if expected_tracking_id in query:
                carrier_called_correctly = True

    # ── Check for state mutation (Ground Truth) ──
    refund_success = False
    for order in db_snapshot.get("orders", []):
        if order["order_id"] == expected_order_id:
            if order.get("refund_status") == "SUCCESS":
                refund_success = True
        elif order.get("refund_status") == "SUCCESS":
            return 0.0 # Wrong order refunded

    # ── CRITICAL: 0.0 override for Hallucination / SOP Violation ──
    if refund_success:
        if not carrier_called_correctly or not policy_verified:
            return 0.0 # Issuing refund without verification is a failure

    # ── Score components ──
    if tracking_found:
        score += 0.2
    if carrier_called_correctly:
        score += 0.4
    if refund_success:
        score += 0.4

    return score


# ── Router ──
GRADERS = {
    "order_check": grade_task_1,
    "refund_logic": grade_task_2,
    "fraud_mitigation": grade_task_3,
}

def grade(state: dict, task_id: str) -> float:
    """Route to the correct grader based on task_id."""
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(GRADERS.keys())}")
    return grader(state)