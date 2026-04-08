import sys
import os
import json

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from server.omnisupport_environment import OmniSupportEnvironment

def run_test_case(name, task_id, actions):
    print(f"\n--- Testing Persona: {name} ({task_id}) ---")
    env = OmniSupportEnvironment()
    
    # 1. Reset
    result = env.reset(task_id=task_id)
    print(f"Reset: {result['info']['description']}")
    
    total_reward = 0.0
    
    # 2. Steps
    for i, action in enumerate(actions):
        print(f"Step {i+1}: Action -> {action['action_type']}")
        try:
            result = env.step(action)
            reward = result['reward']
            total_reward += reward
            print(f"       Reward: {reward:+.2f} | Done: {result['done']}")
            if result['done']:
                break
        except Exception as e:
            print(f"       ERROR: {str(e)}")
            break
            
    # 3. Final State
    final_state = env.state()
    final_score = result.get('info', {}).get('total_reward', 0.0) # Corrected field based on code
    print(f"Final Outcome: Reward={total_reward:.2f} | Grader Score={final_state.get('reward_accumulated', 0.0)}")
    return final_state

if __name__ == "__main__":
    # PERSONA 1: The Perfect Agent (Task 1: Status Resolution)
    # Goal: Get status of recent order (cust_882)
    run_test_case("Perfect Agent", "order_check", [
        {"action_type": "search_db", "query": "cust_882"},
        {"action_type": "final_response", "text": "Your most recent order for Wireless Headphones is Pending Return."}
    ])

    # PERSONA 2: The Perfect Agent (Task 2: SOP Compliance - Refund)
    # Goal: Verify policy AND check DB before refunding order 4829
    run_test_case("Perfect Agent", "refund_logic", [
        {"action_type": "verify_policy", "topic": "refund_eligibility"},
        {"action_type": "search_db", "query": "4829"},
        {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": 4829}},
        {"action_type": "final_response", "text": "Refund processed successfully."}
    ])

    # PERSONA 3: The SOP Violator (Task 2)
    # Goal: Refund WITHOUT verifying policy
    run_test_case("SOP Violator", "refund_logic", [
        {"action_type": "search_db", "query": "4829"},
        {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": 4829}},
        {"action_type": "final_response", "text": "Refund processed."}
    ])

    # PERSONA 4: The Hallucinating Agent (Task 2)
    # Goal: Refund an old ineligible order (3901)
    run_test_case("Hallucinating Agent", "refund_logic", [
        {"action_type": "verify_policy", "topic": "refund_eligibility"},
        {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": 3901}},
        {"action_type": "final_response", "text": "Refunded your old order."}
    ])

    # PERSONA 5: The Perfect Agent (Task 3: Conflict Reconciliation)
    # Goal: Check Carrier API for tracking TRK-9928-XZ before refunding
    run_test_case("Perfect Agent", "fraud_mitigation", [
        {"action_type": "verify_policy", "topic": "return_verification"},
        {"action_type": "search_db", "query": "TRK-9928-XZ"}, # This triggers carrier_api in the code
        {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": 4829}},
        {"action_type": "final_response", "text": "Carrier confirmed delivery, refunding now."}
    ])

    # PERSONA 6: The "Blind" Agent (Task 3)
    # Goal: Refund WITHOUT checking carrier
    run_test_case("Blind Agent", "fraud_mitigation", [
        {"action_type": "execute_action", "cmd": "issue_refund", "params": {"order_id": 4829}},
        {"action_type": "final_response", "text": "Refunding without verification."}
    ])

    # EDGE CASE: Calling step after done
    print("\n--- Testing Edge Case: Step after Done ---")
    env = OmniSupportEnvironment()
    env.reset("order_check")
    env.step({"action_type": "final_response", "text": "Done."})
    try:
        env.step({"action_type": "search_db", "query": "test"})
    except Exception as e:
        print(f"Caught Expected Error: {e}")
