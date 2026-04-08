"""
OmniSupport-Sim Environment — Core implementation.
Implements the OpenEnv spec: reset(), step(), state()
"""
import uuid
from typing import Optional

from omnisupport_sim.models import (
    OmniSupportObservation,
    OmniSupportAction,
    OmniSupportState,
    SearchDB,
    VerifyPolicy,
    ExecuteAction,
    FinalResponse,
)
from server.mock_db import MockDB
from server.policy_kb import lookup_policy
from server.carrier_api import query_carrier
from server.reward import RewardCalculator
from server.graders import grade


# ── Task Scenario Definitions ──────────────────────────────────────────────
TASK_SCENARIOS = {
    "order_check": {
        "ticket_id": "TK-1001",
        "ticket_text": "Hi, I have multiple orders and I need to know the current status of my most recent one. My customer ID is cust_882.",
        "customer_id": "cust_882",
        "description": "Status Resolution — retrieve correct order status from multiple active orders."
    },
    "refund_logic": {
        "ticket_id": "TK-1002",
        "ticket_text": "I received my Wireless Headphones (order #4829) and they're damaged. I want a full refund immediately. My customer ID is cust_882.",
        "customer_id": "cust_882",
        "description": "SOP Compliance — verify refund policy before processing. Item: Wireless Headphones, $89.99."
    },
    "fraud_mitigation": {
        "ticket_id": "TK-1003",
        "ticket_text": "I already returned my Wireless Headphones (order #4829) days ago but I still haven't received my refund. The tracking shows it was delivered back to you. Please process my refund now. Customer ID: cust_882.",
        "customer_id": "cust_882",
        "description": "Conflict Reconciliation — customer claims return, DB shows 'Pending Return'. Must verify via CarrierAPI before refunding."
    },
}


class OmniSupportEnvironment:
    """OpenEnv-compliant environment for Tier-2 Support Simulation."""

    def __init__(self):
        self.db = MockDB()
        self.reward_calc = RewardCalculator()
        self._state: Optional[OmniSupportState] = None
        self._current_scenario: Optional[dict] = None

    def reset(self, task_id: str = "order_check") -> dict:
        """Initialize a new episode.

        Args:
            task_id: One of 'order_check', 'refund_logic', 'fraud_mitigation'

        Returns:
            StepResult-like dict with observation, reward, done
        """
        self.db.reset()
        self.reward_calc.reset()

        scenario = TASK_SCENARIOS.get(task_id)
        if scenario is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        self._current_scenario = scenario
        customer_history = self.db.get_customer_history(scenario["customer_id"])

        self._state = OmniSupportState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
            db_snapshot=self.db.get_snapshot(),
            policy_calls_made=[],
            actions_taken=[],
            tools_called=[],
            reward_accumulated=0.0,
            done=False,
        )

        observation = OmniSupportObservation(
            ticket_id=scenario["ticket_id"],
            customer_history=customer_history,
            internal_notes=f"New ticket: {scenario['ticket_text']}",
            last_tool_output=None,
        )

        return {
            "observation": observation.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {"task_id": task_id, "description": scenario["description"]},
        }

    def step(self, action: dict) -> dict:
        """Process an agent action.

        Args:
            action: Dict with action_type and relevant fields

        Returns:
            StepResult-like dict with observation, reward, done
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        if self._state.done:
            # Idempotency: If final_response is called on a done episode, just return the final state again
            if action.get("action_type") == "final_response":
                observation = OmniSupportObservation(
                    ticket_id=self._current_scenario["ticket_id"],
                    customer_history=self.db.get_customer_history(self._current_scenario["customer_id"]),
                    internal_notes="Ticket already resolved.",
                    last_tool_output={"response_sent": True, "already_done": True},
                )
                return {
                    "observation": observation.model_dump(),
                    "reward": 0.0,
                    "done": True,
                    "info": {"status": "already_done"},
                }
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        action_type = action.get("action_type")
        tool_output = None
        internal_notes = ""
        done = False

        # ── Route action ──
        if action_type == "search_db":
            query = action.get("query", "")
            tool_output = self.db.search_orders(query)
            # Check if this is actually a carrier query
            if any(tid in query.upper() for tid in ["TRK-"]):
                carrier_result = query_carrier(query.strip())
                tool_output = carrier_result
                self.reward_calc.carrier_queried = True
            if isinstance(tool_output, list):
                tool_output = {"results": tool_output, "count": len(tool_output)}
            internal_notes = f"SearchDB('{query}') returned {len(tool_output) if isinstance(tool_output, dict) else 0} fields"

        elif action_type == "verify_policy":
            topic = action.get("topic", "")
            tool_output = lookup_policy(topic)
            self._state.policy_calls_made.append(topic)
            self.reward_calc.policy_verified = True
            internal_notes = f"VerifyPolicy('{topic}') — policy retrieved"

        elif action_type == "execute_action":
            cmd = action.get("cmd", "")
            params = action.get("params", {})
            if cmd == "issue_refund":
                order_id = params.get("order_id")
                if order_id:
                    tool_output = self.db.update_refund_status(int(order_id), "SUCCESS")
                else:
                    tool_output = {"error": "Missing order_id parameter"}
                internal_notes = f"ExecuteAction('{cmd}') — refund processed"
            elif cmd == "change_shipping":
                tool_output = {"success": True, "message": "Shipping address updated"}
                internal_notes = f"ExecuteAction('{cmd}') — shipping updated"
            else:
                tool_output = {"error": f"Unknown command: {cmd}"}
                internal_notes = f"ExecuteAction('{cmd}') — unknown command"

        elif action_type == "final_response":
            text = action.get("text", "")
            tool_output = {"response_sent": True, "text": text}
            internal_notes = f"FinalResponse sent to customer"
            done = True

        else:
            tool_output = {"error": f"Unknown action_type: {action_type}"}
            internal_notes = f"Unknown action_type: {action_type}"

        # ── Record action ──
        self._state.actions_taken.append(action)
        self._state.tools_called.append(action_type)
        self._state.step_count += 1

        # ── Compute reward ──
        step_reward = self.reward_calc.compute_step_reward(
            action,
            tool_output if isinstance(tool_output, dict) else {"results": tool_output}
        )

        # ── Terminal reward if done ──
        if done:
            self._state.db_snapshot = self.db.get_snapshot()
            grader_score = grade(self._state.model_dump(), self._state.current_task_id)
            terminal_reward = self.reward_calc.compute_terminal_reward(grader_score)
            step_reward += terminal_reward

        self._state.reward_accumulated += step_reward
        self._state.done = done
        self._state.db_snapshot = self.db.get_snapshot()

        # Build observation
        customer_history = self.db.get_customer_history(self._current_scenario["customer_id"])
        observation = OmniSupportObservation(
            ticket_id=self._current_scenario["ticket_id"],
            customer_history=customer_history,
            internal_notes=internal_notes,
            last_tool_output=tool_output if isinstance(tool_output, dict) else {"results": tool_output},
        )

        return {
            "observation": observation.model_dump(),
            "reward": step_reward,
            "done": done,
            "info": {
                "step_count": self._state.step_count,
                "total_reward": self._state.reward_accumulated,
                "sop_violations": self.reward_calc.sop_violations,
            },
        }

    def state(self) -> dict:
        """Return full state snapshot for grading and debugging."""
        if self._state is None:
            return {"error": "No active episode. Call reset() first."}
        self._state.db_snapshot = self.db.get_snapshot()
        return self._state.model_dump()