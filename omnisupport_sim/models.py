"""
OmniSupport-Sim Pydantic Models
Defines Action, Observation, and State types for the OpenEnv spec.
"""
from __future__ import annotations
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
import uuid


# ── Observation Model ──────────────────────────────────────────────────────
class OmniSupportObservation(BaseModel):
    """What the agent sees after each step."""
    ticket_id: str = Field(description="Current support ticket ID")
    customer_history: dict = Field(description="Customer profile and order history as JSON")
    internal_notes: str = Field(default="", description="Internal notes from previous tool calls")
    last_tool_output: Optional[dict] = Field(default=None, description="Output from the last tool call")


# ── Action Models (Union of 4 tool types) ──────────────────────────────────
class SearchDB(BaseModel):
    """Search the order/customer database."""
    action_type: Literal["search_db"] = "search_db"
    query: str = Field(description="Search query for order history")

class VerifyPolicy(BaseModel):
    """Fetch specific refund/escalation policy rules."""
    action_type: Literal["verify_policy"] = "verify_policy"
    topic: str = Field(description="Policy topic to look up, e.g. 'refund_eligibility'")

class ExecuteAction(BaseModel):
    """Execute a CRM action like issuing a refund or changing shipping."""
    action_type: Literal["execute_action"] = "execute_action"
    cmd: str = Field(description="Command to execute, e.g. 'issue_refund', 'change_shipping'")
    params: dict = Field(default_factory=dict, description="Parameters for the command")

class FinalResponse(BaseModel):
    """Close the ticket with a customer-facing response."""
    action_type: Literal["final_response"] = "final_response"
    text: str = Field(description="Final response text sent to the customer")

# Discriminated union for Action models
OmniSupportAction = Union[
    SearchDB,
    VerifyPolicy,
    ExecuteAction,
    FinalResponse
]
# Note: In Pydantic V2, use Annotated[Union[...], Field(discriminator='action_type')] for explicit discrimination.


# ── State Model (for grading and debugging) ────────────────────────────────
class OmniSupportState(BaseModel):
    """Full environment state snapshot for grading."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    current_task_id: str = ""
    db_snapshot: dict = Field(default_factory=dict, description="Full database state")
    policy_calls_made: list[str] = Field(default_factory=list)
    actions_taken: list[dict] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list, description="Ordered list of tool types called")
    reward_accumulated: float = 0.0
    done: bool = False