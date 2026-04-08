"""
Dense Reward Function for OmniSupport-Sim.
Bounded between [-2.0, 1.0] per step.

Components:
  R_prog: +0.1 per novel, relevant KV pair extracted from tools
  R_pen:  -0.5 per SOP violation (must VerifyPolicy before ExecuteAction)
  R_step: -0.02 per step (incentivizes efficiency)
  R_dest: -2.0 for destructive actions (e.g. deleting records - simulation limit)
  R_term: +1.0 if programmatic grader validates final state
"""


class RewardCalculator:
    """Tracks and computes rewards across an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all tracking state for a new episode."""
        self.seen_kv_pairs: set[str] = set()
        self.policy_verified: bool = False
        self.carrier_queried: bool = False
        self.sop_violations: int = 0
        self.total_reward: float = 0.0

    def compute_step_reward(self, action: dict, tool_output: dict | None) -> float:
        """Compute reward for a single step.

        Returns:
            Step reward clipped to [-2.0, 1.0]
        """
        reward = -0.02  # ── R_step: Step penalty ──
        action_type = action.get("action_type")

        # ── R_prog: +0.1 per novel KV pair ──
        if tool_output and isinstance(tool_output, dict):
            # Exclude error outputs from progress reward
            if "error" not in tool_output:
                for key, value in self._flatten_dict(tool_output).items():
                    kv_hash = f"{key}:{value}"
                    if kv_hash not in self.seen_kv_pairs:
                        self.seen_kv_pairs.add(kv_hash)
                        reward += 0.1

        # ── Track tool calls for SOP ──
        if action_type == "verify_policy":
            self.policy_verified = True
        
        # ── R_pen: -0.5 for SOP violations ──
        if action_type == "execute_action":
            if not self.policy_verified:
                reward -= 0.5
                self.sop_violations += 1
            
            # ── R_dest: -2.0 for destructive action simulation ──
            cmd = action.get("cmd", "")
            if cmd in ["delete_user", "clear_database", "wipe_records"]:
                reward -= 2.0

        # Clip to bounds
        reward = max(-2.0, min(1.0, reward))
        self.total_reward += reward
        return reward

    def compute_destructive_penalty(self) -> float:
        """Apply a heavy penalty for destructive actions."""
        penalty = -2.0
        self.total_reward += penalty
        return penalty

    def compute_terminal_reward(self, grader_score: float) -> float:
        """Compute terminal reward based on grader output."""
        if grader_score == 0.0:
            # HALLUCINATION / CRITICAL FAILURE: Wipe out all partial progress rewards
            penalty = -self.total_reward
            self.total_reward = 0.0
            return penalty
            
        r_term = grader_score # 1.0 for full success, partial for Task 3
        self.total_reward += r_term
        return r_term

    def _flatten_dict(self, d: dict, parent_key: str = "") -> dict:
        """Flatten a nested dict for KV pair tracking."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = str(item)
            else:
                items[new_key] = str(v)
        return items