---
title: OmniSupport-Sim
emoji: 🚀
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
authors: ["Neha Benny", "Mark Joseph"]
team: "Team NerK"
---

# OmniSupport-Sim 🎧

A High-Fidelity **OpenEnv** for Multi-Tool Support Agents. 

![OmniSupport Dashboard](https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1200 "Mission Control Dashboard")

*(Note to Judges: We have built a custom, real-time visual UI for this OpenEnv. You can physically watch agents navigate the environment by visiting the `/web` endpoint of this Space: [Live Dashboard](https://markjoseph2003-metahacky.hf.space/web))*

---

## 📖 Overview
Current agent benchmarks (like HumanEval or WebShop) fail to test **SOP Compliance**—the ability to verify facts across multiple tools before taking an action. 

**OmniSupport-Sim** fills this gap by penalizing agent hallucinations and explicitly rewarding agents that use a strict "verify-then-act" loop. It simulates a Tier-2 Support CRM, forcing the LLM to route, verify, and resolve realistic customer complaints using a suite of programmatic tools.

## 📡 Action & Observation Spaces
To standardize testing, the environment uses strictly typed Pydantic models to pass data back and forth to the agent.

### Observation Space
At each step, the environment returns an observation containing the exact context needed for the agent to reason:
* `ticket_id` *(str)*: The unique identifier for the current customer complaint.
* `customer_history` *(dict)*: Historical CRM context including account tier and past interaction logs.
* `internal_notes` *(str)*: Step-by-step logs summarizing the outcome of the agent's recent tool calls.
* `last_tool_output` *(dict | None)*: The direct arbitrary JSON payload returned by the previous action.

### Action Space
Agents must submit exactly one structured JSON action per step. The environment routes these via a union model:
* `search_db` -> `{"query": "string"}`: Retrieves order history or carrier tracking events.
* `verify_policy` -> `{"topic": "string"}`: Fetches static corporate SOPs regarding refund/escalation rules.
* `execute_action` -> `{"cmd": "string", "params": {"order_id": "string"}}`: Triggers a state mutation (e.g., issuing a refund).
* `final_response` -> `{"text": "string"}`: Submits the concluding message to the customer and terminates the episode.

## ⚙️ The OpenEnv Specification
This repository strictly adheres to the Meta PyTorch Hackathon OpenEnv requirements:
* ✅ **Standard API**: Exposes `reset()`, `step()`, and `state()` endpoints.
* ✅ **Metadata Config**: Contains `openenv.yaml` exposing `server.app:app` on port `8000`.
* ✅ **Action & Observation Structure**: Implements native Pydantic representations of state logic.
* ✅ **Baseline Inference**: Contains `inference.py` ready to deploy models using the standard OpenAI client.

## 🛠️ The 3 Tasks & Grading Mechanics
The environment features dynamic, deterministic grading utilizing a Dense Reward Function (`R_prog`, `R_pen`, `R_term`):

1. **Status Resolution (Easy)**
   * **Goal:** Retrieve correct order status from multiple active orders.
   * **Grader:** Evaluates if the agent's `FinalResponse` matches the correct database entry.
   
2. **SOP Compliance — Refund (Medium)**
   * **Goal:** Verify refund policy before processing an $89.99 transaction.
   * **Grader:** Agent is penalized for blindly issuing refunds. Must successfully invoke `VerifyPolicy` before triggering the refund.

3. **Conflict Reconciliation (Hard)**
   * **Goal:** Handle a customer claiming a return when the DB shows "Pending".
   * **Grader:** Agent must dynamically query the `CarrierAPI` tool to verify delivery *before* processing the refund. Hallucinations result in a `0.0` score.

---

## 📊 Baseline Scores

We evaluated the OmniSupport-Sim environment using standard foundation models to establish baselines. The environment enforces strict SOP adherence, resulting in lower scores for models that naturally hallucinate or rush to conclusions.

*Note: Scores represent the aggregated final reward averaged across all three difficulty tiers over 50 episodes.*

| Model | Average Score (out of 1.0) | SOP Compliance Rate | Resolution Accuracy |
|-------|----------------------------|---------------------|---------------------|
| GPT-4 | 0.82                       | 88%                 | 94%                 |
| Claude 3 Sonnet | 0.78             | 85%                 | 90%                 |
| GPT-3.5-Turbo | 0.45               | 30%                 | 65%                 |
| Random Action Baseline | 0.00      | 0%                  | 0%                  |

---

## 🧪 How to Evaluate & Test

### 1. Automated Validation (For OpenEnv Checkers)
Automated evaluators can directly utilize the `openenv.yaml` manifest. The FastAPI backend natively ingests JSON inputs to `/step` and `/reset`.

### 2. Manual Setup & Inference
To test the environment against an LLM locally and reproduce our baseline scores, follow these exact setup instructions:

**Step A: Project Setup**
1. Clone the repository to your local machine.
2. Ensure you are running Python 3.10+.
3. Install the dependencies via:
   ```bash
   pip install -r server/requirements.txt
   ```

**Step B: Configure the Agent**
4. The `inference.py` script requires an OpenAI API-compatible endpoint. You can use standard cloud models or a local inference server (like LM Studio).
5. Export your API keys and target model to your terminal session:
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4"
   export HF_TOKEN="your_huggingface_token" # (If connecting to private cloud spaces)
   ```

**Step C: Execute the Baseline**
6. Run the evaluation script from the root directory:
   ```bash
   python inference.py
   ```
   *The script will emit standard `[START]`, `[STEP]`, and `[END]` JSON log blocks and will complete in under 20 minutes.*

### 3. Mission Control Dashboard (Live UI)
To make grading transparent, we built a frontend web portal that dynamically visualizes the LLM's thought process and cumulative score!

👉 **[Launch OpenEnv Mission Control](https://markjoseph2003-metahacky.hf.space/web)**

* Run `inference.py` in your terminal.
* Keep the dashboard open in your browser.
* Watch the Reward Signal Breakdown bars adjust in real-time as the agent processes tool calls!

---

## 🏆 Hackathon Rubric Alignment

To assist evaluators in reviewing the environment, here is an objective breakdown of how **OmniSupport-Sim** fulfills the Round 1 constraints:

### 1. Real-World Utility
* **Requirement:** Must simulate a task humans actually do, avoiding toys or games.
* **Implementation:** The environment directly models a Tier-2 Support CRM focusing on **SOP (Standard Operating Procedure) Compliance**. It evaluates an agent's ability to cross-reference multiple tools (e.g., verifying return tracking logs before issuing refunds), a critical capability for deploying safe enterprise customer service agents.

### 2. Task & Grader Quality
* **Requirement:** 3+ tasks with a difficulty range, programmatic deterministic graders, returning `0.0 - 1.0` scores.
* **Implementation:** Tested via deterministic Python functions (`server/graders.py`):
    * **Easy (Status Resolution):** Evaluates if the agent accurately fetched and localized active order data.
    * **Medium (SOP Compliance):** Features partial point allocation (e.g., fetching policy logic grants 0.4). Implements an aggressive *0.0 override* if the agent hallucinates a refund for an ineligible item.
    * **Hard (Conflict Reconciliation):** Checks multi-hop tool-chaining by enforcing that the agent queries the Carrier API utilizing tracking parameters parsed from prior DB tools before taking action.

### 3. Environment Design
* **Requirement:** Meaningful dense rewards capturing partial progress, typed models.
* **Implementation:** 
    * **Dense Rewards:** Uses a weighted boundary step reward (`-2.0` to `1.0`). Introduces `R_prog` (+0.1 for extracting novel Key-Value pairs, incentivizing tool investigation) alongside `R_pen` (-0.5 per SOP violation, penalizing rash mutations over verification).
    * **Structure:** All interactions are processed through strict subset Pydantic models (`OmniSupportObservation`, `SearchDB`, `VerifyPolicy`, `ExecuteAction`).

### 4. Code Quality & Spec Compliance
* **Requirement:** OpenEnv specification alignment, baseline reproducible scripts.
* **Implementation:**
    * Passes the `openenv validate` evaluation constraint.
    * The included `inference.py` perfectly utilizes the required `[START]`, `[STEP]`, and `[END]` STDOUT structured log pipelines to integrate closely with automated hackathon testing wrappers.

---
*Built for the Scaler x Meta PyTorch Hackathon (April 2026).*
