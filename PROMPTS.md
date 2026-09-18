# AI Prompts Used

AI tooling was used intentionally for scaffolding, architecture discussion,
debugging, and iteration. I kept final design decisions grounded in measured
behavior from the visible evaluation set.

## 1. Initial architecture

**Tool:** ChatGPT

**Prompt / direction:**

> I have a ~2 hour take-home challenge to build an LLM-powered customer
> support assistant. The system has 12 Markdown policy documents, synthetic
> customer orders, and visible golden cases.
>
> Help me design the smallest architecture that demonstrates grounding,
> authorization, escalation, evaluation, and iteration without overengineering.
> Prefer simple components that can later be justified against a production
> target of 100k questions/day, ~$50/day inference cost, and p95 <= 3 seconds.

**Resulting decision:**  
Use a small Python orchestration layer, local Ollama inference, policy retrieval,
and an authorization boundary around order access rather than introducing a
large agent framework or vector database initially.

---

## 2. Authorization boundary

**Tool:** ChatGPT

**Prompt / direction:**

> Design an order-access layer where the LLM can never receive another user's
> order data. Authorization should be enforced structurally in Python rather
> than relying only on a system prompt.

**Resulting decision:**  
`get_order_for_user(user_id, order_id)` and `get_user_orders(user_id)` filter
data before any order context reaches the model.

---

## 3. Baseline policy retrieval

**Tool:** ChatGPT

**Prompt / direction:**

> Build a minimal policy retriever for a corpus of only 12 Markdown files.
> Start with lexical overlap so I can establish a baseline before adding
> semantic complexity. Do not hardcode the visible golden cases.

**Resulting decision:**  
Implemented weighted lexical matching with stronger weight for policy titles.

---

## 4. Structured local LLM output

**Tool:** ChatGPT

**Prompt / direction:**

> Connect qwen3.5:4b through Ollama and use Pydantic structured output.
> The model should determine whether a request needs policy information,
> order information, both, or human escalation. Keep temperature deterministic.

**Resulting decision:**  
Ollama + Pydantic JSON schema, `temperature=0`, and `think=False` for predictable
local evaluation.

---

## 5. Baseline failure analysis

**Tool:** ChatGPT

**Prompt / direction:**

> Here are the measured results from my first full visible evaluation:
> route accuracy 4/10, required content 14/20, forbidden violations 0.
> Classify the failures instead of fixing individual test cases. Identify
> general routing, retrieval, grounding, and escalation problems.

**Observed failure taxonomy:**

- Order-backed questions were sometimes classified as `tool` even when policy
  interpretation was necessary.
- Human-only workflows were often classified as `both` instead of `escalate`.
- Lexical retrieval missed semantic hardship language.
- Some answers omitted important policy conditions even when the correct
  document was available.

---

## 6. Routing and escalation iteration

**Tool:** ChatGPT

**Prompt / direction:**

> Improve the system prompt using general routing invariants rather than
> hardcoding visible cases. Escalation must take precedence when policy
> explicitly requires human handling. `both` should only apply when both
> policy and customer-specific order facts are materially required.

**Measured result:**  

- Route accuracy: 4/10 -> 7/10
- Required content: 14/20 -> 16/20
- Forbidden violations: 0 -> 0

A second evidence-aware routing iteration reached:

- Route accuracy: 8/10
- Required content: 19/20

---

## 7. Hybrid retrieval experiment

**Tool:** ChatGPT

**Prompt / direction:**

> Lexical retrieval fails to associate natural-language hardship descriptions
> such as job loss with the Hardship Assistance policy. Add semantic retrieval
> using local Ollama embeddings without introducing a vector database.
> Combine lexical and semantic signals and keep the implementation appropriate
> for only 12 documents.

**Resulting decision:**  
Added local `all-minilm` embeddings and hybrid lexical/semantic ranking.

**Observation:**  
The missing hardship policy entered the top-3 retrieval results, but the
end-to-end score did not immediately improve. This showed that document
availability was no longer the only failure mode; model decision consistency
also mattered.

---

## 8. Structured decision decomposition

**Tool:** ChatGPT

**Prompt / direction:**

> Instead of asking the model to directly choose a route, have it independently
> decide whether policy information, authorized order information, and human
> handling are required. Let deterministic application logic translate those
> semantic decisions into the contract route.

**Resulting decision:**  

The LLM returns structured fields such as:

- `requires_policy`
- `requires_order`
- `requires_human`
- applicable policy sources
- customer-visible policy rules

Application code then enforces route precedence.

**Measured final result:**

- Route accuracy: **9/10**
- Required content: **20/20**
- Forbidden content violations: **0**

---

## 9. Stopping criterion

An additional escalation consistency validator was explored after the 9/10 run.
It introduced regressions in unrelated cases, reducing route accuracy.

I rolled that experiment back rather than tuning further against the visible
set. With hidden cases also used for grading, I preferred the more general
9/10 solution over additional visible-case optimization.