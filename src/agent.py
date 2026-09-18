import json
import re
from typing import Any

from src.llm import AgentResponse, generate_response
from src.orders import (
    get_order_for_user,
    get_today,
    get_user_orders,
)
from src.retrieval import search_policies


SYSTEM_PROMPT = """
You are an LLM-powered customer support assistant.

Answer the shopper using ONLY the supplied policy context and authorized
order context.

The provided evidence is the only source of truth. Never rely on prior
knowledge about Sezzle.

==================================================
1. GROUNDING
==================================================

- Policy facts must come from POLICY CONTEXT.
- Customer and order facts must come from AUTHORIZED ORDER CONTEXT.
- Never invent fees, dates, limits, eligibility, order details,
  payment states, refunds, reasons, or actions.
- If the evidence is insufficient, clearly say so instead of guessing.
- Never claim an action was completed unless the context explicitly
  states that it happened.

Before answering, identify every material policy rule needed to fully
answer the shopper's question.

Do not omit relevant:
- amounts or percentages
- timing or waiting periods
- fees
- eligibility conditions
- restrictions
- required next steps
- escalation requirements

==================================================
2. AUTHORIZATION
==================================================

Use ONLY order information explicitly present in AUTHORIZED ORDER CONTEXT.

If an order is absent from that context:
- do not infer its details
- do not reveal whether it belongs to another customer
- do not invent information about it

==================================================
3. ROUTING
==================================================

The route describes what evidence and handling are REQUIRED to answer
correctly. It is not based on which source seems most important.

Use:

policy
- The answer can be fully produced from general policy information.
- No customer-specific order facts are required.

tool
- The answer can be fully produced from authorized customer/order facts.
- No policy rule is required to answer correctly.

both
- Correctly answering requires BOTH:
  1. one or more company policy rules, AND
  2. customer-specific or order-specific facts.

IMPORTANT:
If you use an order fact to determine how a policy applies to that
shopper, the route is BOTH, not TOOL.

escalate
- A relevant policy says a human agent must handle the request, OR
- the requested determination/action is reserved for a human.

ESCALATION HAS HIGHEST PRIORITY.

If a policy requires human handling, the route MUST be "escalate"
even when policy information and order information are also used.

The answer may explain relevant policy before escalating.

==================================================
4. FINAL CHECK
==================================================

Before producing the response, verify:

1. Did I rely only on supplied evidence?
2. Did I include the material rules needed to fully answer the question?
3. Did I use both policy and order facts?
   If yes -> route must be "both", unless escalation is required.
4. Does any applicable policy require a human?
   If yes -> route must be "escalate".
5. Did I avoid claiming an unsupported action or result?

Return a concise, helpful, customer-facing answer.
"""


def normalize(text: str) -> str:
    """
    Normalize text for simple merchant-name matching.
    """
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def select_relevant_orders(
    user_id: str,
    question: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Select order context while preserving the authorization boundary.

    Explicit order IDs are resolved only through get_order_for_user().
    Otherwise, merchant names are matched only against orders already
    belonging to the authenticated user.
    """

    # -----------------------------------------
    # 1. Explicit order ID
    # -----------------------------------------

    explicit_order_ids = re.findall(
        r"\bord_\d+\b",
        question.lower(),
    )

    if explicit_order_ids:
        authorized_orders = []

        for order_id in explicit_order_ids:
            order = get_order_for_user(
                user_id,
                order_id,
            )

            if order is not None:
                authorized_orders.append(order)

        if authorized_orders:
            return authorized_orders, ""

        return (
            [],
            "The referenced order is not available for the authenticated user. "
            "Do not reveal whether it exists for another account.",
        )

    # -----------------------------------------
    # 2. Merchant-name matching
    # -----------------------------------------

    normalized_question = normalize(question)
    user_orders = get_user_orders(user_id)

    matched_orders = []

    for order in user_orders:
        merchant = normalize(order["merchant"])

        if merchant and merchant in normalized_question:
            matched_orders.append(order)

    if matched_orders:
        return matched_orders, ""

    # No specific order was safely identified.
    return [], ""


def build_policy_context(
    question: str,
) -> str:
    """
    Retrieve and format relevant policy documents.
    """

    results = search_policies(
        question,
        top_k=3,
    )

    if not results:
        return "No relevant policy context was retrieved."

    blocks = []

    for result in results:
        blocks.append(
            f"""
SOURCE: {result["source"]}
TITLE: {result["title"]}

{result["content"]}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def build_order_context(
    orders: list[dict[str, Any]],
) -> str:
    """
    Serialize only already-authorized orders.
    """

    if not orders:
        return "No authorized order context was selected."

    return json.dumps(
        orders,
        indent=2,
        ensure_ascii=False,
    )


def answer_question(
    question: str,
    user_id: str,
) -> AgentResponse:
    """
    Main agent entry point.
    """

    today = get_today()

    policy_context = build_policy_context(
        question
    )

    relevant_orders, access_note = (
        select_relevant_orders(
            user_id,
            question,
        )
    )

    order_context = build_order_context(
        relevant_orders
    )

    user_prompt = f"""
AUTHENTICATED USER
{user_id}

FROZEN CURRENT DATE
{today}

SHOPPER QUESTION
{question}

POLICY CONTEXT
----------------
{policy_context}

AUTHORIZED ORDER CONTEXT
------------------------
{order_context}

ORDER ACCESS NOTE
-----------------
{access_note if access_note else "No authorization issue detected."}

Using only this evidence, determine the correct route and write the
customer-facing answer.
""".strip()

    return generate_response(
        SYSTEM_PROMPT,
        user_prompt,
    )